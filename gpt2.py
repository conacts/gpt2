import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass

@dataclass
class GPT2Config:
    block_size: int = 1024 # max context size
    vocab_size: int = 50257 # vocab size || 50,000 BPE merges + 256 chars + 1 <|endoftext|> token = 50257
    n_layers: int = 12 # num layers
    n_head: int = 12 # num heads
    n_embd: int = 768 # embedding dimensions

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads in batches
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projections
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1.0

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # mask in hf & openAI naming
        self.register_buffer('bias', 
                             torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, 1, config.block_size, config.block_size)
                        )

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimension (n_embd)
        # calc values for key, query and value for all heads in batch
        # nh = num of heads, hs = head size, C = num of channels = nh * hs
        # GPT2 = n_head = 12, hs = 64, so nh * hs = C = 12 * 64 = 768 channels in the transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attn (materializes the large (T,T) matrix for all queries and keys)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) = (B, nh, T, hs)
        
        # flash attn
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # reassemble all head outputs side by side
        y = self.c_proj(y) # output projection
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte =  nn.Embedding(config.vocab_size, config.n_embd), # token emb in
            wpe =  nn.Embedding(config.block_size, config.n_embd), # pos emb
            h =  nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.n_embd),
            ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, 'Cannot forward, model block size is exhausted.'
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # shape    (T, n_embd)
        tok_emb = self.transformer.wte(idx) # shape (B, T, n_embd)
        x = tok_emb + pos_emb # shape (B, T, n_embd) broadcasted addition of both embeddings
        # forward the block of the transformer
        for block in self.transformer.h:
            x = block(x)
        # apply the final layer norm
        x = self.transformer.ln_f(x)
        # project back to vocabulary
        logits = self.lm_head(x) # shape (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f'loading weights from pretrained gpt: {model_type}')

        config_args = {
            'gpt2':         dict(n_layers=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layers=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layers=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layers=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257 # always
        config_args['block_size'] = 1024 # always
        # create from scratch a min-GPT model with the same architecture
        config = GPT2Config(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard the mask / buffer
        
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # make sure all the keys align
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys) == len(sd_hf), f'mismatched keys: {len(sd_keys)} != {len(sd_hf)}'
        for k in sd_keys_hf:
            print(k)
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# ---------------------------------
num_return_sequences = 5
max_length = 50

device = 'cpu'
torch.manual_seed(42)
'''
torch.mps.manual_seed(42)
if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.manual_seed_all(42)
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
'''

print(f'Using device {device}')

import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        enc = tiktoken.get_encoding('gpt2')
        with open('data/tiny.txt', 'r') as f:
            text = f.read()
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens).to(device)
        self.pos = 0
        print('num of tokens', len(self.tokens))

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.pos:self.pos+B*T+1]
        x = (buf[:-1]).view(B,T)
        y = (buf[1:]).view(B,T)
        self.pos += B*T
        if self.pos + (B*T+1) > len(self.tokens):
            self.pos = 0
        return x, y


model = GPT(GPT2Config(vocab_size=50304))
model.to(device)
model = torch.compile(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
train_loader = DataLoaderLite(B=4, T=1024)

torch.set_float32_matmul_precision('medium')

import time
for i in range(50):
    x, y = train_loader.next_batch()
    t0 = time.time()
    optimizer.zero_grad()
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    torch.mps.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f'iter {i} loss: {loss.item()}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}')
