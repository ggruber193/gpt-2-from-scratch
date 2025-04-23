import inspect
import math
from typing import Literal
from enum import Enum

import torch
from torch import nn
from dataclasses import dataclass
from torch.nn import functional as F

@dataclass
class GPT2Config:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257 # number of tokens
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    feedforward_dim: int = 4 * n_embd

    def get(self, key):
        return self.__dict__.get(key, None)

class GPT2ConfigPretrained(GPT2Config):
    def __init__(self, model_type: Literal["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]):
        model_configs = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }
        assert model_type in model_configs
        model_config = model_configs[model_type]
        super().__init__(**model_config)

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0  # make sure embedding dimension matches number of heads

        # key, query, value in batch
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)

        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_embd = config.n_embd
        self.n_head = config.n_head

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        batch_size, sequence_length, embedding_dim = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(batch_size, sequence_length, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.n_head, self.n_embd // self.n_head).transpose(1, 2)

        # Replace our version of attention with optimized flash attention
        """attn = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(k.size(-1)))
        attn = attn.masked_fill(self.bias[:, :, :sequence_length, :sequence_length] == 0, -torch.inf)
        attn = F.softmax(attn, dim=-1)
        y = attn @ v"""
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(batch_size, sequence_length, embedding_dim)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.feedforward_dim)
        self.gelu = nn.GELU(approximate="tanh")   # approximation is historical artifact because exact version was slow in tensorflow therefore approximation was used
        self.c_proj = nn.Linear(config.feedforward_dim, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.gelu(self.c_fc(x))
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing of wte and lm_head
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, 0, std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, 0, 0.02)

    def forward(self, idx, targets=None):
        batch_size, sequence_length = idx.shape
        assert sequence_length <= self.config.block_size

        pos = torch.arange(0, sequence_length, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
            return logits, loss
        return logits


    def configure_optimizers(self, weight_decay=0.1, learning_rate=6e-4):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        print(f"Number of decayed parameter tensors: {len(decay_params)}")
        print(f"Number of non-decayed parameter tensors: {len(nodecay_params)}")

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True)
        return optimizer


    @classmethod
    def from_pretrained(cls, model_type: Literal["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]):
        from transformers import GPT2LMHeadModel
        config = GPT2ConfigPretrained(model_type)
        model = GPT2(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}\n{set(sd_keys_hf).symmetric_difference(set(sd_keys))}"
        for k in sd_keys_hf:
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

if __name__ == '__main__':
    model = GPT2.from_pretrained("gpt2")
    print(model)
