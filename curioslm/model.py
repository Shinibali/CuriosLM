"""
Source of inspiration, Karpathy's implementation: https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

import math
import inspect

import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import GPTConfig # architecture knobs live in config.py, with everything else


def build_rope_cache(head_size, block_size, theta=10000):
    # RoPE Eq 15: https://arxiv.org/pdf/2104.09864 for rotating embedding dimension pair
    # theta_i = rope_theta ** (-2i / head_size)   for i = 0, 1, ..., head_size/2 - 1

    inv_freq = theta ** (-2.0 * torch.arange(head_size//2, dtype=torch.float32) / head_size) # (hs/2)

    # Eq. 34 for cos and sin implementation for (block size m, head_size//2) angles
    # R = x * cos (m*theta) +  x_mod * sin (m*theta)

    m = torch.arange(block_size, dtype=torch.float32) # (T)

    m_theta = torch.outer(m, inv_freq) # (T, hs/2)

    return torch.cos(m_theta), torch.sin(m_theta) # (T, hs/2)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_size, block_size, theta):
        super().__init__()

        cosines, sines = build_rope_cache(head_size, block_size, theta) # (T, hs/2)
        full_cosines = torch.cat((cosines, cosines), dim=-1) # (T, hs)
        full_sines = torch.cat((sines, sines), dim=-1) # (T, hs)

        assert head_size % 2 == 0, "Head_size needs to be even for ROPE to work with pair-wise embedding rotation"

        self.register_buffer("rope_cosines", full_cosines, persistent=False) # (T, hs)
        self.register_buffer("rope_sines", full_sines, persistent=False) # (T, hs)

    @staticmethod
    def _rotate_half(x):
        first, second = torch.chunk(x, 2, dim=-1) # split into 2 chunks
        return torch.cat((-second, first), dim=-1)

    def forward(self, q, k):

        B, nh, T, hs = q.size()
        assert T <= self.rope_cosines.size(0), "Seq len should be smaller than block size"
        
        rope_cosines = self.rope_cosines[:T, :].to(dtype=q.dtype) # (T, hs), change dtype to that of q for optimal compile
        rope_sines = self.rope_sines[:T, :].to(dtype=q.dtype) # (T, hs)

        # note this is different than the paper https://arxiv.org/pdf/2104.09864 interleaved style. 
        # It is "GPT-NeoX style" for (x_i, x_{i+d/2}) instead of (x_{2i-1}, x_{2i}) dimension pair rotation. 
        # Does not change the outcome, unless mixing weights with other interleaved/"GPT-J style" models from HF
        q_rotated = q * rope_cosines + self._rotate_half(q) * rope_sines # B, nh, T, hs
        k_rotated = k * rope_cosines + self._rotate_half(k) * rope_sines # B, nh, T, hs

        return q_rotated, k_rotated

# The nn.RMSNorm dircet usage was throwing error with dtype mismatch, hence this wrapper to take care of dtypes
class RMSNormFP32(nn.Module):
    def __init__(self, n_embd, eps=1e-5):
        super().__init__()
        self.norm = nn.RMSNorm(n_embd, eps=eps)

    def forward(self, x):
        return self.norm(x.float()).type_as(x)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.num_kv_heads = config.num_kv_heads
        self.head_size = self.n_embd // self.n_head
        assert config.n_head % config.num_kv_heads == 0

        self.enable_gqa = self.num_kv_heads != self.n_head
        self.kv_size = self.num_kv_heads * self.head_size

        # for the c_attn out, Q stays as is for each head + 2 is for K and V weights
        self.c_attn = nn.Linear(config.n_embd, config.n_embd + 2 * self.kv_size, bias = config.bias) 
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias = config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.rope = RotaryEmbedding(self.head_size, config.block_size, config.rope_theta) if config.use_rope else None

        # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):

        B,T,C = x.size()

        q,k,v = self.c_attn(x).split([self.n_embd, self.kv_size, self.kv_size], dim=-1) # (B,T, ne), 2(B,T, kv_size)
        
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2) # B, nh, T, hs
        k = k.view(B, T, self.num_kv_heads, self.head_size).transpose(1, 2) # B, nkvh, T, hs
        v = v.view(B, T, self.num_kv_heads, self.head_size).transpose(1, 2) # B, nkvh, T, hs

        if self.rope is not None:
            q, k = self.rope(q,k) # rotated q, k for every (x_i, x_{i+d/2}) dimension pair

        # attn = q @ k.transpose(-2,-1) / math.sqrt(k.size(-1)) # B, nh, T, T
        # attn = attn.masked_fill(self.bias[:,:,:T,:T] == 0, float('inf'))
        # attn = F.softmax(attn, dim=-1)
        # attn = self.attn_dropout(attn)
        # y = attn @ v # B, nh, T, hs

        y = F.scaled_dot_product_attention(q,k,v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True, enable_gqa=self.enable_gqa) # B, nh, T, hs

        y = y.transpose(1, 2).contiguous().view(B,T,C) 

        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GeGLU(nn.Module):

    def __init__(self, config):
        super().__init__()
        # In: ne, Out: 4ne * 2 for the gating + val of GeGLU
        n_hidden = int(2/3 * 4 * config.n_embd) # this is my special construct to keep FFN # params comparable to GPT2 FFN # params
        self.c_fc = nn.Linear(config.n_embd, n_hidden * 2 , bias=config.bias) # n -> 8n/3 * 2
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(n_hidden, config.n_embd, bias=config.bias) # 8n/3 -> n
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        gate, val = self.c_fc(x).chunk(2, dim=-1) # n -> 8n/3 * 2 -> 8n/3
        x = self.gelu(gate) * val # 8n/3 -> 8n/3
        x = self.c_proj(x) # 8n/3 -> n
        x = self.dropout(x) # n -> n
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config) if config.use_gelu_activation else GeGLU(config)
        self.use_sandwich_norm = config.use_sandwich_norm

        if self.use_sandwich_norm:
            self.ln_1 = RMSNormFP32(config.n_embd, eps=1.0e-5)
            self.ln_2 = RMSNormFP32(config.n_embd, eps=1.0e-5)
            self.ln_3 = RMSNormFP32(config.n_embd, eps=1.0e-5)
            self.ln_4 = RMSNormFP32(config.n_embd, eps=1.0e-5)
        else:
            self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
            self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, x):

        if self.use_sandwich_norm:
            x = x + self.ln_2(self.attn(self.ln_1(x)))
            x = x + self.ln_4(self.mlp(self.ln_3(x)))
        else:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))

        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        if config.use_absolute_pos:
            self.transformer['wpe'] = nn.Embedding(config.block_size, config.n_embd)

        if config.use_sandwich_norm:
            self.transformer["ln_f"] = RMSNormFP32(config.n_embd, eps=1.0e-5)
        else:
            self.transformer["ln_f"] = nn.LayerNorm(config.n_embd, bias=config.bias)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # set wte weight same as lm_head weight
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std = 0.02/math.sqrt(2*config.n_layer))

        print(f"number of parameters: {self.get_num_params()/1e6:0.2f}M")

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self.transformer, 'wpe'):
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)


    def forward(self, idx, targets=None):

        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"input seq len {t} is larger than model context len {self.config.block_size}"

        token_emb = self.transformer.wte(idx) # b,t,ne

        if self.config.use_absolute_pos:
            pos = torch.arange(0,t,dtype = torch.long, device=device) # (t)
            pos_emb = self.transformer.wpe(pos) # t,ne
            x = self.transformer.drop(token_emb + pos_emb) # b,t,ne
        else:
            x = self.transformer.drop(token_emb) # b,t,ne

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x) # b,t,ne

        loss = None

        if targets is not None:
            logits = self.lm_head(x) # b,t,vs
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) # (1)
        else:
            logits = self.lm_head(x[:,[-1],:]) # (b,1,vs) inference-time mini-optimization: only forward the lm_head on the very last position using list [-1] to preserve the time dim

        return logits, loss


    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
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


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):

        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n,p in param_dict.items() if p.dim() >= 2]
        nondecay_params = [p for n,p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nondecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nondecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nondecay_params)}, with {num_nodecay_params:,} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        optimizer = torch.optim.AdamW(optim_groups, lr = learning_rate, betas=betas, **extra_args)

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt, flops_promised=312e12):
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311 shows flops_per_token = 6*N + 12*L*H*Q*T
        # flops_promised is the hardware's dense peak; the caller passes the one
        # matching the actual device (default is A100 bfloat16, 312 TFLOPS)

        N = self.get_num_params()
        cfg = self.config
        L,H,Q,T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size

        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        flops_achieved = flops_per_iter / dt
        mfu = flops_achieved/flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):

        for _ in range(max_new_tokens):

            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:] # (1,t)

            logits, _ = self(idx_cond) # (1,1,vs)
            logits = logits[:,-1,:] / temperature # 1,vs
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1))) # 1,k
                logits[logits < v[:, [-1]]] = -float('Inf') # everything smaller than the kth largest value is set to -inf, as v[:,-1] gives the kth largest value from the original distr

            probs = F.softmax(logits, dim=-1) # 1,vs
            idx_next = torch.multinomial(probs, num_samples=1) # 1,1
            idx = torch.cat((idx, idx_next), dim=1)

        return idx