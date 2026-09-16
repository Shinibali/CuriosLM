"""
Source of inspiration, Karpathy's implementation: https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

import math
import inspect

import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import GPTConfig # architecture knobs live in config.py, with everything else

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias = config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias = config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):

        B,T,C = x.size()
        head_size = C // self.n_head

        q, k, v = self.c_attn(x).split(self.n_embd, dim=-1) # B,T,ne

        q = q.view(B, T, self.n_head, head_size).transpose(1, 2) # B, nh, T, hs
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2) # B, nh, T, hs
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2) # B, nh, T, hs

        # attn = q @ k.transpose(-2,-1) / math.sqrt(k.size(-1)) # B, nh, T, T
        # attn = attn.masked_fill(self.bias[:,:,:T,:T] == 0, float('inf'))
        # attn = F.softmas(attn, dim=-1)
        # attn = self.attn_dropout(attn)
        # y = attn @ v # B, nh, T, hs

        y = F.scaled_dot_product_attention(q,k,v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True) # B, nh, T, hs

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


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
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
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # set wte weight same as lm_head weight
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        self.apply(self.__init_weights)
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
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def __init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)


    def forward(self, idx, targets=None):

        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"input seq len {t} is larger than model context len {self.config.block_size}"
        pos = torch.arange(0,t,dtype = torch.long, device=device) # (t)

        token_emb = self.transformer.wte(idx) # b,t,ne
        pos_emb = self.transformer.wpe(pos) # t,ne

        x = self.transformer.drop(token_emb + pos_emb) # b,t,ne
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