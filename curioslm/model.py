"""
Source of inspiration, Karpathy's implementation: https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

import math
import inspect

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from .config import GPTConfig


YARN_ALPHA, YARN_BETA = 1, 32  # YaRN band thresholds https://arxiv.org/abs/2309.00071

def yarn_frequencies(inv_freq, factor, original_context):
    # YaRN 'NTK-by-parts': fast bands kept, slow bands divided by the factor, a linear blend in between.

    turns = original_context * inv_freq / (2 * math.pi)  # r = L / λ: full turns each band makes in original context
    keep = ((turns - YARN_ALPHA) / (YARN_BETA - YARN_ALPHA)).clamp(0, 1)  # γ(r): 1 = fast, kept; 0 = slow, interpolated
    return keep * inv_freq + (1 - keep) * inv_freq / factor  # h(θ) = γ·θ + (1 − γ)·θ/s

def yarn_temperature(factor):
    # √(1/t) from YaRN Eq. 22, applied to both q and k so q·k scales by 1/t jointly.
    return 0.1 * math.log(factor) + 1  # 1.139 at s = 4

    
def build_rope_cache(head_size, block_size, theta=10000, yarn_factor=1.0, yarn_original_context=None):
    # RoPE Eq 15: https://arxiv.org/pdf/2104.09864 for rotating embedding dimension pair
    # theta_i = rope_theta ** (-2i / head_size)   for i = 0, 1, ..., head_size/2 - 1

    inv_freq = theta ** (-2.0 * torch.arange(head_size//2, dtype=torch.float32) / head_size) # (hs/2)

    temperature = 1.0  # for YARN
    if yarn_factor != 1.0:
        inv_freq = yarn_frequencies(inv_freq, yarn_factor, yarn_original_context)  # (hs/2)
        temperature = yarn_temperature(yarn_factor)
                                    
    # Eq. 34, R = x * cos(m*theta) + x_mod * sin(m*theta), over (block_size, head_size//2) angles
    m = torch.arange(block_size, dtype=torch.float32) # (T)

    m_theta = torch.outer(m, inv_freq) # (T, hs/2)

    return temperature * torch.cos(m_theta), temperature * torch.sin(m_theta) # (T, hs/2)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_size, block_size, theta, yarn_factor=1.0, yarn_original_context=None):
        super().__init__()

        cosines, sines = build_rope_cache(head_size, block_size, theta, yarn_factor, yarn_original_context) # (T, hs/2)
        full_cosines = torch.cat((cosines, cosines), dim=-1) # (T, hs)
        full_sines = torch.cat((sines, sines), dim=-1) # (T, hs)

        assert head_size % 2 == 0, "Head_size needs to be even for ROPE to work with pair-wise embedding rotation"

        self.register_buffer("rope_cosines", full_cosines, persistent=False) # (T, hs)
        self.register_buffer("rope_sines", full_sines, persistent=False) # (T, hs)

    @staticmethod
    def _rotate_half(x):
        first, second = torch.chunk(x, 2, dim=-1) # split into 2 chunks
        return torch.cat((-second, first), dim=-1)

    def forward(self, q, k, start=0):

        B, nh, T, hs = q.size()        
        assert start + T <= self.rope_cosines.size(0), "Seq len should be smaller than block size"
        
        rope_cosines = self.rope_cosines[start:start+T, :].to(dtype=q.dtype) # (T, hs), match q's dtype for compile
        rope_sines = self.rope_sines[start:start+T, :].to(dtype=q.dtype) # (T, hs)

        # "GPT-NeoX style" pairing (x_i, x_{i+d/2}) rather than the paper's interleaved
        # (x_{2i-1}, x_{2i}). Same outcome, unless mixing weights with interleaved HF models.
        q_rotated = q * rope_cosines + self._rotate_half(q) * rope_sines # B, nh, T, hs
        k_rotated = k * rope_cosines + self._rotate_half(k) * rope_sines # B, nh, T, hs

        return q_rotated, k_rotated

# Wrapper: nn.RMSNorm used directly throws on dtype mismatch under autocast.
class RMSNormFP32(nn.Module):
    def __init__(self, n_embd, eps=1e-5):
        super().__init__()
        self.norm = nn.RMSNorm(n_embd, eps=eps)

    def forward(self, x):
        return self.norm(x.float()).type_as(x)

def local_causal(q_idx, kv_idx, window):
    # think of k_idx falling in this range: q_idx - window < k_idx <= q_idx
    causal = kv_idx <= q_idx
    windowed = q_idx - window < kv_idx # at most window-1 positions back
    return causal & windowed

class KVCache:
    def __init__(self, n_layer):
        self.k = [None] * n_layer # per layer (B, nkvh, T_stored, hs) or None
        self.v = [None] * n_layer
        self.pos = 0 # absolute pos of next token

    def seq_lens(self):
        return [0 if k is None else k.shape[2] for k in self.k]  # stored positions per layer (T axis; 0 if empty)
    
class CausalSelfAttention(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.layer_idx = layer_idx
        self.is_global_layer = config.is_global_layer(layer_idx)

        if not self.is_global_layer:
            # Local layers need an explicit causal+window mask. Global layers skip this
            # entirely and use SDPA's native is_causal=True fast path 
            idx = torch.arange(config.block_size)
            # idx.unsqueeze(1) reshapes idx from (block_size,) to (block_size, 1) -- q
            # idx.unsqueeze(0) reshapes to (1, block_size) -- k. Comparing them triggers broadcasting:
            self.attention_window = config.attention_window
            self.register_buffer("window_mask", local_causal(idx.unsqueeze(1), idx.unsqueeze(0), 
                                                             config.attention_window), persistent=False)
            self._block_masks = {}

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.num_kv_heads = config.num_kv_heads
        self.head_size = self.n_embd // self.n_head
        assert config.n_head % config.num_kv_heads == 0

        self.enable_gqa = self.num_kv_heads != self.n_head
        self.kv_size = self.num_kv_heads * self.head_size

        # Q keeps full width; the +2 is K and V, which are narrower under GQA
        self.c_attn = nn.Linear(config.n_embd, config.n_embd + 2 * self.kv_size, bias = config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias = config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.rope = RotaryEmbedding(self.head_size, config.block_size, config.rope_theta,
                                    config.yarn_factor, config.yarn_original_context) if config.use_rope else None 
        self.capture_attention = False  # eval only: compute attention explicitly and keep its probabilities
        self.last_attention = None  # (B, n_head, T, T), from the latest forward with capture_attention on
        
    def _flex_block_mask(self, T, device):
        key = (T, str(device))
        if key not in self._block_masks:
            window = self.attention_window
            # FlexAttention calls this per (q_idx, kv_idx) position pair to find whole blocks it can skip. 
            # Batch (b) and head (h) don't matter for a sliding window.
            self._block_masks[key] = create_block_mask(
                lambda b, h, q_idx, kv_idx: local_causal(q_idx, kv_idx, window),
                None, None, T, T, device=device)
        return self._block_masks[key]

    def forward(self, x, kv_cache=None):

        B,T,C = x.size()
        start = kv_cache.pos if kv_cache is not None else 0  # absolute position of the first new token

        q,k,v = self.c_attn(x).split([self.n_embd, self.kv_size, self.kv_size], dim=-1) # (B,T, ne), 2(B,T, kv_size)

        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2) # B, nh, T, hs
        k = k.view(B, T, self.num_kv_heads, self.head_size).transpose(1, 2) # B, nkvh, T, hs
        v = v.view(B, T, self.num_kv_heads, self.head_size).transpose(1, 2) # B, nkvh, T, hs

        if self.rope is not None:
            q, k = self.rope(q,k, start) # rotated q, k for every (x_i, x_{i+d/2}) dimension pair

        if kv_cache is not None:
            past_k = kv_cache.k[self.layer_idx]
            if past_k is not None:
                k = torch.cat([past_k, k], dim = 2) # dim T in (B, nkvh, T, hs)
                v = torch.cat([kv_cache.v[self.layer_idx], v], dim = 2)  # dim T in (B, nkvh, T, hs)
            kv_cache.k[self.layer_idx] = k
            kv_cache.v[self.layer_idx] = v

        use_flex_attn = q.is_cuda and torch.compiler.is_compiling() and not (self.training and self.dropout > 0)

        if kv_cache is not None:
            # cached path (generation) T new queries attend to every stored key
            T_stored  = k.shape[2]  # keys stored after the concat       
            first_pos = (start + T) - T_stored  # absolute position of the OLDEST stored key 
            k_pos = torch.arange(first_pos, start + T, device=x.device)  # positions of stored keys (T_stored,)
            q_pos = torch.arange(start, start + T, device=x.device)  # absolute positions of the new queries (T,)

            if self.is_global_layer:  
                mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)  # causal mask (T, T_stored) may query see key? 
            else:
                mask = local_causal(q_pos.unsqueeze(1), k_pos.unsqueeze(0), self.attention_window)  # causal + window
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=self.enable_gqa) # B, nh, T, hs

            if not self.is_global_layer: # drop keys no future queries can see
            # the next query (position start + T) sees keys in (start + T − window, start + T]; one of those is its own,
            # added when it arrives, so it needs at most window − 1 of the keys stored now
                keep = self.attention_window - 1
                kv_cache.k[self.layer_idx] = k[:,:,-keep:]
                kv_cache.v[self.layer_idx] = v[:,:,-keep:]

        elif self.capture_attention:
            # explicit softmax(q·kᵀ/√h + mask): the fused kernels below do not expose the probabilities
            assert not self.training, "capture_attention is for inspection; it skips dropout"
            y, self.last_attention = self._attention_with_weights(q, k, v) # y (B, n_head, T, hs), attn (B, n_head, T, T)

        elif self.is_global_layer: 
            # no cache (training / eval) unchanged
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0, enable_gqa=self.enable_gqa) # B, nh, T, hs

        elif use_flex_attn: # works only in cuda for both forward/backward. in mps, only forward works
            y = flex_attention(q,k,v, block_mask=self._flex_block_mask(T, q.device), enable_gqa=self.enable_gqa)
        else: # this is for local attention that works during training on mps for both forward and backward
            window_mask = self.window_mask[:T, :T] # (T, T), sliced down from the block_size buffer
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=window_mask, dropout_p=self.dropout if self.training else 0, enable_gqa=self.enable_gqa) # B, nh, T, hs

        y = y.transpose(1, 2).contiguous().view(B,T,C)

        y = self.resid_dropout(self.c_proj(y))
        return y


    def _attention_with_weights(self, q, k, v):
        """Attention computed step by step, for inspection.
        q: (B, n_head, T, hs); k, v: (B, n_kv_head, T, hs), already rotated by RoPE when the model uses it.
        Returns y (B, n_head, T, hs) and the attn = probabilities (B, n_head, T, T): row t is query t's distribution
        over the keys it may attend to."""

        # GQA: give every query head the key/value head (B, n_kv_head, T, hs) it shares
        group = self.n_head // self.num_kv_heads  # query heads per key/value head
        k = k.repeat_interleave(group, dim=1)  # [k0, k1, k2] -> [k0, k0, k0, k0, k1, ...]
        v = v.repeat_interleave(group, dim=1)  # same as k
        # now k, v (B, n_head, T, hs) like q

        attn = q @ k.transpose(-1,-2) / math.sqrt(k.size(-1))  # (B, n_head, T, T)

        T = q.size(-2)  # q is (B, n_head, T, hs)
        if self.is_global_layer:
            positions = torch.arange(T, device=q.device)
            allowed = positions[None, :] <= positions[:, None]  # (T, T) key col position <= query row position
        else:
            allowed = self.window_mask[:T, :T]  # the causal + window buffer
        attn = attn.masked_fill(allowed == 0, float('-inf'))  # (B, n_head, T, T) mask where true so we put −∞ on upper non-causal triangle or other relevant segments of local mask

        attn = attn.softmax(dim=-1)  # (B, n_head, T, T)
        return attn @ v, attn  # y (B, n_head, T, hs), attn (B, n_head, T, T)


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
    def __init__(self, config, layer_idx):
        super().__init__()

        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config) if config.use_gelu_activation else GeGLU(config)
        self.use_sandwich_norm = config.use_sandwich_norm

        if self.use_sandwich_norm:
            self.ln_1 = RMSNormFP32(config.n_embd)
            self.ln_2 = RMSNormFP32(config.n_embd)
            self.ln_3 = RMSNormFP32(config.n_embd)
            self.ln_4 = RMSNormFP32(config.n_embd)
        else:
            self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
            self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, x, kv_cache=None):

        if self.use_sandwich_norm:
            x = x + self.ln_2(self.attn(self.ln_1(x), kv_cache=kv_cache))
            x = x + self.ln_4(self.mlp(self.ln_3(x)))
        else:
            x = x + self.attn(self.ln_1(x), kv_cache=kv_cache)
            x = x + self.mlp(self.ln_2(x))

        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        if config.yarn_factor != 1.0:
            assert config.use_rope, " YARN rescales ROPE bands so this run needs ROPE inclusion"
            assert config.yarn_original_context, "yarn_factor needs yarn_original_context training length"
            
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        ))
        if config.use_absolute_pos:
            self.transformer['wpe'] = nn.Embedding(config.block_size, config.n_embd)

        if config.use_sandwich_norm:
            self.transformer["ln_f"] = RMSNormFP32(config.n_embd)
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
        """Parameter count. The non-embedding default subtracts position embeddings but keeps
        token embeddings, which weight tying reuses as the final layer."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self.transformer, 'wpe'):
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    @torch.no_grad()
    def attention_weights(self, idx):
        """Every layer's attention probabilities for idx: a list with one (B, n_head, T, T) tensor per layer, from the
        explicit path. Each layer's switch is turned back off afterwards, so the model's normal forward is unchanged."""
        assert not self.training, "attention_weights is for inspection: call model.eval() first"
        layers = [block.attn for block in self.transformer.h]
        for layer in layers:
            layer.capture_attention = True
        try:
            self(idx)
            weights = [layer.last_attention for layer in layers] # attn (B, n_head, T, T)
        finally:
            for layer in layers:
                layer.capture_attention, layer.last_attention = False, None
        return weights # attn (B, n_head, T, T)


    def new_kv_cache(self, batch_size, device):
        return KVCache(self.config.n_layer)  # one slot per layer; batch_size/device unused while the cache grows by cat

    def forward(self, idx, targets=None, kv_cache=None):

        device = idx.device
        b, t = idx.size() # t is num of new tokens

        start = kv_cache.pos if kv_cache is not None else 0 # pos of the first new
        assert start + t <= self.config.block_size, \
            f"input seq len {start + t} is larger than model context len {self.config.block_size}"

        token_emb = self.transformer.wte(idx) # b,t,ne

        if self.config.use_absolute_pos:
            pos = torch.arange(start,start+t,dtype = torch.long, device=device) # (t)
            pos_emb = self.transformer.wpe(pos) # t,ne
            x = self.transformer.drop(token_emb + pos_emb) # b,t,ne
        else:
            x = self.transformer.drop(token_emb) # b,t,ne

        for block in self.transformer.h:
            x = block(x, kv_cache=kv_cache)
        if kv_cache is not None:
            kv_cache.pos += t
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
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_cache=True):

        block = self.config.block_size
        kv_cache, step_input = None, None

        for _ in range(max_new_tokens):

            if use_cache and idx.size(1) <= block: # cache as long as seq len is within context length
                if kv_cache is None:  # first step: prefill the whole prompt once
                    kv_cache = self.new_kv_cache(batch_size=idx.size(0), device=idx.device)
                    step_input = idx
                logits, _ = self(step_input, kv_cache=kv_cache) # only the new tokens after prefill
            else:                
                # no cache, or the sequence outgrew block_size: recompute on the last block_size tokens
                logits, _ = self(idx[:, -block:]) # (1,1,vs)

            logits = logits[:,-1,:] / temperature # 1,vs
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1))) # 1,k
                logits[logits < v[:, [-1]]] = -float('Inf') # everything smaller than the kth largest value is set to -inf, as v[:,-1] gives the kth largest value from the original distr

            probs = F.softmax(logits, dim=-1) # 1,vs
            idx_next = torch.multinomial(probs, num_samples=1) # 1,1
            idx = torch.cat((idx, idx_next), dim=1)
            step_input = idx_next # the next cached step feeds only this one token

        return idx