"""
Every tunable in the project, declared exactly once.

    GPTConfig      architecture — the only place n_layer/n_embd/... are defined
    DataConfig     which dataset to read
    IOConfig       where checkpoints go, how often to evaluate, wandb
    OptimConfig    optimizer and LR schedule
    RuntimeConfig  device, precision, compile, seed
    SampleConfig   generation settings (sample.py only)

Scripts flatten the groups they need into module globals with `flatten()`, keeping
`configurator.py`'s override mechanism working unchanged — it overrides plain globals, so
these dataclasses are a declaration layer, not a runtime object graph.
"""

from dataclasses import dataclass, asdict, fields
from typing import Literal

@dataclass
class GPTConfig:
    """Model architecture. Consumed by model.GPT and the source of `checkpoints.CONFIG_KEYS` —
    a knob added here propagates to the checkpoint format and resume path automatically."""
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 4#12
    n_head: int = 4#12
    n_embd: int = 124#768
    dropout: float = 0.0 # 0 is good for pretraining, try 0.1+ for finetuning
    bias: bool = True # bias in Linears and LayerNorm; True matches nanoGPT's GPT-2 default

    position_encoding: Literal["absolute", "rope"] = "absolute"
    rope_theta: float = 10000 # from paper Eq 15: https://arxiv.org/pdf/2104.09864
    yarn_factor: float = 1.0 # YaRN context stretch s https://arxiv.org/abs/2309.00071; 1.0 = plain RoPE
    yarn_original_context: int | None = None # the training length the stretch starts from; required when s != 1
    
    n_kv_head: int | None = None # None = regular MHA (n_kv_head == n_head)

    mlp_type: Literal["gelu", "geglu"] = "gelu"
    norm_placement: Literal["pre", "sandwich"] = "pre"

    attention_window: int | None = None  # None means full global attention, no windowing
    local_global_ratio: int = 5  # 1 in every N layers is global; Gemma 3's convention

    def is_global_layer(self, layer_idx: int) -> bool:
        return self.attention_window is None or (layer_idx % self.local_global_ratio == 0)

    # Other modules read these rather than comparing the raw strings themselves.
    @property
    def use_rope(self) -> bool:
        return self.position_encoding == "rope"

    @property
    def use_absolute_pos(self) -> bool:
        return not self.use_rope

    @property
    def num_kv_heads(self) -> int:
        return self.n_kv_head or self.n_head

    @property
    def use_gelu_activation(self) -> bool:
        return self.mlp_type == "gelu"

    @property
    def use_sandwich_norm(self) -> bool:
        return self.norm_placement == "sandwich"


@dataclass
class DataConfig:
    dataset: str = 'fineweb_edu' # directory under data/; also what hf_shards.py uploads/downloads
    hf_dataset_repo_id: str = 'CuriosLM/fineweb_edu-shards' # private repo staging shards between
                                                            # prepare.py (local) and GPU training
    shard_size: int = 100_000_000 # tokens/shard -- NOT independent of world_size*batch_size*block_size
    data_mix: dict | None = None # {dataset: weight}: train on a deterministic blend; None trains on `dataset` alone
    hf_dataset_repo_template: str = 'CuriosLM/{}-shards' # shard repo for any dataset, by name
    long_pages: bool = False # every sequence is one page of block_size+ tokens (its first block_size): long-context
                             # fine-tuning, so no window joins short pages; see loaders/long_page_loader.py


@dataclass
class IOConfig:
    out_dir: str = 'out'
    init_from: str = 'scratch' # 'scratch' or 'resume' or 'extend' or 'gpt2*'
    extend_from: str = '' # init_from='extend': out_dir of the finished run to continue; this run writes to out_dir
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False # if True, exit right after the first eval
    always_save_checkpoint: bool = True # if True, save after every eval, not just on improvement
    snapshot_interval: int = 0 # >0: also keep a weights-only ckpt-<iter>.pt every this many steps of this run
    wandb_log: bool = False
    wandb_project: str = 'owt'
    wandb_run_name: str = 'gpt2'
    hf_checkpoint_repo_id: str = 'CuriosLM/checkpoints-staging' # private repo each run's ckpt.pt is
                                                                # auto-uploaded to, under out_dir --
                                                                # ephemeral pod disk is never the
                                                                # only copy


@dataclass
class OptimConfig:
    # batching
    gradient_accumulation_steps: int = 5 * 8 # simulates a larger batch
    batch_size: int = 12 # micro-batch, if gradient_accumulation_steps > 1
    # adamw
    learning_rate: float = 6e-4
    max_iters: int = 10000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0 # clip gradients at this value, or disable if == 0.0
    # cosine schedule with linear warmup
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = -1 # -1 = follow max_iters (per Chinchilla); resolved after overrides
    min_lr: float = 6e-5 # ~= learning_rate/10 per Chinchilla
    # ddp
    backend: str = 'nccl' # 'nccl', 'gloo', etc.


@dataclass
class RuntimeConfig:
    device: str = '' # '' = auto-detect cuda/mps/cpu
    dtype: str = '' # '' = best this device supports; else float32/bfloat16/float16
    compile: bool = True # torch.compile: big win on CUDA, slow to warm up on MPS
    seed: int = 1337
    peak_flops_override: float = 0.0 # MFU denominator; 0 = look it up by device


@dataclass(frozen=True)
class YarnStudy:
    """The YaRN context-extension study's choices, read by config/yarn_finetune.py and analysis/positions/."""
    factor: float = 4.0 # stretch Model B's context 4x. The YaRN paper's zero-shot results hold to ~2x, so at 2x a
                        # fine-tune has little to add; 8x is past zero-shot range for 124M, and only ~1% of val
                        # pages reach 8k tokens (~3% reach 4k)
    finetune_steps: int = 200 # 200 X 4096 T X 15 B X 8 grad_accum ~98M tokens, 1% of pretraining; no established budget, so snapshots measure it
    snapshot_interval: int = 25 # steps between weights-only snapshots: 8 points on the effect-vs-steps curve


@dataclass
class SampleConfig:
    start: str = "\n" # prompt, or "FILE:prompt.txt"
    num_samples: int = 10
    max_new_tokens: int = 500
    temperature: float = 0.8 # <1.0 = less random, >1.0 = more random
    top_k: int = 200 # keep only the top_k most likely tokens


def flatten(*configs):
    """Merge config groups into one dict of plain values, for `globals().update(...)`.
    Raises on a duplicate key across groups: each setting gets exactly one home."""
    merged = {}
    for cfg in configs:
        for key, value in asdict(cfg).items():
            if key in merged:
                raise ValueError(
                    f"config key {key!r} declared in two groups; each setting gets one home"
                )
            merged[key] = value
    return merged


def field_names(cfg_cls):
    return tuple(f.name for f in fields(cfg_cls))
