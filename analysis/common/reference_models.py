"""Model shapes the posts extrapolate to, declared once: no model of these sizes was trained here. They feed the
arithmetic-only scaling scripts (FLOPs, memory, RoPE), which apply the formulas checked on the trained models."""

VOCAB = 50304  # GPT-2's padded vocabulary, kept fixed so only width, depth and context change
GPT2_FAMILY = {  # GPT-2 small/medium/large/XL: layers, width, heads (Radford et al. 2019)
    "GPT-2 small (124M)": (12, 768, 12), "GPT-2 medium (350M)": (24, 1024, 16),
    "GPT-2 large (774M)": (36, 1280, 20), "GPT-2 XL (1.5B)": (48, 1600, 25),
}
SEVEN_B = dict(n_layer=32, n_embd=4096, n_head=32)  # a 7B-class width and depth (head size 128)
FOURTEEN_B = dict(n_layer=48, n_embd=5120, n_head=40)  # a ~14B-class width and depth (head size 128)
LONG_CONTEXT = 131072  # 128k tokens, a common long-context target
