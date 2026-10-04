"""The wandb run paths ("project/run_id") the analyses refer to — the one place they are typed. Everything else about
a run (out_dir, dataset, batch settings, GPU count) is fetched from wandb, never retyped."""

LADDER_A0 = "curioslm-ladder/mbwli7wj"          # 124M GPT-2, 1.5B tokens
LADDER_A4 = "curioslm-ladder/kpqh0522"          # 113M Gemma-style, 1.5B tokens
LADDER_A4_SEED2 = "curioslm-ladder/ilwem9b9"    # same rung, seed 1338
HEADLINE_A = "curioslm-headline/fp1ka2zx"       # Model A: 124M GPT-2, 9.8B tokens
HEADLINE_B = "curioslm-headline/2mtqioaa"       # Model B: 113M Gemma-style, 9.8B tokens
