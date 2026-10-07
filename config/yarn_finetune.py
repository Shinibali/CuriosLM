# YaRN fine-tune: continue Model B's headline run (a4, 9.8B tokens) at YarnStudy.factor x its context. Every setting
# is the headline run's -- ladder_gpu.py is exec'd with its rung and budget -- except the ones changed below: the
# context, YaRN, the data (single long pages) and the run's length, naming and cadence. The study's choices live in
# curioslm/config.py (YarnStudy); this file only applies them.
#
#   uv run python train.py config/yarn_finetune.py --device=cuda                          # 1 GPU
#   uv run torchrun --standalone --nproc_per_node=4 train.py config/yarn_finetune.py      # any count dividing 8

import os

from curioslm.config import YarnStudy

for key in ('LADDER_MIX', 'LADDER_SEED', 'LADDER_DATASET'):  # the headline run had none of these
    os.environ.pop(key, None)
os.environ['LADDER_RUNG'], os.environ['LADDER_BUDGET'] = 'a4', '9.8B'
exec(open(os.path.join(os.path.dirname(os.path.abspath(config_file)), 'ladder_gpu.py')).read())

study = YarnStudy()
init_from = 'extend'
extend_from = out_dir  # the headline run's out_dir, as ladder_gpu.py names it; only ever read

# Context: the headline length stretched by the study's factor, with the same tokens per micro-batch and per step
yarn_original_context = block_size
yarn_factor = study.factor
tokens_per_micro_batch = batch_size * block_size
block_size = int(yarn_factor * yarn_original_context)
batch_size = tokens_per_micro_batch // block_size
long_pages = True  # one page per sequence, so the model has a reason to look far back

# Length: the LR continues the headline schedule, which ended at min_lr (lr_decay_iters is unchanged, so get_lr
# returns min_lr from here on); no new warmup. Evals keep the headline run's cadence as a share of the run.
eval_interval = study.finetune_steps * eval_interval // max_iters
max_iters += study.finetune_steps
snapshot_interval = study.snapshot_interval

out_dir = f'{extend_from}-yarn{yarn_factor:g}x'
wandb_project = 'curioslm-yarn'
wandb_run_name = f'{wandb_run_name}-yarn{yarn_factor:g}x'
