# step320_spike

Every run's training loss jumps at step 320 (157.3M tokens) and recovers on the next step. The runs differ in
architecture, run length and GPU count but share the data order, and the batch at that step holds a page
rewritten in "baby talk" (l → w, th → d, ...).

| Script | Needs | Writes |
|---|---|---|
| `01_loss_spike.py` | wandb (first run, or `--refresh`) | `data/loss_history.csv`, `data/run_configs.json` (incl. GPU count), `data/spike.json` (the detected step), `figures/loss_spike.png` |
| `02_unmangle.py` | 01's outputs, local train shards up to the spike's, the ladder a4 checkpoint | `data/page_location.json` (shard, offset and sequence, found by replaying the training loader to the logged batch and taking its worst sequence), `data/page_clean.txt`, `data/unmangle_changes.csv` (each change, runner-up, margin) |
| `03_token_losses.py` | as 02, plus 02's outputs | `figures/word_losses.png` (loss per changed word, real vs baby-talk spelling, with token splits), `figures/token_losses.html` (interactive per-token view, CircuitsVis), `data/token_losses.csv`, `data/word_losses.csv` |

`page.py` holds what 02 and 03 share: the loader replay that finds the page, the run and checkpoint, output paths.
