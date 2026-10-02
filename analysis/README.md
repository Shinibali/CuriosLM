# analysis

Scripts that turn runs and checkpoints into the post's figures. They are not needed for training.

```bash
uv sync --extra analysis
uv run --extra analysis python analysis/<study>/01_<name>.py   # from the repo root
```

**Layout:** one folder per study, i.e. per finding or section of the post.

- **Entry points** are numbered in the order they run: `01_…`, `02_…`. A later script reads an earlier one's
  outputs from `data/`, never imports it.
- **Shared code within a study** is an unnumbered module next to the scripts.
- **Code shared across studies** goes in `common/`.
- **Outputs:** each study writes only to its own `data/` (pulled or computed data, cached so figures rebuild
  offline) and `figures/`.

| Study | What it shows |
|---|---|
| [`step320_spike/`](step320_spike/) | A one-step loss spike shared by every run, traced to one "baby talk" page in the data |
