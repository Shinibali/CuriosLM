"""Pull wandb run histories and configs once and cache them, so figures rebuild without wandb access."""
import csv
import json

WANDB_ENTITY = "models-independent4024"


def fetch(paths, keys, config_keys, history_csv, config_json):
    """Every logged row with all of `keys` set, for each wandb run path ("project/run_id"), plus the named
    config values and `world_size` (wandb's recorded GPU count). Written to history_csv (a `run` column plus
    `keys`) and config_json ({path: {key: value}})."""
    import wandb
    api, rows, configs = wandb.Api(), [], {}
    for path in paths:
        run = api.run(f"{WANDB_ENTITY}/{path}")
        configs[path] = {**{k: run.config[k] for k in config_keys}, "world_size": run.metadata["gpu_count"]}
        for x in run.scan_history(keys=keys):
            if all(x.get(k) is not None for k in keys):
                rows.append({"run": path, **{k: x[k] for k in keys}})
    with open(history_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", *keys])
        w.writeheader()
        w.writerows(rows)
    with open(config_json, "w") as f:
        json.dump(configs, f, indent=2)


def fetch_configs(paths, config_keys, config_json):
    """Only the named config values (plus world_size) for each run — no history download. A key a run never logged
    (e.g. data_mix, before train.py logged dict settings) comes back as None."""
    import wandb
    api = wandb.Api()
    configs = {p: {**{k: (r := api.run(f"{WANDB_ENTITY}/{p}")).config.get(k) for k in config_keys},
                   "world_size": r.metadata["gpu_count"]} for p in paths}
    with open(config_json, "w") as f:
        json.dump(configs, f, indent=2)


def load_configs(paths, config_json):
    with open(config_json) as f:
        configs = json.load(f)
    return {p: configs[p] for p in paths}


def load(paths, history_csv, config_json):
    """({path: [row dict with float values]}, {path: config}) for the given runs; cached runs
    not in `paths` are skipped."""
    history = {p: [] for p in paths}
    with open(history_csv) as f:
        for r in csv.DictReader(f):
            if r["run"] in history:
                history[r["run"]].append({k: float(v) for k, v in r.items() if k != "run"})
    with open(config_json) as f:
        configs = json.load(f)
    return history, {p: configs[p] for p in paths}
