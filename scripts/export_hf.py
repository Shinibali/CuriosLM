"""Export the released models' weights for their public Hugging Face repos: weights only (no optimizer state),
as safetensors, one folder per repo under out-release/ (gitignored, like every top-level out-* folder).

Model A has GPT-2's exact architecture, so it is converted to the 🤗 transformers GPT-2 format and loads with
GPT2LMHeadModel.from_pretrained. The Gemma-style models keep CuriosLM's own format: model.safetensors plus a
config.json holding the checkpoint's model_args, loaded with curioslm.model.GPT. Every export is checked by
reloading it and comparing its logits with the original checkpoint's on the same tokens. Each folder also gets a
README.md model card, its numbers read from the checkpoint and the analysis studies' result tables.

    uv run python scripts/export_hf.py [--only <repo name>]
Needs the checkpoints on local disk (or pulls them from the private staging repo). Upload is a separate step.
"""
import argparse
import csv
import json
import os
import sys

import tiktoken
import torch
from safetensors.torch import load_model, save_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from curioslm import checkpoints  # noqa: E402
from curioslm.config import GPTConfig, IOConfig, YarnStudy  # noqa: E402
from curioslm.model import GPT  # noqa: E402

ORG = "CuriosLM"
OUT = os.path.join(ROOT, "out-release")
HEADLINE_B = "out-ladder-fineweb_edu-a4-9.8B"
YARN_DIR = "out-ladder-fineweb_edu-a4-9.8B-yarn4x"
# repo name -> (run out_dir, checkpoint file in it, format): which run each public repo publishes
RELEASES = {
    "gpt2-124m-fineweb-edu": ("out-ladder-fineweb_edu-a0-9.8B", "ckpt.pt", "transformers-gpt2"),
    "gemma-style-113m-fineweb-edu": (HEADLINE_B, "ckpt.pt", "curioslm"),
    "gemma-style-113m-fineweb-edu-cosmopedia": ("out-ladder-fineweb_edu-a4-9.8B-mix_fineweb50_cosmopedia50",
                                                "ckpt.pt", "curioslm"),
    "gemma-style-113m-fineweb-edu-yarn4k": (YARN_DIR, None, "curioslm"),  # None: the fine-tune's last snapshot
}
# Where each training dataset key comes from (data/<key>/prepare.py)
DATASETS = {"fineweb_edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
            "cosmopedia_v2": ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2")}
LICENSE = "mit"  # the code it's built on (nanoGPT) is MIT
HEADLINE_DATA = os.path.join(ROOT, "analysis", "headline_comparison", "data")
MIX_DATA = os.path.join(ROOT, "analysis", "data_mix", "data")
YARN_CONTEXT_CSV = os.path.join(ROOT, "analysis", "positions", "data", "yarn_context.csv")
PROBE = "The mitochondria is the powerhouse of the cell. Photosynthesis converts light energy into"
TOLERANCE = 1e-4  # max |logit difference| allowed between an export and its checkpoint (both float32)


def checkpoint_path(out_dir, filename):
    """Local path of the checkpoint to export; pulled from the staging repo if it isn't on disk."""
    if filename is None:  # the YaRN fine-tune: its last weights-only snapshot, step = base run's + fine-tune steps
        base = checkpoints.load(os.path.join(ROOT, HEADLINE_B, "ckpt.pt"), map_location="cpu")
        filename = f"ckpt-{base['iter_num'] + YarnStudy().finetune_steps}.pt"
    path = os.path.join(ROOT, out_dir, filename)
    if not os.path.exists(path):
        cwd = os.getcwd()
        os.chdir(ROOT)  # download_checkpoint writes <out_dir>/<filename> relative to the working directory
        try:
            checkpoints.download_checkpoint(out_dir, IOConfig().hf_checkpoint_repo_id, filename=filename)
        finally:
            os.chdir(cwd)
    return path


def to_gpt2_state_dict(state_dict):
    """CuriosLM's Model A weights under 🤗 GPT-2's names: the same names, but the four projection matrices are
    stored transposed (GPT-2's Conv1D keeps (in, out); nn.Linear keeps (out, in))."""
    transposed = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")
    return {name: tensor.t().contiguous() if name.endswith(transposed) else tensor
            for name, tensor in state_dict.items() if not name.endswith(".attn.bias")}  # .attn.bias: causal mask


def export_gpt2(model, config, folder):
    """Save Model A in 🤗 transformers GPT-2 format, with GPT-2's tokenizer; returns the reloaded model."""
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
    hf_config = GPT2Config(
        vocab_size=config.vocab_size, n_positions=config.block_size, n_embd=config.n_embd,
        n_layer=config.n_layer, n_head=config.n_head,
        activation_function="gelu",  # nn.GELU(): exact erf GELU, not GPT-2's default tanh approximation
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, tie_word_embeddings=True,
        bos_token_id=50256, eos_token_id=50256)  # <|endoftext|>, the document separator in training
    hf_model = GPT2LMHeadModel(hf_config)
    missing, unexpected = hf_model.load_state_dict(to_gpt2_state_dict(model.state_dict()), strict=False)
    assert not unexpected and all(name == "lm_head.weight" for name in missing), (missing, unexpected)  # tied
    hf_model.tie_weights()
    hf_model.save_pretrained(folder)
    GPT2TokenizerFast.from_pretrained("openai-community/gpt2").save_pretrained(folder)
    return GPT2LMHeadModel.from_pretrained(folder).eval()


def export_curioslm(model, model_args, iter_num, folder):
    """Save a model in CuriosLM's format; returns the reloaded model."""
    save_model(model, os.path.join(folder, "model.safetensors"))  # handles the tied embedding/output weights
    with open(os.path.join(folder, "config.json"), "w") as f:
        json.dump({"model_args": model_args, "iter_num": iter_num, "tokenizer": "tiktoken gpt2"}, f, indent=2)
    reloaded = GPT(GPTConfig(**model_args))
    load_model(reloaded, os.path.join(folder, "model.safetensors"))
    return reloaded.eval()


def logits_of(model, ids):
    with torch.no_grad():
        out = model(ids)
    logits = out.logits if hasattr(out, "logits") else out[0]  # transformers output, or CuriosLM's (logits, loss)
    return logits[0, -1]  # CuriosLM without targets returns only the last position: compare that one


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def benchmarks(path, column):
    """{(task, metric): value} for one model's column ("a" or "b") of a study's paired-benchmark table."""
    return {(r["task"], r["metric"]): float(r[column]) for r in rows(path)}


def perplexities(path, model_column, model_name):
    """{held-out set: perplexity} for one model in a study's val_perplexity table."""
    return {r["set"]: float(r["perplexity"]) for r in rows(path) if r[model_column] == model_name}


def tokens_seen(checkpoint):
    """Training tokens up to this checkpoint: steps × tokens per step, from its training config. A fine-tune's
    steps beyond the base run are counted at its own (longer) batch."""
    config = checkpoint["config"]
    per_step = config["batch_size"] * config["gradient_accumulation_steps"] * config["block_size"]
    if config.get("init_from") == "extend":  # the YaRN fine-tune continues the headline run
        base = checkpoints.load(os.path.join(ROOT, config["extend_from"], "ckpt.pt"), map_location="cpu")
        return tokens_seen(base) + (checkpoint["iter_num"] - base["iter_num"]) * per_step
    return checkpoint["iter_num"] * per_step


def data_line(config):
    mix = config.get("data_mix") or {config["dataset"]: 1.0}
    return " + ".join(f"{share:.0%} [{DATASETS[key][0]}](https://huggingface.co/datasets/{DATASETS[key][0]}) "
                      f"(`{DATASETS[key][1]}`)" for key, share in mix.items())


def architecture_lines(args):
    if args["position_encoding"] == "absolute":
        return [f"GPT-2 small, unchanged: {args['n_layer']} layers, {args['n_head']} heads, d_model {args['n_embd']}, "
                f"learned absolute positions ({args['block_size']:,}), GELU MLP, pre-LayerNorm, biases on."]
    model_config = GPTConfig(**args)
    globals_ = [layer for layer in range(args["n_layer"]) if model_config.is_global_layer(layer)]
    lines = [f"{args['n_layer']} layers, d_model {args['n_embd']}, context {args['block_size']:,} tokens",
             f"RoPE (base {args['rope_theta']:,}) instead of learned positions",
             f"grouped-query attention: {args['n_head']} query heads, {args['n_kv_head']} key/value heads",
             f"local/global attention: layers {', '.join(map(str, globals_))} global, the rest see the last "
             f"{args['attention_window']} tokens",
             "GeGLU MLP (inner size ⅔ of GPT-2's, matching its parameter count)",
             "RMSNorm before and after each attention and MLP block (sandwich)"]
    if args.get("yarn_factor"):
        lines.append(f"YaRN-scaled RoPE: {args['yarn_factor']:g}× stretch of the original "
                     f"{args['yarn_original_context']:,}-token context")
    return lines


def results_table(name):
    """The card's evaluation table: held-out perplexity and zero-shot benchmarks, from the studies' outputs."""
    fmt = "| {} | {} |"
    if name == "gemma-style-113m-fineweb-edu-yarn4k":
        by_length = {}
        for r in rows(YARN_CONTEXT_CSV):
            by_length.setdefault(r["model"], {})[int(r["length"])] = float(r["perplexity"])
        tuned = by_length[max((m for m in by_length if "fine-tuned" in m), key=lambda m: int(m.split()[-2]))]
        plain = by_length["Model B, plain RoPE"]
        lines = ["| Context (tokens) | Perplexity, this model | Perplexity, base model (plain RoPE) |",
                 "|---|---|---|"]
        lines += [f"| {length:,} | {tuned[length]:.2f} | {plain[length]:.2f} |" for length in sorted(tuned)]
        return lines, ("Perplexity on held-out FineWeb-Edu pages long enough for each context length, scored at the "
                       "length shown.")
    if name == "gemma-style-113m-fineweb-edu-cosmopedia":
        scores = benchmarks(os.path.join(MIX_DATA, "benchmarks.csv"), "b")  # "a" = base Model B, "b" = 50:50 mix
        ppl = {r["set"]: float(r["perplexity"]) for r in rows(os.path.join(MIX_DATA, "val_perplexity.csv"))
               if r["model"] == "Model B, 50:50 mix"}
    else:
        column = "a" if name == "gpt2-124m-fineweb-edu" else "b"  # "a" = Model A, "b" = Model B
        scores = benchmarks(os.path.join(HEADLINE_DATA, "significance.csv"), column)
        ppl = perplexities(os.path.join(HEADLINE_DATA, "val_perplexity.csv"), "model", column.upper())
    lines = ["| Evaluation | Score |", "|---|---|",
             fmt.format("Perplexity, held-out FineWeb-Edu", f"{ppl['fineweb_edu']:.2f}"),
             fmt.format("Perplexity, held-out Cosmopedia-v2", f"{ppl['cosmopedia_v2']:.2f}"),
             fmt.format("ARC-Easy (acc)", f"{scores[('arc_easy', 'acc')]:.1%}"),
             fmt.format("HellaSwag (acc_norm)", f"{scores[('hellaswag', 'acc_norm')]:.1%}"),
             fmt.format("LAMBADA (acc)", f"{scores[('lambada_openai', 'acc')]:.1%}"),
             fmt.format("LAMBADA (perplexity)", f"{scores[('lambada_openai', 'perplexity')]:.1f}")]
    return lines, ("Perplexity on about 2M held-out tokens of each dataset; benchmarks zero-shot with EleutherAI's "
                   "lm-evaluation-harness.")


def usage(name, fmt):
    repo = f"{ORG}/{name}"
    if fmt == "transformers-gpt2":
        return [f"Standard 🤗 GPT-2 format:", "", "```python",
                "from transformers import AutoModelForCausalLM, AutoTokenizer",
                f'tokenizer = AutoTokenizer.from_pretrained("{repo}")',
                f'model = AutoModelForCausalLM.from_pretrained("{repo}")',
                'ids = tokenizer("The mitochondria is", return_tensors="pt").input_ids',
                "print(tokenizer.decode(model.generate(ids, max_new_tokens=40, do_sample=True, top_k=50)[0]))",
                "```"]
    return ["This architecture isn't in 🤗 transformers; it loads with the CuriosLM model code "
            "(`curioslm/model.py`), which will be released publicly. The tokenizer is GPT-2's (tiktoken `gpt2`).",
            "", "```python",
            "import json, tiktoken, torch", "from huggingface_hub import hf_hub_download",
            "from safetensors.torch import load_model", "from curioslm.config import GPTConfig",
            "from curioslm.model import GPT", "",
            f'repo = "{repo}"',
            'config = json.load(open(hf_hub_download(repo, "config.json")))',
            'model = GPT(GPTConfig(**config["model_args"])).eval()',
            'load_model(model, hf_hub_download(repo, "model.safetensors"))',
            'enc = tiktoken.get_encoding("gpt2")',
            'ids = torch.tensor([enc.encode("The mitochondria is")])',
            "print(enc.decode(model.generate(ids, max_new_tokens=40, top_k=50)[0].tolist()))", "```"]


SUMMARIES = {  # one sentence per repo: what it is, in the experiment's terms
    "gpt2-124m-fineweb-edu": "Model A: GPT-2 small (124M) trained from scratch, the baseline of a controlled "
                             "comparison with a Gemma-style model trained identically.",
    "gemma-style-113m-fineweb-edu": "Model B: a 113M model with Gemma-style architecture choices, trained exactly "
                                    "like its GPT-2 baseline (same data, order, tokenizer, optimizer and schedule).",
    "gemma-style-113m-fineweb-edu-cosmopedia": "Model B's architecture trained on a 50:50 mix of FineWeb-Edu and "
                                               "Cosmopedia-v2 instead of FineWeb-Edu alone, everything else equal.",
    "gemma-style-113m-fineweb-edu-yarn4k": "Model B with its context extended 4× by YaRN, then briefly fine-tuned on "
                                           "long pages.",
}


def model_card(name, fmt, checkpoint, model_args):
    config = checkpoint["config"]
    pipeline = "library_name: transformers\npipeline_tag: text-generation\n" if fmt == "transformers-gpt2" else ""
    datasets = "".join(f"- {DATASETS[key][0]}\n" for key in (config.get("data_mix") or {config["dataset"]: 1}))
    table, how = results_table(name)
    training = [f"- Data: {data_line(config)}",
                f"- Tokens: {tokens_seen(checkpoint) / 1e9:.2f}B, {checkpoint['iter_num']:,} optimizer steps"]
    if config.get("init_from") == "extend":
        training.append(f"- Continued from [{ORG}/gemma-style-113m-fineweb-edu]"
                        f"(https://huggingface.co/{ORG}/gemma-style-113m-fineweb-edu) at its final step, with its "
                        f"optimizer state, at a constant learning rate of {config['min_lr']:g}, on pages of at least "
                        f"{model_args['block_size']:,} tokens")
    else:
        training.append(f"- AdamW, peak learning rate {config['learning_rate']:g} with {config['warmup_iters']} warmup "
                        f"steps and cosine decay to {config['min_lr']:g}, weight decay {config['weight_decay']:g}, "
                        f"seed {config['seed']}")
    per_step = config["batch_size"] * config["gradient_accumulation_steps"] * config["block_size"]
    training.append(f"- Tokens per step: {per_step:,}")
    lines = ["---", f"license: {LICENSE}", "language:\n- en", f"{pipeline}datasets:\n{datasets}tags:\n- curioslm\n"
             "- research\n---", "", f"# {ORG}/{name}", "", SUMMARIES[name], "",
             "A small research model from the CuriosLM project, released for reproducibility and study. It is a "
             "base model: not instruction-tuned and not safety-tuned, so it can produce false, biased or "
             "nonsensical text.", "", "## Architecture", ""]
    lines += [f"- {line}" for line in architecture_lines(model_args)]
    lines += ["", "## Training", ""] + training + ["", "## Results", "", *table, "", how, "", "## Usage", ""]
    lines += usage(name, fmt)
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(RELEASES), help="export just this repo")
    args = parser.parse_args()
    ids = torch.tensor([tiktoken.get_encoding("gpt2").encode(PROBE)])
    for name, (out_dir, filename, fmt) in RELEASES.items():
        if args.only and name != args.only:
            continue
        checkpoint = checkpoints.load(checkpoint_path(out_dir, filename), map_location="cpu")
        model, model_args = checkpoints.build_model(checkpoint, overrides={"dropout": 0.0})
        model.eval()
        folder = os.path.join(OUT, name)
        os.makedirs(folder, exist_ok=True)
        if fmt == "transformers-gpt2":
            reloaded = export_gpt2(model, GPTConfig(**model_args), folder)
        else:
            reloaded = export_curioslm(model, model_args, checkpoint["iter_num"], folder)
        gap = (logits_of(model, ids) - logits_of(reloaded, ids)).abs().max().item()
        assert gap < TOLERANCE, f"{name}: export differs from its checkpoint by {gap:.2e}"
        with open(os.path.join(folder, "README.md"), "w") as f:
            f.write(model_card(name, fmt, checkpoint, model_args))
        size = sum(os.path.getsize(os.path.join(folder, f)) for f in os.listdir(folder)) / 1e9
        print(f"{ORG}/{name}: {fmt}, step {checkpoint['iter_num']:,}, {size:.2f} GB in {folder}; "
              f"max logit difference vs checkpoint {gap:.1e}")


if __name__ == "__main__":
    main()
