#!/usr/bin/env bash
# Prepare a fresh GPU pod (RunPod `curioslm` template) for training. Runs on the pod; send it from the
# Mac without copying anything first:
#   ssh -i ~/.ssh/runpod_ed25519 -p <port> -o StrictHostKeyChecking=accept-new root@<ip> 'bash -s' < scripts/pod_setup.sh
# Credentials come from the template as env vars (HF_TOKEN, WANDB_API_KEY, GITHUB_TOKEN); none are stored here.

# Wrapped in main so bash reads the whole script before running it: under `bash -s` the script arrives
# on stdin, and a command that read stdin would otherwise swallow the lines after it.
main() {
    set -eo pipefail
    # RunPod writes the pod's env vars here; ~/.bashrc only sources it for interactive shells.
    source /etc/rp_environment
    set -u

    for var in HF_TOKEN WANDB_API_KEY GITHUB_TOKEN; do
        if [ -z "${!var:-}" ]; then
            echo "missing $var -- check the template's environment variables and RunPod secrets" >&2
            exit 1
        fi
    done
    echo "credentials: HF_TOKEN, WANDB_API_KEY, GITHUB_TOKEN present"

    # The image's uv is package-manager-installed (can't self-update) and too old for our lockfile.
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
    echo "uv: $(uv --version)"

    # git asks this helper for credentials on demand, so the token is read from the env, never written to disk.
    git config --global credential.helper '!f() { echo username=Shinibali; echo "password=$GITHUB_TOKEN"; }; f'

    local repo=/workspace/gpt-from-scratch-notes
    if [ ! -d "$repo" ]; then
        git clone https://github.com/Shinibali/gpt-from-scratch-notes.git "$repo"
    fi
    cd "$repo/shinibali/karpathy-tutorials/CuriosLM"
    git pull --ff-only

    uv sync --locked
    uv run --locked python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable -- host driver must support CUDA >= 13.0'; print('torch', torch.__version__, '|', torch.cuda.device_count(), 'x', torch.cuda.get_device_name())"

    # Before any torchrun: a slow download inside DDP can outlast the process group's barrier timeout.
    uv run --locked python -m curioslm.loaders.hf_shards download
    echo "shards: $(ls data/fineweb_edu/*.npy | wc -l) files"

    echo "ready at commit $(git log --oneline -1)"
}
main "$@"
