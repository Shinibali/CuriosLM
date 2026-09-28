"""
Shared runtime setup: device selection, mixed precision, filesystem paths, and
hardware peak-FLOPS numbers. train.py and sample.py both import from here so their
device handling can't drift apart.
"""

import os
from contextlib import nullcontext

import torch

# Absolute paths, so scripts work regardless of the shell's cwd. The project root
# (holding data/, config/, out/) is one level up from this package.
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)

DTYPES = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'float16': torch.float16,
}


def project_path(*parts):
    """Path relative to the project root: data/, config/, out/."""
    return os.path.join(PROJECT_ROOT, *parts)


def package_path(*parts):
    """Path inside this package — used to locate configurator.py, which is exec'd by path."""
    return os.path.join(PACKAGE_ROOT, *parts)


def pick_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def device_type_of(device):
    """'cuda:3' -> 'cuda'. torch.amp wants the family, not the index."""
    return device.split(':')[0]


def pick_dtype(dev_type):
    """Prefer bfloat16: float32's exponent range means no GradScaler and no silent
    underflow. float16 is only right on CUDA cards too old for bf16 (pre-Ampere)."""
    if dev_type == 'cuda':
        return 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
    if dev_type == 'mps':
        return 'bfloat16'
    return 'float32'


def autocast_ctx(dev_type, dtype):
    if dev_type == 'cpu' or dtype == 'float32':
        return nullcontext()
    return torch.amp.autocast(device_type=dev_type, dtype=DTYPES[dtype])


def grad_scaler(dev_type, dtype):
    """Loss scaling stops float16 gradients underflowing; a no-op for bf16/fp32."""
    return torch.amp.GradScaler(dev_type, enabled=(dtype == 'float16'))


def enable_tf32():
    """TF32 trades mantissa bits for speed on Ampere+ matmuls. CUDA-only knob."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


# Dense (non-sparsity-inflated) tensor-core throughput at bf16/fp16, in FLOP/s. Vendors
# headline the 2x "with sparsity" number; an inflated peak understates MFU.
# Matched by substring in dict order, so the H100 variants must precede the bare 'H100' (SXM).
_CUDA_PEAK_FLOPS = {
    'H200': 989e12,
    'H100 NVL': 835e12,
    'H100 PCIe': 756e12,
    'H100': 989e12,       # SXM, reported as "NVIDIA H100 80GB HBM3"
    'A100': 312e12,
    'L40S': 362e12,
    'A6000': 155e12,
    'A40': 150e12,
    '4090': 165e12,
    '3090': 71e12,
    'V100': 125e12,       # fp16; V100 has no bf16
}


def peak_flops(dev_type, override=0.0):
    """Denominator for MFU; None when unknown, so callers report n/a instead of a number
    computed against a guess. None on MPS: Apple publishes no comparable dense figure.
    Pass `override` if you have one you trust."""
    if override and override > 0:
        return override
    if dev_type != 'cuda':
        return None
    return peak_flops_for_name(torch.cuda.get_device_name())


def peak_flops_for_name(device_name):
    """Look up a CUDA device name in the peak table; None if unknown."""
    for key, flops in _CUDA_PEAK_FLOPS.items():
        if key.lower() in device_name.lower():
            return flops
    return None


def peak_memory_gb(dev_type):
    """Peak memory PyTorch allocated on this process's GPU since startup, in GB; None off CUDA
    (MPS exposes current usage but no peak)."""
    if dev_type != 'cuda':
        return None
    return torch.cuda.max_memory_allocated() / 1e9


def describe(device, dtype, compile_enabled):
    """One line saying what we actually ended up running on."""
    bits = [f"device={device}", f"dtype={dtype}", f"compile={compile_enabled}"]
    if device_type_of(device) == 'cuda':
        bits.append(f"gpu={torch.cuda.get_device_name()}")
    return "runtime: " + ", ".join(bits)
