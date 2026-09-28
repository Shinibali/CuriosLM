import pytest

from curioslm.runtime import peak_flops_for_name


@pytest.mark.parametrize("device_name, expected", [
    ("NVIDIA H100 80GB HBM3", 989e12),  # SXM
    ("NVIDIA H100 NVL", 835e12),
    ("NVIDIA H100 PCIe", 756e12),
    ("NVIDIA H200", 989e12),
    ("NVIDIA A100-SXM4-80GB", 312e12),
    ("Some Future GPU", None),
])
def test_peak_flops_matches_the_specific_variant_before_the_generic_name(device_name, expected):
    """'H100' is a substring of every H100 variant, so NVL/PCIe must not fall through to the SXM figure."""
    assert peak_flops_for_name(device_name) == expected
