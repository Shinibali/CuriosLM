import torch

from curioslm.model import RotaryEmbedding


def test_rope_relative_position_invariance():
    """
    The defining property of RoPE: the (pre-softmax) dot product between a rotated
    query and a rotated key should depend only on their relative position (m - n),
    not on their absolute positions. This test checks two things:

    1. Two (q_pos, k_pos) pairs with the *same* relative distance must give the
       same dot product, even though their absolute positions differ.
    2. A *different* relative distance must give a *different* dot product --
       otherwise a broken no-op rotation would trivially pass check 1 too, since
       an unrotated q/k pair's dot product never depends on position at all.
    """
    torch.manual_seed(0)
    head_size = 8
    block_size = 64
    theta = 10000.0
    rope = RotaryEmbedding(head_size, block_size, theta)

    q_vec = torch.randn(1, 1, 1, head_size)
    k_vec = torch.randn(1, 1, 1, head_size)

    def rotated_dot(q_pos, k_pos):
        T = max(q_pos, k_pos) + 1
        q = torch.zeros(1, 1, T, head_size)
        k = torch.zeros(1, 1, T, head_size)
        q[:, :, q_pos, :] = q_vec
        k[:, :, k_pos, :] = k_vec
        q_rot, k_rot = rope(q, k)
        return torch.sum(q_rot[:, :, q_pos, :] * k_rot[:, :, k_pos, :])

    # same relative distance (5), different absolute positions -> dot products must match
    dot_a = rotated_dot(q_pos=10, k_pos=5)
    dot_b = rotated_dot(q_pos=30, k_pos=25)
    assert torch.allclose(dot_a, dot_b, atol=1e-5), (
        f"expected equal dot products for equal relative distance, got {dot_a} vs {dot_b}"
    )

    # different relative distance -> dot product should differ
    dot_c = rotated_dot(q_pos=10, k_pos=2)
    assert not torch.allclose(dot_a, dot_c, atol=1e-5), (
        "dot product did not change with relative distance -- rotation may be a no-op"
    )
