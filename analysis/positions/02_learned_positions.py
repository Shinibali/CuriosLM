"""Model A's learned position embeddings become waves on their own; Model B's RoPE is waves by construction.

Left: cosine similarity between every pair of Model A's position vectors. If similarity depends only on the
distance between positions, the matrix is constant along its diagonals (a Toeplitz matrix). Middle: the top
principal components of those vectors against position. A Toeplitz matrix has near-sinusoidal eigenvectors, so the
components look like waves.

Model B has no position table: RoPE rotates queries and keys instead. Two things compare directly. Right:
similarity against distance. Rotating a vector to positions m and n and comparing gives Σᵢ eᵢ·cos((m − n)·θᵢ) for
RoPE, a function of distance alone by construction (shown with equal energy eᵢ in every band, i.e. position alone,
before content weights the bands); for Model A, the average of each diagonal. The script also prints how Model
A's variation splits by wavelength (a Fourier transform along positions), quoted in the post's text.

    uv run --extra analysis python analysis/positions/02_learned_positions.py
Needs Model A's checkpoint (downloaded if missing). Writes data/learned_positions.csv and
figures/learned_positions.png.
"""
import csv
import math
import os

import matplotlib.pyplot as plt
import torch

import positions  # this study's shared choices; also puts analysis/ on sys.path
from common import models, style  # noqa: E402

OUT_CSV = os.path.join(positions.DATA, "learned_positions.csv")
OUT_PNG = os.path.join(positions.FIGURES, "learned_positions.png")
COMPONENTS = 3  # principal components to plot (the 4th is mostly the first few positions)
SKIP = 1  # leading positions left out of the components: position 0 is an outlier (it starts every sequence)


def main():
    model = models.load_model(positions.out_dir(positions.LEARNED), "cpu")
    position_table = model.transformer.wpe.weight.detach().float()  # (positions, n_embd): one vector per position
    num_positions = len(position_table)
    unit_vectors = position_table / position_table.norm(dim=1, keepdim=True)
    similarity = unit_vectors @ unit_vectors.T  # cosine similarity, (positions, positions)
    # How much of the similarity is explained by distance alone: average each diagonal, compare with the matrix
    mean_by_distance = torch.stack([similarity.diagonal(offset).mean() for offset in range(num_positions)])
    position_index = torch.arange(num_positions)
    distance_matrix = (position_index[:, None] - position_index[None, :]).abs()
    distance_only = mean_by_distance[distance_matrix]  # the matrix rebuilt from distance alone (a Toeplitz matrix)
    explained = 1 - ((similarity - distance_only) ** 2).sum() / ((similarity - similarity.mean()) ** 2).sum()  # R²

    kept_positions = position_table[SKIP:]
    centred = kept_positions - kept_positions.mean(dim=0)
    # columns of component_curves: each principal component's value at positions SKIP .. num_positions − 1
    component_curves, singular_values, _ = torch.linalg.svd(centred, full_matrices=False)
    variance_share = (singular_values ** 2 / (singular_values ** 2).sum()).tolist()
    os.makedirs(positions.DATA, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["position"] + [f"pc{c + 1}" for c in range(COMPONENTS)] + ["similarity_at_distance"])
        writer.writerows(
            [position] + (component_curves[position - SKIP, :COMPONENTS].tolist() if position >= SKIP
                          else [""] * COMPONENTS) + [mean_by_distance[position].item()]
            for position in range(num_positions))
    top_shares = ", ".join(f"{share:.1%}" for share in variance_share[:COMPONENTS])
    print(f"distance alone explains {explained:.1%} of the similarity matrix's variation; top {COMPONENTS} components "
          f"carry {sum(variance_share[:COMPONENTS]):.1%} of the variance ({top_shares})")

    # Model B's RoPE: its similarity at each distance, from its frequencies
    rope_freqs, _ = positions.rope_frequencies(positions.model_args(positions.out_dir(positions.ROPE)))
    distances = torch.arange(num_positions, dtype=torch.float32)
    rope_similarity = torch.cos(distances[:, None] * rope_freqs[None, :]).mean(dim=1)  # equal energy in every band
    pair_spread = [torch.quantile(similarity.diagonal(offset), torch.tensor([0.1, 0.9]))
                   for offset in range(num_positions - 1)]  # 10th and 90th percentile of the pairs at each distance
    # Model A's table as waves along positions: power at each frequency, summed over the embedding dimensions
    power = torch.fft.rfft(centred, dim=0).abs().pow(2).sum(dim=1)[1:]  # drop frequency 0 (the mean, removed anyway)
    power_share = power / power.sum()
    learned_wavelengths = (num_positions - SKIP) / torch.arange(1, len(power_share) + 1)  # positions per cycle
    rope_wavelengths = (2 * math.pi / rope_freqs).tolist()
    slow_share = power_share[:3].sum().item()  # the three longest waves
    short_share = power_share[learned_wavelengths < 30].sum().item()  # the short waves: a flat, noise-like floor
    print(f"learned table: {short_share:.1%} of its variation in waves shorter than 30 positions")
    print(f"learned table: {slow_share:.1%} of its variation in waves of {learned_wavelengths[2]:.0f}+ positions; "
          f"RoPE bands span {min(rope_wavelengths):.0f} to {max(rope_wavelengths):,.0f} positions")

    style.apply()
    fig, (similarity_ax, components_ax, distance_ax) = plt.subplots(
        1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1, 1.3, 1.2]})
    image = similarity_ax.imshow(similarity.numpy(), cmap=style.SEQUENTIAL_CMAP, vmin=similarity.min().item(), vmax=1,
                                 origin="upper")
    similarity_ax.set_xlabel("Position")
    similarity_ax.set_ylabel("Position")
    similarity_ax.grid(False)
    similarity_ax.set_title("Model A: position similarity", fontsize=10)
    fig.colorbar(image, ax=similarity_ax, fraction=0.046, pad=0.04)
    for c in range(COMPONENTS):  # stacked, each scaled to unit height, so their shapes are comparable
        curve = component_curves[:, c] / component_curves[:, c].abs().max()
        offset = -2.4 * c
        components_ax.plot(range(SKIP, num_positions), (curve + offset).numpy(), color=style.OTHER[c], linewidth=1.4)
        components_ax.text(num_positions + 15, offset, f"PC {c + 1}\n{variance_share[c]:.0%}", fontsize=8.5,
                           color=style.INK, va="center")
    components_ax.set_xlabel("Position")
    components_ax.set_yticks([])
    components_ax.set_title(f"Model A: top {COMPONENTS} principal components", fontsize=10)
    components_ax.set_xlim(0, num_positions * 1.12)  # room for the labels

    low, high = zip(*[(q10.item(), q90.item()) for q10, q90 in pair_spread])
    distance_ax.fill_between(range(num_positions - 1), low, high, color=style.MODEL_A, alpha=0.15, linewidth=0)
    distance_ax.plot(range(num_positions), mean_by_distance.numpy(), color=style.MODEL_A,
                     label="Model A, learned table (10–90% shaded)")
    distance_ax.plot(range(num_positions), rope_similarity.numpy(), color=style.MODEL_B,
                     label="Model B, RoPE (all bands equal)")
    distance_ax.axhline(0, color=style.INK_SECONDARY, linewidth=0.8)
    distance_ax.set_xlabel("Distance between positions")
    distance_ax.set_ylabel("Similarity")
    distance_ax.set_title("Both: similarity depends on distance", fontsize=10)
    distance_ax.legend(loc="upper right", fontsize=8)

    fig.text(0.02, 0.98, "Position as waves: learned by Model A, built into Model B", fontsize=12,
             fontweight="bold", va="top")
    fig.text(0.02, 0.92, f"Distance alone explains {explained:.0%} of Model A's position similarity; for RoPE it's "
             f"100% by construction.", fontsize=9.5, color=style.INK_SECONDARY, va="top")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.82, bottom=0.13, wspace=0.3)
    os.makedirs(positions.FIGURES, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"wrote {OUT_CSV}, {OUT_PNG}")

if __name__ == "__main__":
    main()
