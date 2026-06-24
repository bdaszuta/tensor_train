"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Demo: compress a smooth 2D function with tensor-train,
           showing original, TT cores with support nodes, and
           reconstruction error.  Compression ratio in title.
"""
import sys
import os as _os
from pathlib import Path

# Locate bindings directory (one level up: usage/ is inside bind_py/).
_script_dir = Path(__file__).resolve().parent
_bind_dir = _script_dir.parent
if str(_bind_dir) not in sys.path:
    sys.path.insert(0, str(_bind_dir))

# Repo root: two levels up from usage/ (usage/ -> bind_py/ -> repo root).
_repo_root = _script_dir.parent.parent
_figs_dir = _repo_root / "figs"
_figs_dir.mkdir(parents=True, exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensor_train import from_dense, TensorTrain


# =========================================================================
# 2D test function
# =========================================================================

def demo_func(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-separable 2D: smooth ridge + Gaussians, decaying SVs."""
    return (np.exp(-0.5 * ((x - y) / 0.08) ** 2)            # diagonal ridge
            + 0.3 * np.exp(-((x - 0.25) ** 2 + (y - 0.75) ** 2) / 0.02)
            + 0.2 * np.exp(-((x - 0.7) ** 2 + (y - 0.3) ** 2) / 0.03))


# =========================================================================
# Grid setup
# =========================================================================

N = 256                     # grid points per axis
xs = np.linspace(0, 1, N)
ys = np.linspace(0, 1, N)
X, Y = np.meshgrid(xs, ys, indexing="ij")
F = demo_func(X, Y)            # shape (N, N), row-major in x

dense = np.ascontiguousarray(F.ravel())  # row-major flat vector
total_values = dense.size

print(f"Grid: {N}x{N}  = {total_values} values")
print()


# =========================================================================
# Compression sweep
# =========================================================================

eps_values = [1e-2, 1e-4, 1e-6]

for eps in eps_values:
    # -- compress ----------------------------------------------------------
    tt = from_dense(dense, [N, N], eps=eps)

    rank = tt.ranks[1]        # only interior bond for d=2
    n_params = tt.num_params
    cr = total_values / n_params
    recon = tt.to_dense().reshape(N, N)
    error_map = np.abs(recon - F)
    rel_err = (np.linalg.norm(recon - F)
               / max(np.linalg.norm(F), 1e-300))

    print(f"eps={eps:.0e}: rank={rank:3d}  n_params={n_params:6d}  "
          f"cr={cr:6.1f}x  rel_err={rel_err:.2e}")

    # Extract core views.
    cores = tt.cores          # [core0(1,N,rank), core1(rank,N,1)]

    c0 = np.asarray(cores[0]).reshape(N, rank)    # (N_x, r)
    c1 = np.asarray(cores[1]).reshape(rank, N)    # (r, N_y)

    # -- build figure --------------------------------------------------
    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(
        f"TT compression of 2D function  |  "
        f"$\\varepsilon = {eps:.0e}$  |  "
        f"${N}\\!\\times\\!{N}$ grid  |  "
        f"rank $r={rank}$  |  "
        f"${total_values}\\,/\\,{n_params} = {cr:.1f}\\!\\times$",
        fontsize=12, fontweight="bold")

    gs = plt.GridSpec(1, 3, figure=fig, width_ratios=[1, 1.1, 1],
                  wspace=0.35)

    # -- Panel 1: Original function ------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.pcolormesh(X, Y, F, shading="auto", cmap="viridis",
                         rasterized=True)
    ax1.set_title("Original  $f(x,y)$", fontsize=11)
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$")
    ax1.set_aspect("equal")
    plt.colorbar(im1, ax=ax1, shrink=0.82)

    # -- Panel 2: TT cores (support nodes) -----------------------------
    # Use a subgridspec confined to the middle cell of the outer grid.
    gs2 = gs[0, 1].subgridspec(2, 1, hspace=0.45)
    ax2a = fig.add_subplot(gs2[0, 0])
    ax2b = fig.add_subplot(gs2[1, 0])

    # Core 0: (N_x, r)  --  x support nodes per bond index.
    im2a = ax2a.pcolormesh(np.arange(rank), xs, c0,
                           shading="auto", cmap="coolwarm",
                           rasterized=True)
    ax2a.set_title(f"Core 0  $(1,\\,N_x,\\,r)$    "
                   f"$N_x={N},\\; r={rank}$", fontsize=10)
    ax2a.set_xlabel("bond index  $a_1$")
    ax2a.set_ylabel("$x$  (support node)")
    plt.colorbar(im2a, ax=ax2a, shrink=0.85)

    # Mark a few support-node stripes to illustrate concept.
    for b in range(0, rank, max(1, rank // 5)):
        ax2a.axvline(b, color="black", lw=0.5, alpha=0.3)

    # Core 1: (r, N_y)  --  y support nodes per bond index.
    im2b = ax2b.pcolormesh(ys, np.arange(rank), c1,
                           shading="auto", cmap="coolwarm",
                           rasterized=True)
    ax2b.set_title(f"Core 1  $(r,\\,N_y,\\,1)$    "
                   f"$r={rank},\\; N_y={N}$", fontsize=10)
    ax2b.set_xlabel("$y$  (support node)")
    ax2b.set_ylabel("bond index  $a_1$")
    plt.colorbar(im2b, ax=ax2b, shrink=0.85)

    for b in range(0, rank, max(1, rank // 5)):
        ax2b.axhline(b, color="black", lw=0.5, alpha=0.3)

    # -- Panel 3: Reconstruction error ---------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.pcolormesh(X, Y, error_map, shading="auto",
                         cmap="inferno", norm="log",
                         rasterized=True)
    ax3.set_title(
        f"Reconstruction error  $|f - \\tilde{{f}}|$\n"
        f"$\\|f - \\tilde{{f}}\\|_F\\,/\\,\\|f\\|_F = {rel_err:.2e}$",
        fontsize=10)
    ax3.set_xlabel("$x$")
    ax3.set_ylabel("$y$")
    ax3.set_aspect("equal")
    plt.colorbar(im3, ax=ax3, shrink=0.82)

    # -- Save ----------------------------------------------------------
    fname = _figs_dir / f"demo_2d_compress_eps_{eps:.0e}.png"
    fig.savefig(str(fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {fname}")

print()
print("Done.")
#
# :D
#
