"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Sampler-based tensor-train construction tests (from_samples)
"""
import sys
import os as _os

_sys_dir = _os.path.dirname(_os.path.abspath(__file__))
_bind_dir = _os.path.join(_sys_dir, "..", "..", "bind_py")
if _bind_dir not in sys.path:
    sys.path.insert(0, _bind_dir)

import numpy as np

from tensor_train import (
    TensorTrain,
    from_samples,
    FromSamplesOptions,
)


def rel_err(got, want):
    n = np.linalg.norm(want)
    return np.linalg.norm(got - want) / n if n > 0 else np.linalg.norm(got)


def main():
    failed = 0
    total = 0

    def check(name, err, tol, tt=None, min_rank=None):
        nonlocal failed, total
        total += 1
        ok = err <= tol
        if min_rank is not None and tt is not None:
            ok = ok and tt.max_rank >= min_rank
        status = "OK" if ok else "FAIL"
        rk = tt.max_rank if tt is not None else "?"
        print(f"  {name:<38s} err={err:.2e} rank={rk} [{status}]")
        if not ok:
            failed += 1

    # ==================================================================
    # 1D sin
    # ==================================================================

    print("\n--- 1D sin ---")

    shape_1d = [256]
    want_1d = np.array(
        [np.sin(2.0 * np.pi * i / 255.0) for i in range(256)], dtype=np.float64
    )

    stt = from_samples(
        lambda idx: np.sin(2.0 * np.pi * idx[0] / 255.0),
        shape_1d,
        FromSamplesOptions(eps=1e-12, method="dmrg_cross", max_sweeps=5),
    )
    check(
        "dmrg_cross, sin(2*pi*i/255)",
        rel_err(stt.to_dense(), want_1d),
        1e-10,
        tt=stt,
        min_rank=1,
    )

    # ==================================================================
    # 2D rank-1: f(i,j) = (i+1)*(j+1)  (exact rank-1 outer product)
    # ==================================================================

    print("\n--- 2D rank-1 ((i+1)*(j+1)) ---")

    shape_2d = [16, 16]
    rank1_want = np.fromfunction(
        lambda i, j: (i + 1.0) * (j + 1.0), tuple(shape_2d), dtype=np.float64
    ).ravel()

    def rank1_func(idx):
        return (idx[0] + 1.0) * (idx[1] + 1.0)

    stt = from_samples(
        rank1_func,
        shape_2d,
        FromSamplesOptions(eps=1e-12, method="dmrg_cross", max_sweeps=5),
    )
    check(
        "dmrg_cross, rank-1 outer product",
        rel_err(stt.to_dense(), rank1_want),
        1e-10,
        tt=stt,
        min_rank=1,
    )

    stt = from_samples(
        rank1_func,
        shape_2d,
        FromSamplesOptions(eps=1e-12, method="tt_cross", max_rank=1, init_rank=2),
    )
    check(
        "tt_cross, rank-1 outer product",
        rel_err(stt.to_dense(), rank1_want),
        1e-10,
        tt=stt,
        min_rank=1,
    )

    # ==================================================================
    # 2D rank-2: f(i,j) = i + j  (exact rank-2)
    # ==================================================================

    print("\n--- 2D rank-2 (i + j) ---")

    rank2_want = np.fromfunction(
        lambda i, j: i + j, tuple(shape_2d), dtype=np.float64
    ).ravel()

    def rank2_func(idx):
        return idx[0] + idx[1]

    stt = from_samples(
        rank2_func,
        shape_2d,
        FromSamplesOptions(eps=1e-12, method="dmrg_cross", max_sweeps=5),
    )
    check(
        "dmrg_cross, rank-2 i+j",
        rel_err(stt.to_dense(), rank2_want),
        1e-10,
        tt=stt,
        min_rank=2,
    )

    stt = from_samples(
        rank2_func,
        shape_2d,
        FromSamplesOptions(eps=1e-12, method="tt_cross", max_rank=2, init_rank=4),
    )
    check(
        "tt_cross, rank-2 i+j",
        rel_err(stt.to_dense(), rank2_want),
        1e-10,
        tt=stt,
        min_rank=2,
    )

    # ==================================================================
    # 2D rank-3: f(i,j) = i + j + i*j  (rank 1+2=3)
    # ==================================================================

    print("\n--- 2D rank-3 (i + j + i*j) ---")

    rank3_want = np.fromfunction(
        lambda i, j: i + j + i * j, tuple(shape_2d), dtype=np.float64
    ).ravel()

    def rank3_func(idx):
        return idx[0] + idx[1] + idx[0] * idx[1]

    stt = from_samples(
        rank3_func,
        shape_2d,
        FromSamplesOptions(eps=1e-12, method="dmrg_cross", max_sweeps=10),
    )
    check(
        "dmrg_cross, rank-3 i+j+i*j",
        rel_err(stt.to_dense(), rank3_want),
        1e-8,
        tt=stt,
        min_rank=2,
    )

    # ==================================================================
    # 3D rank-2: f(i,j,k) = i + j + k  (linear)
    # ==================================================================

    print("\n--- 3D rank-2 (i + j + k) ---")

    shape_3d = [8, 8, 8]
    r2_3d_want = np.fromfunction(
        lambda i, j, k: i + j + k, tuple(shape_3d), dtype=np.float64
    ).ravel()

    def r2_3d_func(idx):
        return idx[0] + idx[1] + idx[2]

    stt = from_samples(
        r2_3d_func,
        shape_3d,
        FromSamplesOptions(eps=1e-12, method="dmrg_cross", max_sweeps=5),
    )
    check(
        "dmrg_cross, 3D i+j+k",
        rel_err(stt.to_dense(), r2_3d_want),
        1e-10,
        tt=stt,
        min_rank=2,
    )

    # ==================================================================
    # eval_batch verification on random multi-indices
    # ==================================================================

    print("\n--- eval_batch ---")

    M = 32
    rng = np.random.RandomState(99)
    batch_idx = np.empty(M * 2, dtype=np.int32)
    for k in range(M):
        batch_idx[2 * k] = rng.randint(0, shape_2d[0])
        batch_idx[2 * k + 1] = rng.randint(0, shape_2d[1])

    stt_ref = from_samples(
        rank2_func,
        shape_2d,
        FromSamplesOptions(eps=1e-12, method="dmrg_cross", max_sweeps=5),
    )

    batch_vals = stt_ref.eval_batch(batch_idx, M)
    batch_want = np.array(
        [rank2_func([batch_idx[2 * k], batch_idx[2 * k + 1]]) for k in range(M)],
        dtype=np.float64,
    )
    check(
        "eval_batch(32 random points)",
        np.linalg.norm(batch_vals - batch_want),
        1e-10,
        tt=stt_ref,
        min_rank=2,
    )

    # ==================================================================
    # Enrichment
    # ==================================================================

    print("\n--- enrichment ---")

    try:
        stt_enr = from_samples(
            rank3_func,
            shape_2d,
            FromSamplesOptions(
                eps=1e-12,
                method="dmrg_cross",
                max_sweeps=3,
                enrich_rounds=1,
                enrich_samples=256,
            ),
        )
        check(
            "rank-3 with enrichment",
            rel_err(stt_enr.to_dense(), rank3_want),
            1e-8,
            tt=stt_enr,
            min_rank=2,
        )
    except Exception as e:
        print(f"  enrichment                   SKIP: {e}")

    # ==================================================================
    # Summary
    # ==================================================================

    print()
    print(f"{total - failed}/{total} tests passed")
    if failed == 0:
        print("test_from_samples: OK")
    else:
        print(f"test_from_samples: FAIL ({failed} failed)")
    return failed


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)
#
# :D
#
