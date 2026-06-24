"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Demonstration tests for the mva::tensor_train Python bindings
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
    from_dense,
    from_samples,
    RoundOptions,
    FromDenseOptions,
    FromSamplesOptions,
    RoundResult,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

SEED = 42


def dense_ref(shape, seed=SEED):
    """Deterministic dense tensor in row-major order."""
    rng = np.random.RandomState(seed)
    return rng.random(int(np.prod(shape))).astype(np.float64)


def rel_err(got, want):
    """Relative Frobenius error."""
    n = np.linalg.norm(want)
    return np.linalg.norm(got - want) / n if n > 0 else np.linalg.norm(got)


# -------------------------------------------------------------------
# Test runner
# -------------------------------------------------------------------


def main():
    failed = 0
    total = 0

    def check(name, err, tol):
        nonlocal failed, total
        total += 1
        ok = err <= tol
        status = "OK" if ok else "FAIL"
        print(f"  {name:<32s} err={err:.2e} [{status}]")
        if not ok:
            failed += 1

    # ==================================================================
    # Setup: two reference tensors
    # ==================================================================

    d2_shape = [2, 3]
    d2_data = dense_ref(d2_shape)

    d3_shape = [3, 2, 4]
    d3_data = dense_ref(d3_shape)

    print("\n--- from_dense ---")

    tt2 = from_dense(d2_data, d2_shape, eps=1e-12)
    check("from_dense(eps, 2D)", rel_err(tt2.to_dense(), d2_data), 1e-10)

    tt3 = from_dense(d3_data, d3_shape, opts=FromDenseOptions(eps=1e-12, method="svd"))
    check("from_dense(opts, 3D)", rel_err(tt3.to_dense(), d3_data), 1e-10)

    tt3b = from_dense(d3_data, d3_shape, opts={"eps": 1e-12, "method": "svd"})
    check("from_dense(dict, 3D)", rel_err(tt3b.to_dense(), d3_data), 1e-10)

    # Regression: verify opts=FromDenseOptions(...) is not silently dropped
    qtt_shape = [2, 2, 4, 4]
    qtt_data = dense_ref(qtt_shape, seed=99)
    tt_qtt_opts = from_dense(
        qtt_data, qtt_shape,
        opts=FromDenseOptions(method="qtt", qtt_base=2, eps=1e-12))
    check("from_dense(opts qtt) not dropped",
          0.0 if tt_qtt_opts.d > len(qtt_shape) else 1e9, 0.5)
    check("from_dense(opts qtt) accuracy",
          rel_err(tt_qtt_opts.to_dense(), qtt_data), 1e-10)

    # ==================================================================
    # Properties
    # ==================================================================

    print("\n--- properties ---")

    ok = True
    ok &= tuple(tt3.shape) == tuple(d3_shape)
    ok &= tt3.d == 3
    ok &= tt3.ranks[0] == 1
    ok &= tt3.ranks[-1] == 1
    ok &= tt3.max_rank >= 1
    check("shape / ranks / d / boundary", 0.0 if ok else 1e9, 0.5)

    # ==================================================================
    # Algebra
    # ==================================================================

    # ==================================================================
    # Factories
    # ==================================================================

    print("\n--- factories ---")

    from tensor_train import zeros, ones, canonical_unit, random

    z = zeros([2, 3, 4])
    check("zeros norm", z.norm(), 1e-14)

    o = ones([2, 3])
    check("ones eval", abs(o.eval_at([0, 1]) - 1.0), 1e-14)

    cu = canonical_unit([3, 4], [2, 1])
    cu_dense = cu.to_dense()
    li = 2 * 4 + 1
    check("canonical_unit at target", abs(cu_dense[li] - 1.0), 1e-14)
    check("canonical_unit elsewhere", np.sum(cu_dense) - 1.0, 1e-14)

    r = random([3, 4, 5], max_rank=8, seed=123)
    check("random ranks bounded", 0.0 if r.max_rank <= 8 else 1e9, 0.5)
    check("random boundary r0", 0.0 if r.ranks[0] == 1 else 1e9, 0.5)
    check("random boundary rd", 0.0 if r.ranks[-1] == 1 else 1e9, 0.5)

    print("\n--- algebra ---")

    a = from_dense(d2_data, d2_shape, eps=1e-12)
    b = from_dense(d2_data, d2_shape, eps=1e-12)

    check("add (a + b)", rel_err((a + b).to_dense(), 2.0 * d2_data), 1e-10)

    a_plus_b = a + b
    check(
        "add doubles ranks",
        0.0
        if all(r == 2 * s for r, s in zip(a_plus_b.ranks[1:-1], a.ranks[1:-1]))
        else 1e9,
        0.5,
    )

    check("sub (a - a)", np.linalg.norm((a - a).to_dense()), 1e-12)

    check("scale (2 * a)", rel_err((2.0 * a).to_dense(), 2.0 * d2_data), 1e-10)
    check("neg (-a)", rel_err((-a).to_dense(), -d2_data), 1e-10)

    check("axpy (a + 2*b)", rel_err(a.axpy(2.0, b).to_dense(), 3.0 * d2_data), 1e-10)

    h = a.hadamard(b)
    check("hadamard", rel_err(h.to_dense(), d2_data * d2_data), 1e-10)
    check(
        "hadamard multiplies ranks",
        0.0
        if all(
            r == s * t for r, s, t in zip(h.ranks[1:-1], a.ranks[1:-1], b.ranks[1:-1])
        )
        else 1e9,
        0.5,
    )

    # ==================================================================
    # Evaluation
    # ==================================================================

    print("\n--- evaluation ---")

    idx2 = [1, 1]
    li2 = idx2[0] * d2_shape[1] + idx2[1]
    check("eval_at (2D)", abs(a.eval_at(idx2) - d2_data[li2]), 1e-12)

    idx3 = [2, 1, 3]
    li3 = (idx3[0] * d3_shape[1] + idx3[1]) * d3_shape[2] + idx3[2]
    check("eval_at (3D)", abs(tt3.eval_at(idx3) - d3_data[li3]), 1e-12)

    M = 4
    batch_idx = np.array([0, 0, 0, 1, 1, 1, 1, 2], dtype=np.int32)
    batch = a.eval_batch(batch_idx, M)
    want_batch = np.empty(M, dtype=np.float64)
    for k in range(M):
        di = k * a.d
        want_batch[k] = d2_data[batch_idx[di] * a.shape[1] + batch_idx[di + 1]]
    check("eval_batch", np.linalg.norm(batch - want_batch), 1e-12)

    # ==================================================================
    # norm / inner
    # ==================================================================

    print("\n--- norm / inner ---")

    check("norm", abs(a.norm() - np.linalg.norm(d2_data)), 1e-10)
    check("inner", abs(a.inner(b) - np.dot(d2_data, d2_data)), 1e-10)

    # ==================================================================
    # round: SVD
    # ==================================================================

    print("\n--- round (SVD) ---")

    big = a + b
    svd_r = big.round(RoundOptions(eps=1e-8, method="svd"))
    check("round(svd, opts obj)", rel_err(svd_r.to_dense(), 2.0 * d2_data), 1e-7)

    svd_r2 = big.round({"eps": 1e-8, "method": "svd"})
    check("round(svd, dict)", rel_err(svd_r2.to_dense(), 2.0 * d2_data), 1e-7)

    svd_r3 = big.round(eps=1e-8, method="svd")
    check("round(svd, kwargs)", rel_err(svd_r3.to_dense(), 2.0 * d2_data), 1e-7)

    # ==================================================================
    # round: ALS (eps-driven, no rank cap)
    # ==================================================================

    print("\n--- round (ALS) ---")

    als_r = big.round(RoundOptions(eps=1e-4, method="als", max_iters=50, tol=1e-10))
    check("round(als, eps-driven)", rel_err(als_r.to_dense(), 2.0 * d2_data), 1e-7)

    # ==================================================================
    # round: DMRG (eps-driven)
    # ==================================================================

    print("\n--- round (DMRG) ---")

    dmrg_r = big.round(RoundOptions(eps=1e-8, method="dmrg", max_iters=20, tol=1e-12))
    check("round(dmrg, eps-driven)", rel_err(dmrg_r.to_dense(), 2.0 * d2_data), 1e-7)

    # ==================================================================
    # round_ex: diagnostics
    # ==================================================================

    print("\n--- round_ex ---")

    rx, info = big.round_ex({"eps": 1e-8, "method": "svd"})
    check("round_ex(result accuracy)", rel_err(rx.to_dense(), 2.0 * d2_data), 1e-7)
    check(
        "round_ex(iters_run==1, converged)",
        0.0 if info.iters_run == 1 and info.converged else 1e9,
        0.5,
    )

    rx2, info2 = big.round_ex(
        RoundOptions(eps=1e-4, method="als", max_iters=50, tol=1e-10)
    )
    check("round_ex(als) converged", 0.0 if info2.converged else 1e9, 0.5)
    check(
        "round_ex(als) final_resid >= 0", 0.0 if info2.final_resid >= 0.0 else 1e9, 0.5
    )

    # ==================================================================
    # from_dense: QTT (valid power-of-2 shape)
    # ==================================================================

    print("\n--- from_dense (QTT) ---")

    qtt_shape = [2, 2, 4, 4]
    qtt_data = dense_ref(qtt_shape, seed=99)
    try:
        qtt = from_dense(
            qtt_data, qtt_shape, opts=FromDenseOptions(method="qtt", qtt_base=2)
        )
        check("from_dense(qtt, base=2)", rel_err(qtt.to_dense(), qtt_data), 1e-10)
    except Exception as e:
        print(f"  from_dense(qtt)             SKIP: {e}")

    # ==================================================================
    # from_samples (cross interpolation)
    # ==================================================================

    print("\n--- from_samples ---")

    def separable(idx):
        return (idx[0] + 1.0) * (idx[1] + 1.0)

    s_shape = [4, 4]
    s_want = np.array(
        [[(i + 1) * (j + 1) for j in range(4)] for i in range(4)], dtype=np.float64
    ).ravel()

    stt = from_samples(
        separable,
        s_shape,
        FromSamplesOptions(eps=1e-12, method="dmrg_cross", max_sweeps=5),
    )
    check("from_samples(dmrg_cross, separable)", rel_err(stt.to_dense(), s_want), 1e-10)

    stt2 = from_samples(
        separable, s_shape, FromSamplesOptions(eps=1e-12, method="tt_cross", max_rank=1)
    )
    check("from_samples(tt_cross, separable)", rel_err(stt2.to_dense(), s_want), 1e-10)

    stt3 = from_samples(
        lambda idx: idx[0] + idx[1] * 2.0,
        s_shape,
        FromSamplesOptions(method="dmrg_cross", max_sweeps=5),
    )
    s_want3 = np.array(
        [[i + j * 2.0 for j in range(4)] for i in range(4)], dtype=np.float64
    ).ravel()
    check("from_samples(lambda)", rel_err(stt3.to_dense(), s_want3), 1e-10)

    stt4 = from_samples(
        separable, s_shape, {"eps": 1e-12, "method": "dmrg_cross", "max_sweeps": 5}
    )
    check("from_samples(dict opts)", rel_err(stt4.to_dense(), s_want), 1e-10)

    # Regression: kwargs merged on top of dict opts (not silently dropped)
    stt6 = from_samples(
        separable, s_shape,
        {"eps": 1e-12, "max_sweeps": 1},
        method="tt_cross", max_rank=1)
    check("from_samples(dict + kwargs merge)",
          rel_err(stt6.to_dense(), s_want), 1e-10)

    stt5 = from_samples(separable, s_shape, FromSamplesOptions(eps=1e-12, max_sweeps=5))
    check("from_samples(default method)", rel_err(stt5.to_dense(), s_want), 1e-10)

    s_batch_idx = np.array([0, 0, 0, 1, 2, 0, 3, 3], dtype=np.int32)
    s_batch = stt.eval_batch(s_batch_idx, 4)
    s_batch_want = np.array(
        [separable((0, 0)), separable((0, 1)), separable((2, 0)), separable((3, 3))],
        dtype=np.float64,
    )
    check("from_samples + eval_batch", np.linalg.norm(s_batch - s_batch_want), 1e-10)

    # ==================================================================
    # Summary
    # ==================================================================

    print()
    print(f"{total - failed}/{total} tests passed")
    if failed == 0:
        print("test_bindings: OK")
    else:
        print(f"test_bindings: FAIL ({failed} failed)")
    return failed


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)
#
# :D
#
