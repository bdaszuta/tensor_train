"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for TensorTrainMatrix (MPO) Python bindings
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
    TensorTrainMatrix,
    TTCoreMatrix,
    from_dense,
    RoundOptions,
    FromDenseOptions,
    zeros,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

SEED = 42


def dense_ref(shape, seed=SEED):
    rng = np.random.RandomState(seed)
    return rng.random(int(np.prod(shape))).astype(np.float64)


def rel_err(got, want):
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
        print(f"  {name:<42s} err={err:.2e} [{status}]")
        if not ok:
            failed += 1

    # ==================================================================
    # Construction
    # ==================================================================

    print("\n--- construction ---")

    # Identity
    I = TensorTrainMatrix.identity([2, 3, 4])
    dI = I.to_dense()
    eye = np.eye(int(np.prod([2, 3, 4]))).ravel()
    check("identity to_dense", np.linalg.norm(dI - eye), 1e-12)

    # Zeros
    Z = TensorTrainMatrix.zeros([2, 3], [2, 3])
    check("zeros norm", Z.frob_norm(), 1e-14)

    # Random
    R = TensorTrainMatrix.random([3, 4], [3, 4], max_rank=8, seed=123)
    check("random max_rank bounded", 0.0 if R.max_rank <= 8 else 1e9, 0.5)
    check("random boundary r0", 0.0 if R.ranks[0] == 1 else 1e9, 0.5)
    check("random boundary rd", 0.0 if R.ranks[-1] == 1 else 1e9, 0.5)

    # Diag from tt
    v = zeros([4, 5])
    D = TensorTrainMatrix.diag_from_tt(v)
    check("diag d matches", 0.0 if D.d == v.d else 1e9, 0.5)

    # from_dense
    shape = [2, 3]
    data = dense_ref([int(np.prod(shape)), int(np.prod(shape))], seed=99)
    A = TensorTrainMatrix.from_dense(data, shape, shape, eps=1e-12)
    check("from_dense d", 0.0 if A.d == len(shape) else 1e9, 0.5)
    check("from_dense accuracy", rel_err(A.to_dense().ravel(), data), 1e-10)

    # Regression: verify opts=FromDenseOptions(...) works (was crashing)
    A_opts = TensorTrainMatrix.from_dense(
        data, shape, shape,
        opts=FromDenseOptions(eps=1e-12, method="svd"))
    check("from_dense(opts) no crash",
          rel_err(A_opts.to_dense().ravel(), data), 1e-10)

    # ==================================================================
    # Properties
    # ==================================================================

    print("\n--- properties ---")

    check("d", 0.0 if A.d == 2 else 1e9, 0.5)
    check("len", 0.0 if len(A) == 2 else 1e9, 0.5)
    check("row_shape", 0.0 if A.row_shape == shape else 1e9, 0.5)
    check("col_shape", 0.0 if A.col_shape == shape else 1e9, 0.5)
    check("total_rows", 0.0 if A.total_rows == 6 else 1e9, 0.5)
    check("total_cols", 0.0 if A.total_cols == 6 else 1e9, 0.5)
    check("ranks boundary r0", 0.0 if A.ranks[0] == 1 else 1e9, 0.5)
    check("ranks boundary rd", 0.0 if A.ranks[-1] == 1 else 1e9, 0.5)

    # ==================================================================
    # Algebra
    # ==================================================================

    print("\n--- algebra ---")

    B = TensorTrainMatrix.from_dense(data, shape, shape, eps=1e-12)

    C = A + B
    check("add accuracy", rel_err(C.to_dense().ravel(), 2.0 * data), 1e-10)
    check("add doubles ranks", 0.0 if all(
        r == 2 * s for r, s in zip(C.ranks[1:-1], A.ranks[1:-1])
    ) else 1e9, 0.5)

    Zsub = A - A
    check("sub (A-A) to_dense", np.linalg.norm(Zsub.to_dense()), 1e-12)

    check("scale (2*A) accuracy", rel_err(
        (2.0 * A).to_dense().ravel(), 2.0 * data), 1e-10)

    check("neg (-A) accuracy", rel_err(
        (-A).to_dense().ravel(), -data), 1e-10)

    check("axpy (A+2*B) accuracy", rel_err(
        A.axpy(2.0, B).to_dense().ravel(), 3.0 * data), 1e-10)

    check("axpby (A+0.5*B) accuracy", rel_err(
        A.axpby(1.0, 0.5, B).to_dense().ravel(), 1.5 * data), 1e-10)

    # ==================================================================
    # Inner products
    # ==================================================================

    print("\n--- inner products ---")

    n = A.frob_norm()
    check("frob_norm", abs(n - np.linalg.norm(data)), 1e-10)

    fi = A.frob_inner(B)
    check("frob_inner", abs(fi - np.dot(data, data)), 1e-10)

    # ==================================================================
    # matvec
    # ==================================================================

    print("\n--- matvec ---")

    x_data = dense_ref(shape, seed=77)
    x = from_dense(x_data, shape, eps=1e-12)

    y = A.matvec(x)
    check("matvec shape", 0.0 if y.shape == shape else 1e9, 0.5)

    # Verify against dense: A_dense @ x_dense
    A_dense = data.reshape(A.total_rows, A.total_cols)
    y_ref = A_dense @ x_data
    check("matvec accuracy", rel_err(y.to_dense(), y_ref), 1e-10)

    # Identity matvec: I_id @ x_id == x_id (use 2D for consistency)
    I2 = TensorTrainMatrix.identity(shape)
    y_id = I2.matvec(x)
    check("identity matvec", rel_err(y_id.to_dense(), x_data), 1e-10)

    # ==================================================================
    # matvec_round
    # ==================================================================

    print("\n--- matvec_round ---")

    yr = A.matvec_round(x, RoundOptions(eps=1e-8))
    check("matvec_round accuracy", rel_err(yr.to_dense(), y_ref), 1e-7)

    # ==================================================================
    # Round
    # ==================================================================

    print("\n--- round ---")

    Cbig = A + B  # doubled ranks
    Cr = Cbig.round(RoundOptions(eps=1e-8, method="svd"))
    check("round accuracy", rel_err(Cr.to_dense().ravel(), 2.0 * data), 1e-7)
    check("round reduces ranks", 0.0 if Cr.max_rank <= Cbig.max_rank else 1e9, 0.5)

    # ==================================================================
    # Core extraction and round-trip
    # ==================================================================

    print("\n--- core round-trip ---")

    core_views = A.cores
    check("core count == d", 0.0 if len(core_views) == A.d else 1e9, 0.5)

    for k, arr in enumerate(core_views):
        check(f"core[{k}] is 4D", 0.0 if len(arr.shape) == 4 else 1e9, 0.5)
        check(f"core[{k}].shape[1] == m_k",
              0.0 if arr.shape[1] == shape[k] else 1e9, 0.5)
        check(f"core[{k}].shape[2] == n_k",
              0.0 if arr.shape[2] == shape[k] else 1e9, 0.5)

    A2 = TensorTrainMatrix.from_cores(core_views)
    check("round-trip accuracy",
          rel_err(A2.to_dense().ravel(), A.to_dense().ravel()), 1e-12)

    # from_cores via TTCoreMatrix objects from core(k) calls
    c0 = A.core(0)
    check("core(0) is TTCoreMatrix",
          0.0 if isinstance(c0, TTCoreMatrix) else 1e9, 0.5)
    check("core(0) repr",
          0.0 if "TTCoreMatrix" in repr(c0) else 1e9, 0.5)
    check("core(0).data matches cores[0]",
          np.linalg.norm(c0.data - A.cores[0]), 1e-14)

    core_objs = [A.core(k) for k in range(A.d)]
    A3 = TensorTrainMatrix.from_cores(core_objs)
    check("round-trip via TTCoreMatrix list",
          rel_err(A3.to_dense().ravel(), A.to_dense().ravel()), 1e-12)

    # ==================================================================
    # Validation: from_cores rejects invalid input
    # ==================================================================

    print("\n--- validation ---")

    try:
        bad = TensorTrainMatrix.from_cores([
            np.ones((2, 3, 3, 4), dtype=np.float64),
            np.ones((4, 3, 3, 1), dtype=np.float64),
        ])
        check("invalid from_cores caught", 1e9, 0.5)
    except RuntimeError as e:
        check("invalid from_cores caught", 0.0, 0.5)

    # ==================================================================
    # Summary
    # ==================================================================

    print()
    print(f"{total - failed}/{total} tests passed")
    if failed == 0:
        print("test_tt_matrix_bindings: OK")
    else:
        print(f"test_tt_matrix_bindings: FAIL ({failed} failed)")
    return failed


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)
#
# :D
#
