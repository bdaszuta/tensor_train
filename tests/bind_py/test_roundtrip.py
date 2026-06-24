"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for roundtrip.
"""
import sys
import os as _os

_sys_dir = _os.path.dirname(_os.path.abspath(__file__))
_bind_dir = _os.path.join(_sys_dir, "..", "..", "bind_py")
if _bind_dir not in sys.path:
    sys.path.insert(0, _bind_dir)

import numpy as np
import tempfile

from tensor_train import (
    TensorTrain,
    TTCore,
    from_dense,
)


def rel_err(got, want):
    n = np.linalg.norm(want)
    return np.linalg.norm(got - want) / n if n > 0 else np.linalg.norm(got)


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

    shape = [3, 4, 5]
    rng = np.random.RandomState(42)
    dense = rng.random(int(np.prod(shape))).astype(np.float64)

    # -- Compress --------------------------------------------------------
    print("\n--- compress ---")
    tt = from_dense(dense, shape, eps=1e-12)
    check("compress accuracy", rel_err(tt.to_dense(), dense), 1e-10)

    # -- Extract core views (zero-copy) ----------------------------------
    print("\n--- extract cores (views) ---")
    core_views = tt.cores
    check("core count == d", 0.0 if len(core_views) == len(shape) else 1e9, 0.5)

    for k, arr in enumerate(core_views):
        check(f"core[{k}].shape == (r,n,rr)",
              0.0 if len(arr.shape) == 3 and arr.shape[1] == shape[k] else 1e9,
              0.5)
        if k == 0:
            check("core[0].shape[0] == 1",
                  0.0 if arr.shape[0] == 1 else 1e9, 0.5)
        if k == len(core_views) - 1:
            check("core[-1].shape[2] == 1",
                  0.0 if arr.shape[2] == 1 else 1e9, 0.5)

    # -- Views are read-only (verify immutability) -----------------------
    print("\n--- view immutability ---")
    try:
        core_views[0][0, 0, 0] = 99.0
        check("views are read-only", 1e9, 0.5)
    except (ValueError, TypeError):
        check("views are read-only", 0.0, 0.5)

    # -- Save to file ----------------------------------------------------
    print("\n--- save/load ---")
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp_path = f.name

    try:
        np.savez(tmp_path,
                 **{f"c{k}": arr for k, arr in enumerate(core_views)},
                 shape=np.array(shape, dtype=np.int32))

        # -- Load from file ----------------------------------------------
        loaded = np.load(tmp_path)
        shape_loaded = list(loaded["shape"])
        check("shape preserved", 0.0 if shape_loaded == shape else 1e9, 0.5)

        cores_loaded = [loaded[f"c{k}"] for k in range(len(shape_loaded))]

        # -- Rebuild -----------------------------------------------------
        tt2 = TensorTrain.from_cores(cores_loaded)
        check("rebuild shape", 0.0 if tt2.shape == tt.shape else 1e9, 0.5)
        check("rebuild ranks", 0.0 if tt2.ranks == tt.ranks else 1e9, 0.5)
        check("rebuild accuracy", rel_err(tt2.to_dense(), dense), 1e-12)
        check("rebuild vs original TT",
              rel_err(tt2.to_dense(), tt.to_dense()), 1e-12)

        # -- Round-trip via TTCore objects -------------------------------
        print("\n--- round-trip via TTCore ---")
        core_objs = [tt.core(k) for k in range(tt.d)]
        check("core(k) count", 0.0 if len(core_objs) == tt.d else 1e9, 0.5)
        for k, c in enumerate(core_objs):
            check(f"core[{k}] is TTCore",
                  0.0 if isinstance(c, TTCore) else 1e9, 0.5)
            check(f"core[{k}].data matches view",
                  np.linalg.norm(c.data - core_views[k]), 1e-14)

        tt3 = TensorTrain.from_cores(core_objs)
        check("from_cores(TTCore list)",
              rel_err(tt3.to_dense(), tt.to_dense()), 1e-12)

    finally:
        import os as _os2
        _os2.unlink(tmp_path)

    # -- Compression ratio -----------------------------------------------
    print("\n--- stats ---")
    n_dense = int(np.prod(shape))
    n_tt = tt.num_params
    print(f"  dense elements:       {n_dense}")
    print(f"  TT parameters:        {n_tt}")
    print(f"  compression ratio:    {n_dense / n_tt:.2f}x")
    print(f"  max rank:             {tt.max_rank}")

    # -- Verify eval_at matches -----------------------------------------
    print("\n--- evaluation ---")
    idx = [1, 2, 3]
    val_orig = tt.eval_at(idx)
    val_rebuilt = tt2.eval_at(idx)
    check("eval_at after rebuild", abs(val_orig - val_rebuilt), 1e-14)

    # -- View lifetime: verify views survive after tt deletion -----------
    print("\n--- view lifetime ---")
    view_copy = np.array(core_views[1], copy=True)
    del tt
    check("view survives tt deletion",
          np.linalg.norm(view_copy - core_views[1]), 1e-14)

    # -- Summary ---------------------------------------------------------
    print()
    print(f"{total - failed}/{total} tests passed")
    if failed == 0:
        print("test_roundtrip: OK")
    else:
        print(f"test_roundtrip: FAIL ({failed} failed)")
    return failed


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)
#
# :D
#
