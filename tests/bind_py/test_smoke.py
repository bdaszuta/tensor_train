"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Smoke test: import and basic round-trip
"""
import sys
import os as _os

_sys_dir = _os.path.dirname(_os.path.abspath(__file__))
_bind_dir = _os.path.join(_sys_dir, "..", "..", "bind_py")
if _bind_dir not in sys.path:
    sys.path.insert(0, _bind_dir)

import numpy as np
from tensor_train import from_dense, TensorTrain, RoundOptions
from tensor_train import zeros, ones, canonical_unit, random


def test_from_dense_roundtrip():
    data = np.arange(2 * 3 * 4, dtype=np.float64)
    tt = from_dense(data, [2, 3, 4], eps=1e-12)
    recon = tt.to_dense()
    assert np.max(np.abs(recon - data)) < 1e-10


def test_zeros():
    z = zeros([2, 3, 4])
    assert z.norm() < 1e-14


def test_ones():
    o = ones([2, 3])
    d = o.to_dense()
    assert np.allclose(d, 1.0)


def test_canonical_unit():
    cu = canonical_unit([3, 4], [2, 1])
    d = cu.to_dense()
    assert abs(d[2 * 4 + 1] - 1.0) < 1e-14
    assert abs(np.sum(d) - 1.0) < 1e-14


def test_random():
    r = random([3, 4, 5], max_rank=8, seed=42)
    assert r.max_rank <= 8
    assert r.ranks[0] == 1
    assert r.ranks[-1] == 1


if __name__ == "__main__":
    failed = 0
    for name, fn in [
        ("from_dense_roundtrip", test_from_dense_roundtrip),
        ("zeros", test_zeros),
        ("ones", test_ones),
        ("canonical_unit", test_canonical_unit),
        ("random", test_random),
    ]:
        try:
            fn()
            print(f"  {name:<28s} [OK]")
        except Exception as e:
            failed += 1
            print(f"  {name:<28s} [FAIL] {e}")
    print()
    if failed == 0:
        print("test_smoke: OK")
    else:
        print(f"test_smoke: FAIL ({failed} failed)")
    sys.exit(0 if failed == 0 else 1)
#
# :D
#
