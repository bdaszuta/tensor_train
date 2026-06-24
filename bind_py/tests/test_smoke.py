"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Smoke test: import and basic round-trip
"""
import numpy as np
import pytest
from tensor_train import from_dense, TensorTrain, TensorTrainMatrix, RoundOptions
from tensor_train import zeros, ones, canonical_unit, random
from tensor_train import FromSamplesOptions, RoundResult


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


def test_warm_start_type_check():
    """Passing TensorTrainMatrix as warm_start to RoundOptions raises TypeError."""
    tt = zeros([2, 3])
    mpo = TensorTrainMatrix.identity([2, 3])
    with pytest.raises(TypeError):
        RoundOptions(warm_start=mpo, method="dmrg", eps=1e-4, max_iters=5)


def test_neg():
    tt = random([3, 4], max_rank=2, seed=99)
    neg_tt = -tt
    np.testing.assert_allclose(neg_tt.to_dense(), -tt.to_dense(), atol=1e-14)


def test_round_ex_matrix():
    A = TensorTrainMatrix.random([3, 4], [3, 4], max_rank=4, seed=7)
    result, info = A.round_ex(RoundOptions(method="svd", eps=1e-8))
    assert isinstance(result, TensorTrainMatrix)
    assert isinstance(info, RoundResult)
    assert info.iters_run >= 0


def test_from_samples_matrix():
    def func(row, col):
        return float(row[0] + col[0])
    A = TensorTrainMatrix.from_samples(
        func, [3], [3],
        FromSamplesOptions(method="dmrg_cross", eps=1e-12,
                           max_sweeps=3, init_rank=2))
    dense = A.to_dense().reshape(3, 3)
    expected = np.fromfunction(lambda r, c: r + c, (3, 3))
    assert np.max(np.abs(dense - expected)) < 1e-10


def test_orthogonalize():
    tt = random([3, 4, 5], max_rank=4, seed=13)
    ro = tt.right_orthogonalize()
    lo = tt.left_orthogonalize()
    # Orthogonalization preserves the tensor (up to roundoff)
    np.testing.assert_allclose(ro.to_dense(), tt.to_dense(), atol=1e-13)
    np.testing.assert_allclose(lo.to_dense(), tt.to_dense(), atol=1e-13)


def test_soft_threshold():
    tt = random([3, 4, 5], max_rank=4, seed=21)
    st = tt.soft_threshold(0.01 * tt.norm())
    assert st.norm() <= tt.norm() * (1.0 + 1e-14)


def test_frob_norm_apply():
    A = TensorTrainMatrix.random([3, 4], [3, 4], max_rank=3, seed=3)
    x = random([3, 4], max_rank=3, seed=5)
    fn1 = A.frob_norm_apply_matvec(x)
    fn2 = A.matvec(x).norm()
    assert abs(fn1 - fn2) < 1e-12 * max(1.0, fn2)
#
# :D
#
