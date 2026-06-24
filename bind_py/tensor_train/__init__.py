"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Python bindings for mva::tensor_train
"""
import os as _os
import sys as _sys

_build_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "build")
if _build_dir not in _sys.path:
    _sys.path.insert(0, _build_dir)

import numpy as np

from tt_core import (  # type: ignore[import]  # noqa: F811
    from_dense as _from_dense,
    from_samples as _from_samples,
    round as _round,
    round_ex as _round_ex,
    add as _add,
    sub as _sub,
    scale as _scale,
    axpy as _axpy,
    axpby as _axpby,
    hadamard as _hadamard,
    inner as _inner,
    norm as _norm,
    eval_at as _eval_at,
    eval_batch as _eval_batch,
    tt as _Ctt,
    round_options as _Cround_options,
    from_dense_options as _Cfrom_dense_options,
    from_samples_options as _Cfrom_samples_options,
    round_result as _Cround_result,
    round_method as _Cround_method,
    round_gauge as _Cround_gauge,
    from_dense_method as _Cfrom_dense_method,
    from_samples_method as _Cfrom_samples_method,
    tt_core as _Ctt_core,
    from_cores as _from_cores,
    zeros as _zeros,
    ones as _ones,
    canonical_unit as _canonical_unit,
    random as _random,
    tt_matrix_core as _Ctt_matrix_core,
    tt_matrix as _Ctt_matrix,
    from_matrix_cores as _from_matrix_cores,
    zeros_matrix as _zeros_matrix,
    identity_matrix as _identity_matrix,
    random_matrix as _random_matrix,
    diag_from_tt as _diag_from_tt,
    matrix_from_dense as _matrix_from_dense,
    add_matrix as _add_matrix,
    sub_matrix as _sub_matrix,
    scale_matrix as _scale_matrix,
    axpy_matrix as _axpy_matrix,
    axpby_matrix as _axpby_matrix,
    frob_inner as _frob_inner,
    frob_norm_matrix as _frob_norm_matrix,
    matvec as _matvec,
    matmat as _matmat,
    round_matrix as _round_matrix,
    matvec_round as _matvec_round,
    matvec_round_ex as _matvec_round_ex,
    matmat_round as _matmat_round,
    matmat_round_ex as _matmat_round_ex,
    neg as _neg,
    neg_matrix as _neg_matrix,
    round_matrix_ex as _round_matrix_ex,
    from_samples_matrix as _from_samples_matrix,
    right_orthogonalize as _right_orthogonalize,
    left_orthogonalize as _left_orthogonalize,
    right_orthogonalize_matrix as _right_orthogonalize_matrix,
    soft_threshold as _soft_threshold,
    frob_norm_apply_matvec as _frob_norm_apply_matvec,
    frob_norm_apply_matmat as _frob_norm_apply_matmat,
)


__all__ = [
    "FromDenseOptions",
    "FromSamplesOptions",
    "RoundOptions",
    "RoundResult",
    "TTCore",
    "TTCoreMatrix",
    "TensorTrain",
    "TensorTrainMatrix",
    "canonical_unit",
    "from_dense",
    "from_samples",
    "from_dense_method",
    "from_samples_method",
    "neg",
    "neg_matrix",
    "ones",
    "random",
    "round_gauge",
    "round_method",
    "zeros",
]

# -------------------------------------------------------------------
# Options helpers
# -------------------------------------------------------------------


def _to_enum(enum_cls, value):
    """Convert a string or enum value to the nanobind enum."""
    if isinstance(value, str):
        return getattr(enum_cls, value)
    return value


class _OptionsMixin:
    """Options base that accepts keyword args or a dict."""

    def __init__(self, **kwargs):
        if len(kwargs) == 1:
            val = next(iter(kwargs.values()))
            if isinstance(val, dict):
                kwargs = val
        for key, value in kwargs.items():
            setattr(self, key, value)


class RoundOptions(_OptionsMixin):
    """
    Options for round() compression.

    Parameters
    ----------
    eps : float
      SVD truncation epsilon (default 0.0).
    max_rank : int
      Max bond rank (0 = no cap).
    method : str or round_method
      Compression method: ``"svd"``, ``"als"``, or ``"dmrg"``
      (valid for ``round()``).  ``"svd_streaming"`` and
      ``"svd_naive"`` are only valid for ``matvec_round()`` and
      ``matmat_round()`` respectively; they are rejected by
      ``round()``.
    max_iters : int
      Max iterations for ALS / DMRG (default 20).
    tol : float
      Convergence tolerance for ALS / DMRG (default 1e-12).
    seed : int
      Random seed for warm-start synthesis (default 0).
    gauge : str or round_gauge
      Output gauge for round() when method is ``"svd"``:
      ``"none"`` (default, no guarantee) or
      ``"right_canonical"`` (cores 1..d-1 right-orthogonal).
    warm_start : TensorTrain or None
      Initial guess for ALS / DMRG (default None).  Must be a
      TensorTrain with compatible shape and ranks.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if hasattr(self, "method"):
            self.method = _to_enum(_Cround_method, self.method)
        if hasattr(self, "gauge"):
            self.gauge = _to_enum(_Cround_gauge, self.gauge)
        if hasattr(self, "warm_start"):
            if self.warm_start is not None:
                if not isinstance(self.warm_start, TensorTrain):
                    raise TypeError(
                        "warm_start must be a TensorTrain or None, "
                        f"got {type(self.warm_start).__name__}"
                    )

    def _as_c(self):
        opts = _Cround_options()
        for attr in ("eps", "max_rank", "max_iters", "tol", "seed", "gauge"):
            if hasattr(self, attr):
                setattr(opts, attr, getattr(self, attr))
        if hasattr(self, "method"):
            opts.method = self.method
        return opts


class FromDenseOptions(_OptionsMixin):
    """
    Options for from_dense() construction.

    Parameters
    ----------
    eps : float
      Truncation epsilon (default 1e-10).
    max_rank : int
      Max bond rank (0 = no cap).
    method : str or from_dense_method
      Construction method: ``"svd"`` or ``"qtt"``.
    qtt_base : int
      Base for QTT (default 2).
    validate : bool
      Validate QTT shape (default True).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if hasattr(self, "method"):
            self.method = _to_enum(_Cfrom_dense_method, self.method)

    def _as_c(self):
        opts = _Cfrom_dense_options()
        for attr in ("eps", "max_rank", "qtt_base", "validate"):
            if hasattr(self, attr):
                setattr(opts, attr, getattr(self, attr))
        if hasattr(self, "method"):
            opts.method = self.method
        return opts


class FromSamplesOptions(_OptionsMixin):
    """
    Options for from_samples() cross interpolation.

    Parameters
    ----------
    eps : float
      Truncation tolerance (default 1e-10).
    max_rank : int
      Max bond rank (0 = no cap).
    rmin : int
      Minimum bond rank enforced during truncation (default 1).
    method : str or from_samples_method
      Cross method: ``"dmrg_cross"``, ``"tt_cross"``, ``"als_cross"``,
      ``"amen_cross"``, ``"qtt_dmrg_cross"``, ``"qtt_tt_cross"``,
      ``"qtt_als_cross"``, or ``"qtt_amen_cross"``.
    max_sweeps : int
      Max DMRG-cross sweeps (default 10).
    init_rank : int
      Initial rank for index sets (default 2).
    seed : int
      Random seed (default 0).
    maxvol_resolve_interval : int
      Maxvol re-solve frequency (default 16).
    enrich_rank : int
      Number of enrichment columns per supercore (default 4).
    use_qr_pivots : bool
      Use QR column-pivoting instead of maxvol for pivot selection
      (default False).  Recommended for d >= 10 or functions with
      large singular-value gaps.
    enrich_rounds : int
      Number of residual enrichment passes after cross approximation
      (default 0).
    enrich_samples : int
      Number of random validation points per enrichment round
      (default 1024).
    enrich_k : int
      Number of worst-offender points kept per enrichment round
      (default 32).
    qtt_base : int
      Base for QTT (default 2).
    qtt_maxbits : list[int]
      Bits per raw dimension for QTT.
    qtt_pool_modes : int
      Pool this many binary modes together (default 3).
    qtt_row_maxbits : list[int]
      Bits per row dimension for matrix QTT.
    qtt_col_maxbits : list[int]
      Bits per column dimension for matrix QTT.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if hasattr(self, "method"):
            self.method = _to_enum(_Cfrom_samples_method, self.method)

    def _as_c(self):
        opts = _Cfrom_samples_options()
        for attr in (
            "eps",
            "max_rank",
            "rmin",
            "method",
            "max_sweeps",
            "init_rank",
            "seed",
            "maxvol_resolve_interval",
            "enrich_rank",
            "use_qr_pivots",
            "enrich_rounds",
            "enrich_samples",
            "enrich_k",
            "qtt_base",
            "qtt_pool_modes",
        ):
            if hasattr(self, attr):
                setattr(opts, attr, getattr(self, attr))
        if hasattr(self, "qtt_maxbits"):
            opts.qtt_maxbits = list(self.qtt_maxbits)
        if hasattr(self, "qtt_row_maxbits"):
            opts.qtt_row_maxbits = list(self.qtt_row_maxbits)
        if hasattr(self, "qtt_col_maxbits"):
            opts.qtt_col_maxbits = list(self.qtt_col_maxbits)
        return opts


class RoundResult:
    """Result of a round() call with diagnostics."""

    def __init__(self, c_result):
        self.iters_run = c_result.iters_run
        self.final_resid = c_result.final_resid
        self.converged = c_result.converged


# -------------------------------------------------------------------
# TTCore
# -------------------------------------------------------------------


class TTCore:
    """Single TT core of shape (r_left, n_phys, r_right).

    Created manually or extracted via TensorTrain.core(k) (returns a copy).
    For bulk zero-copy extraction, use TensorTrain.cores.
    """

    def __init__(self, r_left, n_phys, r_right, data=None):
        self._c = _Ctt_core(r_left, n_phys, r_right)
        if data is not None:
            self._c.fill(np.asarray(data, dtype=np.float64).reshape(
                r_left, n_phys, r_right))

    @property
    def r_left(self):
        return self._c.r_left

    @property
    def n_phys(self):
        return self._c.n_phys

    @property
    def r_right(self):
        return self._c.r_right

    @property
    def dims(self):
        return tuple(self._c.dims)

    @property
    def size(self):
        return self._c.size

    @property
    def data(self):
        """Read-only 3D numpy view into core data (zero-copy).

        Shape: (r_left, n_phys, r_right).  The view is valid as long
        as this TTCore object exists.
        """
        return self._c.data_view()

    @classmethod
    def _from_c(cls, c_core):
        """Internal: wrap a C++ tt_core copy."""
        obj = cls.__new__(cls)
        obj._c = c_core
        return obj

    def __repr__(self):
        return f"TTCore(dims=({self.r_left}, {self.n_phys}, {self.r_right}))"


# -------------------------------------------------------------------
# TTCoreMatrix
# -------------------------------------------------------------------


class TTCoreMatrix:
    """Single TT-matrix core of shape (r_left, m_phys, n_phys, r_right).

    Created manually or extracted via TensorTrainMatrix.core(k)
    (returns a copy).  For bulk zero-copy extraction, use
    TensorTrainMatrix.cores.
    """

    def __init__(self, r_left, m_phys, n_phys, r_right, data=None):
        self._c = _Ctt_matrix_core(r_left, m_phys, n_phys, r_right)
        if data is not None:
            self._c.fill(np.asarray(data, dtype=np.float64).reshape(
                r_left, m_phys, n_phys, r_right))

    @property
    def r_left(self):
        return self._c.r_left

    @property
    def m_phys(self):
        return self._c.m_phys

    @property
    def n_phys(self):
        return self._c.n_phys

    @property
    def r_right(self):
        return self._c.r_right

    @property
    def dims(self):
        return tuple(self._c.dims)

    @property
    def size(self):
        return self._c.size

    @property
    def data(self):
        """Read-only 4D numpy view into core data (zero-copy).

        Shape: (r_left, m_phys, n_phys, r_right).  The view is valid
        as long as this TTCoreMatrix object exists.
        """
        return self._c.data_view()

    @classmethod
    def _from_c(cls, c_core):
        """Internal: wrap a C++ tt_matrix_core copy."""
        obj = cls.__new__(cls)
        obj._c = c_core
        return obj

    def __repr__(self):
        return (
            f"TTCoreMatrix(dims=({self.r_left}, {self.m_phys}, "
            f"{self.n_phys}, {self.r_right}))")


# -------------------------------------------------------------------
# TensorTrain
# -------------------------------------------------------------------


class TensorTrain:
    """
    Tensor-train decomposition of a multi-dimensional array.

    Created via ``from_dense()`` and manipulated with algebra / round.
    """

    def __init__(self, c_tt):
        self._c = c_tt

    # -- properties --

    @property
    def shape(self):
        """List of mode sizes."""
        return list(self._c.shape)

    @property
    def ranks(self):
        """List of TT ranks (length d + 1)."""
        return list(self._c.ranks)

    @property
    def d(self):
        """Number of modes (tensor order)."""
        return self._c.d

    def __len__(self):
        """Number of modes (tensor order)."""
        return self._c.d

    @property
    def max_rank(self):
        """Maximum bond rank."""
        return self._c.max_rank

    @property
    def num_params(self):
        """Total number of scalar parameters across all cores."""
        return self._c.num_params

    @property
    def cores(self):
        """List of read-only 3D numpy views, one per core (zero-copy).

        Each array has shape (r_k, n_k, r_{k+1}).  The views share
        a single copy of the TT data and remain valid until all are
        garbage-collected.  For individual core objects, use core(k).
        """
        return self._c.cores()

    def core(self, k):
        """Return a copy of core k as a TTCore object.

        Use this to inspect or modify a single core independently
        of the parent TT.  For bulk zero-copy extraction, use .cores.
        """
        return TTCore._from_c(self._c.core(k))

    @staticmethod
    def from_cores(cores):
        """Build a TensorTrain from a list of TTCore objects or 3D numpy arrays.

        Parameters
        ----------
        cores : list[TTCore | ndarray]
            One entry per mode.  Each core has shape (r_k, n_k, r_{k+1})
            with r_0 = r_d = 1.

        Returns
        -------
        TensorTrain
        """
        arrays = []
        for c in cores:
            if isinstance(c, TTCore):
                arrays.append(c.data)
            else:
                arrays.append(np.asarray(c, dtype=np.float64))
        return TensorTrain(_from_cores(arrays))

    # -- serialization --

    def to_dense(self):
        """Reconstruct the full dense tensor (row-major)."""
        return self._c.to_dense()

    # -- metrics --

    def norm(self):
        """Frobenius norm."""
        return _norm(self._c)

    def inner(self, other):
        """Inner product ``<self, other>``."""
        if isinstance(other, TensorTrain):
            other = other._c
        return _inner(self._c, other)

    # -- evaluation --

    def eval_at(self, idx):
        """Evaluate at a single multi-index."""
        return _eval_at(self._c, idx)

    def eval_batch(self, indices, M=None):
        """Evaluate at M multi-indices (flat array, length M*d).

        Parameters
        ----------
        indices : ndarray of int32, shape (M*d,)
          Flat array of multi-indices.
        M : int, optional
          Number of multi-indices.  Auto-detected from indices length
          if None (default).  Explicit M is validated against the
          inferred count.

        Returns
        -------
        ndarray of float64, shape (M,)
        """
        indices = np.asarray(indices, dtype=np.int32)
        if M is None:
            if len(indices) % self.d != 0:
                raise ValueError(
                    f"eval_batch: indices length {len(indices)} not "
                    f"divisible by d={self.d}")
            M = len(indices) // self.d
        elif M * self.d != len(indices):
            raise ValueError(
                f"eval_batch: M={M} * d={self.d} != "
                f"len(indices)={len(indices)}")
        return _eval_batch(self._c, indices, M)

    # -- compression --

    def round(self, opts=None, **kwargs):
        """
        Compress the TT.

        Parameters
        ----------
        opts : RoundOptions, dict, or kwargs
          Compression options.  If a plain dict is passed, it is
          treated as keyword arguments to RoundOptions.

        Returns
        -------
        TensorTrain
        """
        return self._round_impl(opts, **kwargs)

    def round_ex(self, opts=None, **kwargs):
        """
        Compress with diagnostics.

        Returns
        -------
        tuple[TensorTrain, RoundResult]
        """
        return self._round_impl(opts, _ex=True, **kwargs)

    def _round_impl(self, opts, _ex=False, **kwargs):
        """Shared opts dispatch for tt compression methods.

        Parameters
        ----------
        opts : RoundOptions
            Compression options object.
        _ex : bool
            If True, return (result, RoundResult) tuple instead of
            just the result.
        **kwargs
            Additional keyword overrides applied on top of opts.

        Returns
        -------
        TensorTrain or tuple[TensorTrain, RoundResult]
        """
        if isinstance(opts, RoundOptions):
            c_opts = opts._as_c()
            # Merge any additional kwargs as overrides on top of the opts object.
            for k, v in kwargs.items():
                if k == "method":
                    v = _to_enum(_Cround_method, v)
                elif k == "gauge":
                    v = _to_enum(_Cround_gauge, v)
                setattr(c_opts, k, v)
            warm = None
            if hasattr(opts, "warm_start") and opts.warm_start is not None:
                warm = opts.warm_start._c
        elif isinstance(opts, dict):
            opts_copy = dict(opts)
            warm = opts_copy.pop("warm_start", None)
            if warm is not None and hasattr(warm, "_c"):
                warm = warm._c
            c_opts = RoundOptions(**opts_copy)._as_c()
        elif opts is None:
            warm_obj = kwargs.pop("warm_start", None)
            warm = warm_obj._c if warm_obj is not None and hasattr(
                warm_obj, "_c") else None
            c_opts = RoundOptions(**kwargs)._as_c()
        else:
            warm = None
            c_opts = RoundOptions(**kwargs)._as_c()
        if _ex:
            if warm is not None:
                result, info = _round_ex(self._c, c_opts, warm)
            else:
                result, info = _round_ex(self._c, c_opts)
            return TensorTrain(result), RoundResult(info)
        if warm is not None:
            return TensorTrain(_round(self._c, c_opts, warm))
        return TensorTrain(_round(self._c, c_opts))

    # -- orthogonalization --

    def right_orthogonalize(self):
        """Return a copy with cores[1..d-1] right-orthogonal."""
        return TensorTrain(_right_orthogonalize(self._c))

    def left_orthogonalize(self):
        """Return a copy with cores[0..d-2] left-orthogonal."""
        return TensorTrain(_left_orthogonalize(self._c))

    # -- thresholding --

    def soft_threshold(self, tau):
        """Singular-value soft-thresholding: new_SV = max(SV - tau, 0)."""
        return TensorTrain(_soft_threshold(self._c, tau))

    # -- algebra --

    def scale(self, alpha):
        """Scalar multiplication: ``alpha * self``."""
        return TensorTrain(_scale(self._c, alpha))

    def __neg__(self):
        return TensorTrain(_neg(self._c))

    def __mul__(self, alpha):
        if isinstance(alpha, (int, float)):
            return TensorTrain(_scale(self._c, float(alpha)))
        return NotImplemented

    def __rmul__(self, alpha):
        return self.__mul__(alpha)

    def __add__(self, other):
        if isinstance(other, TensorTrain):
            return TensorTrain(_add(self._c, other._c))
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, TensorTrain):
            return TensorTrain(_sub(self._c, other._c))
        return NotImplemented

    def axpy(self, alpha, other):
        """``alpha * self + other``."""
        if isinstance(other, TensorTrain):
            other = other._c
        return TensorTrain(_axpy(alpha, self._c, other))

    def axpby(self, alpha, beta, other):
        """``alpha * self + beta * other``."""
        if isinstance(other, TensorTrain):
            other = other._c
        return TensorTrain(_axpby(alpha, self._c, beta, other))

    def hadamard(self, other):
        """Element-wise product."""
        if isinstance(other, TensorTrain):
            other = other._c
        return TensorTrain(_hadamard(self._c, other))

    def __repr__(self):
        return f"TensorTrain(shape={self.shape}, max_rank={self.max_rank}, d={self.d})"



class TensorTrainMatrix:
    """Tensor-train matrix (MPO / matrix product operator).

    Represents a linear operator A of size
    (prod_k m_k) x (prod_k n_k) in TT-matrix format.
    """

    def __init__(self, c_mpo):
        self._c = c_mpo

    # -- properties --
    @property
    def d(self):
        """Number of modes (operator order)."""
        return self._c.d

    def __len__(self):
        """Number of modes (operator order)."""
        return self._c.d

    @property
    def row_shape(self):
        """List of row-mode sizes."""
        return list(self._c.row_shape)

    @property
    def col_shape(self):
        """List of column-mode sizes."""
        return list(self._c.col_shape)

    @property
    def ranks(self):
        """List of TT-matrix ranks (length d + 1)."""
        return list(self._c.ranks)

    @property
    def max_rank(self):
        """Maximum bond rank."""
        return self._c.max_rank

    @property
    def total_rows(self):
        """Total number of rows (product of m_k)."""
        return self._c.total_rows

    @property
    def total_cols(self):
        """Total number of columns (product of n_k)."""
        return self._c.total_cols

    @property
    def num_params(self):
        """Total number of scalar parameters across all cores."""
        return self._c.num_params

    @property
    def cores(self):
        """List of read-only 4D numpy views (zero-copy)."""
        return self._c.cores()

    def core(self, k):
        """Return a copy of core k as a TTCoreMatrix object.

        Use this to inspect or modify a single core independently
        of the parent TT-matrix.  For bulk zero-copy extraction,
        use .cores.
        """
        return TTCoreMatrix._from_c(self._c.core(k))

    # -- serialization --
    def to_dense(self):
        """Reconstruct the full dense matrix (row-major)."""
        return self._c.to_dense()

    # -- metrics --
    def frob_norm(self):
        """Frobenius norm."""
        return _frob_norm_matrix(self._c)

    def frob_inner(self, other):
        """Frobenius inner product <self, other>."""
        if isinstance(other, TensorTrainMatrix):
            other = other._c
        return _frob_inner(self._c, other)

    # -- algebra --
    def __add__(self, other):
        if isinstance(other, TensorTrainMatrix):
            return TensorTrainMatrix(_add_matrix(self._c, other._c))
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, TensorTrainMatrix):
            return TensorTrainMatrix(_sub_matrix(self._c, other._c))
        return NotImplemented

    def __neg__(self):
        return TensorTrainMatrix(_scale_matrix(self._c, -1.0))

    def __mul__(self, alpha):
        if isinstance(alpha, (int, float)):
            return TensorTrainMatrix(_scale_matrix(self._c, float(alpha)))
        return NotImplemented

    def __rmul__(self, alpha):
        return self.__mul__(alpha)

    def scale(self, alpha):
        """Scalar multiplication: alpha * self."""
        return TensorTrainMatrix(_scale_matrix(self._c, alpha))

    def axpy(self, alpha, other):
        """Return alpha * self + other."""
        if isinstance(other, TensorTrainMatrix):
            other = other._c
        return TensorTrainMatrix(_axpy_matrix(alpha, self._c, other))

    def axpby(self, alpha, beta, other):
        """Return alpha * self + beta * other."""
        if isinstance(other, TensorTrainMatrix):
            other = other._c
        return TensorTrainMatrix(_axpby_matrix(alpha, self._c, beta, other))

    # -- apply --
    def matvec(self, x):
        """Matrix-vector product A * x (rank-multiplying; round afterwards)."""
        if isinstance(x, TensorTrain):
            x = x._c
        return TensorTrain(_matvec(self._c, x))

    def matmat(self, other):
        """Matrix-matrix product A * B (rank-multiplying; round afterwards)."""
        if isinstance(other, TensorTrainMatrix):
            other = other._c
        return TensorTrainMatrix(_matmat(self._c, other))

    # -- fast norm of apply --

    def frob_norm_apply_matvec(self, x):
        """||A * x||_F without materializing the full result."""
        if isinstance(x, TensorTrain):
            x = x._c
        return _frob_norm_apply_matvec(self._c, x)

    def frob_norm_apply_matmat(self, other):
        """||A * B||_F without materializing the full result."""
        if isinstance(other, TensorTrainMatrix):
            other = other._c
        return _frob_norm_apply_matmat(self._c, other)

    # -- gauge --

    def right_orthogonalize(self):
        """Return a copy with cores[1..d-1] right-orthogonal."""
        return TensorTrainMatrix(_right_orthogonalize_matrix(self._c))

    # -- compression --
    def round(self, opts=None, **kwargs):
        """Compress this TT-matrix via the method specified in opts."""
        return TensorTrainMatrix(
            self._matrix_round_impl(
                opts, _round_matrix, **kwargs))

    def round_ex(self, opts=None, **kwargs):
        """Compress with diagnostics. Returns (TensorTrainMatrix, RoundResult)."""
        result, info = self._matrix_round_impl(
            opts, _round_matrix_ex, **kwargs)
        return TensorTrainMatrix(result), RoundResult(info)

    def matvec_round(self, x, opts=None, **kwargs):
        """Fused matvec + round: A * x followed by compression."""
        if isinstance(x, TensorTrain):
            x = x._c
        return TensorTrain(
            self._matrix_round_impl(
                opts, _matvec_round, x=x, **kwargs))

    def matvec_round_ex(self, x, opts=None, **kwargs):
        """matvec_round with diagnostics. Returns (TensorTrain, RoundResult)."""
        if isinstance(x, TensorTrain):
            x = x._c
        result, info = self._matrix_round_impl(
            opts, _matvec_round_ex, x=x, **kwargs)
        return TensorTrain(result), RoundResult(info)

    def matmat_round(self, other, opts=None, **kwargs):
        """Fused matmat + round: A * B followed by compression."""
        if isinstance(other, TensorTrainMatrix):
            other = other._c
        return TensorTrainMatrix(
            self._matrix_round_impl(
                opts, _matmat_round, other=other, **kwargs))

    def matmat_round_ex(self, other, opts=None, **kwargs):
        """matmat_round with diagnostics. Returns (TensorTrainMatrix, RoundResult)."""
        if isinstance(other, TensorTrainMatrix):
            other = other._c
        result, info = self._matrix_round_impl(
            opts, _matmat_round_ex, other=other, **kwargs)
        return TensorTrainMatrix(result), RoundResult(info)

    def _matrix_round_impl(self, opts, c_fn, x=None, other=None, **kwargs):
        """Shared opts dispatch for tt_matrix compression methods."""
        if isinstance(opts, RoundOptions):
            c_opts = opts._as_c()
            for k, v in kwargs.items():
                if k == "method":
                    v = _to_enum(_Cround_method, v)
                elif k == "gauge":
                    v = _to_enum(_Cround_gauge, v)
                setattr(c_opts, k, v)
            warm = None
            if hasattr(opts, "warm_start") and opts.warm_start is not None:
                warm = opts.warm_start._c
        elif isinstance(opts, dict):
            opts_copy = dict(opts)
            warm = opts_copy.pop("warm_start", None)
            if warm is not None and hasattr(warm, "_c"):
                warm = warm._c
            c_opts = RoundOptions(**opts_copy)._as_c()
        elif opts is None:
            warm_obj = kwargs.pop("warm_start", None)
            warm = warm_obj._c if warm_obj is not None and hasattr(
                warm_obj, "_c") else None
            c_opts = RoundOptions(**kwargs)._as_c()
        else:
            warm = None
            c_opts = RoundOptions(**kwargs)._as_c()
        args = [self._c, c_opts]
        if x is not None:
            args.insert(1, x)
        if other is not None:
            args.insert(1, other)
        if warm is not None:
            args.append(warm)
        return c_fn(*args)

    # -- static factories --
    @staticmethod
    def from_cores(cores):
        """Build a TensorTrainMatrix from a list of TTCoreMatrix objects or 4D
        numpy arrays.

        Parameters
        ----------
        cores : list[TTCoreMatrix | ndarray]
            One entry per mode.  Each core has shape (r_k, m_k, n_k, r_{k+1})
            with r_0 = r_d = 1.

        Returns
        -------
        TensorTrainMatrix
        """
        arrays = []
        for c in cores:
            if isinstance(c, TTCoreMatrix):
                arrays.append(c.data)
            elif isinstance(c, np.ndarray):
                arrays.append(np.asarray(c, dtype=np.float64))
            else:
                raise TypeError(
                    f"TensorTrainMatrix.from_cores: expected TTCoreMatrix "
                    f"or ndarray, got {type(c).__name__}")
        return TensorTrainMatrix(_from_matrix_cores(arrays))

    @staticmethod
    def zeros(row_shape, col_shape):
        """All-zero rank-1 TensorTrainMatrix with given row/column shapes."""
        return TensorTrainMatrix(_zeros_matrix(row_shape, col_shape))

    @staticmethod
    def identity(shape):
        """Identity operator as a rank-1 TensorTrainMatrix.
        
        Requires row_shape == col_shape == shape."""
        return TensorTrainMatrix(_identity_matrix(shape))

    @staticmethod
    def random(row_shape, col_shape, max_rank=0, seed=0):
        """Random TensorTrainMatrix with N(0,1) entries.
        
        max_rank=0 means unlimited (capped by feasible bound)."""
        return TensorTrainMatrix(
            _random_matrix(row_shape, col_shape, max_rank, int(seed)))

    @staticmethod
    def diag_from_tt(tt):
        """Construct a diagonal TT-matrix from a TT vector.
        
        Returns an operator A such that A * x = diag(tt) * x
        (Hadamard product of tt and x)."""
        if isinstance(tt, TensorTrain):
            tt = tt._c
        return TensorTrainMatrix(_diag_from_tt(tt))

    @staticmethod
    def from_samples(func, row_shape, col_shape, opts=None, **kwargs):
        """Build a TensorTrainMatrix from a black-box function via cross
        interpolation.

        Parameters
        ----------
        func : callable
            Python function ``f(row_idx, col_idx) -> float`` where
            both ``row_idx`` and ``col_idx`` are tuples of ``d`` ints.
        row_shape : list[int]
            Row mode sizes.
        col_shape : list[int]
            Column mode sizes.
        opts : FromSamplesOptions or dict, optional
            Cross-interpolation options.
        **kwargs
            Passed to FromSamplesOptions if opts is not given.

        Returns
        -------
        TensorTrainMatrix
        """
        if isinstance(opts, FromSamplesOptions):
            c_opts = opts._as_c()
            for k, v in kwargs.items():
                if k == "method":
                    v = _to_enum(_Cfrom_samples_method, v)
                setattr(c_opts, k, v)
            return TensorTrainMatrix(
                _from_samples_matrix(func, row_shape, col_shape, c_opts))
        if isinstance(opts, dict):
            merged = dict(opts)
            merged.update(kwargs)
            return TensorTrainMatrix(
                _from_samples_matrix(
                    func, row_shape, col_shape,
                    FromSamplesOptions(**merged)._as_c()))
        return TensorTrainMatrix(
            _from_samples_matrix(
                func, row_shape, col_shape,
                FromSamplesOptions(**kwargs)._as_c()))

    @staticmethod
    def from_dense(data, row_shape, col_shape, eps=None, opts=None, **kwargs):
        """Build a TensorTrainMatrix from a dense numpy array via TT-SVD compression."""
        if eps is not None and not kwargs and opts is None:
            return TensorTrainMatrix(
                _matrix_from_dense(data, row_shape, col_shape, float(eps)))
        if isinstance(opts, FromDenseOptions):
            c_opts = opts._as_c()
            for k, v in kwargs.items():
                if k == "method":
                    v = _to_enum(_Cfrom_dense_method, v)
                setattr(c_opts, k, v)
            if eps is not None:
                c_opts.eps = float(eps)
            return TensorTrainMatrix(
                _matrix_from_dense(data, row_shape, col_shape, c_opts))
        if isinstance(opts, dict):
            merged = dict(opts)
            merged.update(kwargs)
            if eps is not None:
                merged["eps"] = float(eps)
            c_opts = FromDenseOptions(**merged)._as_c()
            return TensorTrainMatrix(
                _matrix_from_dense(data, row_shape, col_shape, c_opts))
        c_opts = FromDenseOptions(**kwargs)._as_c()
        if eps is not None:
            c_opts.eps = float(eps)
        return TensorTrainMatrix(
            _matrix_from_dense(data, row_shape, col_shape, c_opts))

    def __repr__(self):
        return (
            f"TensorTrainMatrix(rows={self.row_shape}, " +
            f"cols={self.col_shape}, d={self.d})"
        )

# -------------------------------------------------------------------
# Module-level construction functions
# -------------------------------------------------------------------


def from_dense(data, shape, eps=None, opts=None, **kwargs):
    """
    Build a TensorTrain from a dense numpy array via TT-SVD compression
    (Oseledets 2011, Algorithm 1).

    The tensor-train representation of a d-dimensional tensor T with
    mode sizes n_0, ..., n_{d-1} is a factorization into d 3-tensors
    (cores) G_0, ..., G_{d-1} such that each element is recovered by
    contracting the cores along their shared bond indices::

        T(i_0, ..., i_{d-1}) = sum_{a_1 ... a_{d-1}}
            G_0(1, i_0, a_1) *
            G_1(a_1, i_1, a_2) * ... *
            G_{d-1}(a_{d-1}, i_{d-1}, 1)

    where core G_k has shape (r_k, n_k, r_{k+1}) with r_0 = r_d = 1.
    The bond dimensions r_k are the TT ranks; they control the
    representational power and storage cost.

    The algorithm performs a left-to-right sweep of truncated SVDs.  At
    each mode k it reshapes the working matrix C (initially the full
    dense tensor) into shape (r_k * n_k, rest), computes the SVD,
    truncates singular values below the per-step tolerance, reshapes the
    left singular vectors into core[k], and carries diag(s) * Vt forward
    as the new C.  The final mode becomes the last core unmodified.

    The per-step truncation tolerance is ::

        delta = eps * ||T||_F / sqrt(d - 1)

    which guarantees a global error bound ::

        ||T - T_TT||_F <= eps * ||T||_F

    Ranks are chosen adaptively by the SVD at each cut -- no manual
    rank selection is needed.  An optional ``max_rank`` (via ``opts``)
    provides a hard cap.

    Parameters
    ----------
    data : ndarray
      Flattened dense tensor in row-major order.
    shape : list[int]
      Mode sizes.
    eps : float, optional
      Truncation epsilon (shortcut for simple SVD compression).
    opts : FromDenseOptions or dict, optional
      Full options for construction method.
    **kwargs
      Additional option overrides, merged on top of opts if given.

    Returns
    -------
    TensorTrain
    """
    if isinstance(opts, FromDenseOptions):
        c_opts = opts._as_c()
        for k, v in kwargs.items():
            if k == "method":
                v = _to_enum(_Cfrom_dense_method, v)
            setattr(c_opts, k, v)
        if eps is not None:
            c_opts.eps = float(eps)
        return TensorTrain(_from_dense(data, shape, c_opts))
    if isinstance(opts, dict):
        merged = dict(opts)
        merged.update(kwargs)
        if eps is not None:
            merged["eps"] = float(eps)
        return TensorTrain(_from_dense(
            data, shape, FromDenseOptions(**merged)._as_c()))
    if kwargs or eps is not None:
        if eps is not None:
            kwargs["eps"] = float(eps)
        return TensorTrain(_from_dense(
            data, shape, FromDenseOptions(**kwargs)._as_c()))
    return TensorTrain(_from_dense(data, shape, FromDenseOptions()._as_c()))


def from_samples(func, shape, opts=None, **kwargs):
    """
    Build a TensorTrain from a black-box function via cross interpolation.

    Evaluates ``func`` at adaptively-chosen multi-indices without ever
    building the full dense tensor.

    Parameters
    ----------
    func : callable
      Python function ``f(idx) -> float`` where ``idx`` is a tuple
      of ``d`` integers (one per mode).
    shape : list[int]
      Mode sizes.  Pass an empty list ``[]`` for QTT methods (the
      shape is then derived from ``qtt_maxbits``).
    opts : FromSamplesOptions or dict, optional
      Cross-interpolation options.
    **kwargs
      Passed to FromSamplesOptions if opts is not given.

    Returns
    -------
    TensorTrain
    """
    if isinstance(opts, FromSamplesOptions):
        c_opts = opts._as_c()
        for k, v in kwargs.items():
            if k == "method":
                v = _to_enum(_Cfrom_samples_method, v)
            setattr(c_opts, k, v)
        return TensorTrain(_from_samples(func, shape, c_opts))
    if isinstance(opts, dict):
        merged = dict(opts)
        merged.update(kwargs)
        return TensorTrain(
            _from_samples(func, shape, FromSamplesOptions(**merged)._as_c())
        )
    return TensorTrain(_from_samples(func, shape, FromSamplesOptions(**kwargs)._as_c()))


# -------------------------------------------------------------------
# Module-level factory functions
# -------------------------------------------------------------------


def zeros(shape):
    """All-zero rank-1 TensorTrain with given mode sizes."""
    return TensorTrain(_zeros(shape))


def ones(shape):
    """All-one rank-1 TensorTrain (outer product of ones vectors)."""
    return TensorTrain(_ones(shape))


def canonical_unit(shape, idx):
    """Canonical basis TensorTrain: T[i] = 1 if i == idx else 0."""
    return TensorTrain(_canonical_unit(shape, list(idx)))


def random(shape, max_rank=0, seed=0):
    """Random TensorTrain with N(0,1) entries.

    max_rank=0 means unlimited (capped by feasible bound)."""
    return TensorTrain(_random(shape, max_rank, int(seed)))


def neg(a):
    """Negate a TensorTrain (element-wise sign flip)."""
    if isinstance(a, TensorTrain):
        return -a
    raise TypeError(
        f"Expected TensorTrain, got {type(a).__name__}")


def neg_matrix(a):
    """Negate a TensorTrainMatrix (element-wise sign flip)."""
    if isinstance(a, TensorTrainMatrix):
        return -a
    raise TypeError(
        f"Expected TensorTrainMatrix, got {type(a).__name__}")
#
# :D
#
