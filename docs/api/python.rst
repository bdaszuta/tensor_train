Python Bindings
===============

Minimal nanobind bindings for the ``mva::tensor_train`` C++ library.
Module: ``tt_core``.

Build::

   conda activate <py311-env-with-nanobind>
   make bind
   make test-bindings

Import::

   import tt_core

All functions return new objects (the TT/TT-matrix is a value type in
C++, so every operation produces a copy).  Functions that produce
rank growth (``add``, ``sub``, ``axpy``, ``axpby``, ``hadamard``,
``matvec``, ``matmat``) must be followed by ``round`` or
``round_matrix`` to control bond ranks.

Enums
-----

.. py:class:: round_method

   Compression method for ``round``, ``matvec_round``, ``matmat_round``.

   .. py:attribute:: svd

      Oseledets Alg. 2: right-to-left QR sweep followed by left-to-right
      truncated SVD.

   .. py:attribute:: svd_streaming

      Fused matvec+round with no intermediate full-rank product.
      Valid for ``matvec_round`` only.

   .. py:attribute:: svd_naive

      Full apply then round.  Valid for ``matmat_round`` only.

   .. py:attribute:: als

      Alternating least squares: iteratively optimising one core at a
      time.  Warm-start friendly.

   .. py:attribute:: dmrg

      Two-site DMRG: updates core pairs simultaneously, adaptive-rank
      SVD at each bond.

.. py:class:: round_gauge

   Output gauge for ``round()`` (SVD method only).

   .. py:attribute:: none

      Norm-balanced output with no gauge guarantee.

   .. py:attribute:: right_canonical

      Output has cores[1..d-1] right-orthogonal.

.. py:class:: from_dense_method

   Compression method for ``from_dense()``.

   .. py:attribute:: svd

      Standard TT-SVD compression (Oseledets Alg. 1).

   .. py:attribute:: qtt

      Quantised TT: factors power-of-two modes for logarithmic-depth
      compression.

.. py:class:: from_samples_method

   Cross-interpolation method for ``from_samples()``.

   .. py:attribute:: dmrg_cross

      Two-site DMRG cross-approximation.  Typically best compression per
      evaluation budget.

   .. py:attribute:: tt_cross

      Classical Oseledets TT-cross with maxvol pivots.

   .. py:attribute:: als_cross

      ALS-based cross with one-core-at-a-time updates.

   .. py:attribute:: amen_cross

      Adaptive Minimum Enrichment (AMEn) cross.  More robust than ALS
      for complex functions.

   .. py:attribute:: qtt_dmrg_cross
   .. py:attribute:: qtt_tt_cross
   .. py:attribute:: qtt_als_cross
   .. py:attribute:: qtt_amen_cross

      QTT variants of the above methods.  Require ``qtt_base``,
      ``qtt_maxbits`` in ``from_samples_options``.

Options & Result Structs
-------------------------

.. py:class:: round_options

   Configuration for ``round()``, ``matvec_round()``, ``matmat_round()``.

   .. py:attribute:: eps: float = 0.0

      SVD truncation tolerance.  Singular values are dropped such that
      the relative Frobenius error is bounded by ``eps``.
      Set to 0.0 to disable truncation on the SVD path (DMRG requires
      ``eps > 0``).

   .. py:attribute:: max_rank: int = 0

      Hard cap on per-bond rank (0 = unlimited).

   .. py:attribute:: method: round_method = round_method.svd

      Compression method.

   .. py:attribute:: max_iters: int = 20

      Maximum ALS/DMRG iterations per sweep.

   .. py:attribute:: tol: float = 1e-12

      ALS/DMRG convergence tolerance.

   .. py:attribute:: seed: int = 0

      RNG seed for ALS/DMRG warm-start synthesis.

   .. py:attribute:: gauge: round_gauge = round_gauge.none

      Output gauge (SVD method only).

.. py:class:: from_dense_options

   Configuration for ``from_dense()``.

   .. py:attribute:: eps: float = 1e-10

      Relative Frobenius truncation tolerance.

   .. py:attribute:: max_rank: int = 0

      Hard cap on per-bond rank.

   .. py:attribute:: method: from_dense_method = from_dense_method.svd

      Compression method (svd or qtt).

   .. py:attribute:: qtt_base: int = 2

      Base for QTT mode splitting.

   .. py:attribute:: validate: bool = True

      If True, verifies the dense reconstruction against the input after
      compression.

.. py:class:: round_result

   Diagnostics returned by ``*_ex`` functions.

   .. py:attribute:: iters_run: int (read-only)

      Number of iterations completed.

   .. py:attribute:: final_resid: float (read-only)

      Final residual (ALS/DMRG) or 0.0 (SVD).

   .. py:attribute:: converged: bool (read-only)

      Whether the iterative method converged.  Always True for SVD.

.. py:class:: from_samples_options

   Configuration for ``from_samples()`` and ``from_samples_matrix()``.

   .. py:attribute:: eps: float

      Truncation tolerance.

   .. py:attribute:: max_rank: int

      Hard cap on per-bond rank.

   .. py:attribute:: rmin: int

      Minimum bond rank to retain.

   .. py:attribute:: method: from_samples_method

      Cross-interpolation method.

   .. py:attribute:: max_sweeps: int

      Maximum number of left-right sweep pairs.

   .. py:attribute:: init_rank: int

      Initial bond rank for random warm-start.

   .. py:attribute:: seed: int

      RNG seed.

   .. py:attribute:: maxvol_resolve_interval: int

      Re-solve maxvol pivots every N iterations (improves stability).

   .. py:attribute:: enrich_rank: int

      Number of extra rank candidates during enrichment.

   .. py:attribute:: use_qr_pivots: bool

      Use QR with column pivoting for initial pivot selection.

   .. py:attribute:: enrich_rounds: int

      Number of enrichment rounds per sweep.

   .. py:attribute:: enrich_samples: int

      Number of extra function evaluations per enrichment.

   .. py:attribute:: enrich_k: int

      Core index to enrich.

   .. py:attribute:: qtt_base: int

      QTT base for mode pooling.

   .. py:attribute:: qtt_maxbits: list[int]

      Maximum bits per QTT mode.

   .. py:attribute:: qtt_pool_modes: int = 3

      Number of modes to pool in QTT layout.

   .. py:attribute:: qtt_row_maxbits: list[int]

      Maximum bits per QTT row mode (tt_matrix only).

   .. py:attribute:: qtt_col_maxbits: list[int]

      Maximum bits per QTT column mode (tt_matrix only).

Core Classes
------------

.. py:class:: tt_core

   Single TT core with shape ``(r_left, n_phys, r_right)``.
   Row-major layout: element ``(a, n, b)`` is stored at
   ``(a * n_phys + n) * r_right + b``.

   .. py:method:: __init__(r_left: int, n_phys: int, r_right: int)

      Allocate a zero-filled core of shape ``(r_left, n_phys, r_right)``.

   .. py:attribute:: r_left: int (read-only)

      Left bond dimension.

   .. py:attribute:: n_phys: int (read-only)

      Physical mode size.

   .. py:attribute:: r_right: int (read-only)

      Right bond dimension.

   .. py:attribute:: dims: tuple[int, int, int] (read-only)

      Packed ``(r_left, n_phys, r_right)``.

   .. py:attribute:: size: int (read-only)

      Total number of stored ``double`` elements.

   .. py:method:: data_view() -> numpy.ndarray

      Read-only 3D numpy view of shape ``(r_left, n_phys, r_right)``
      into the core data.  Zero-copy: the view is valid only as long as
      this ``tt_core`` object exists.

   .. py:method:: fill(src: numpy.ndarray)

      Copy data from a 3D numpy array into this core.
      ``src`` must have shape ``(r_left, n_phys, r_right)``.

.. py:class:: tt_matrix_core

   Single TT-matrix core with shape ``(r_left, m_phys, n_phys, r_right)``.
   Row-major: element ``(a, i, j, b)`` is stored at
   ``((a * m_phys + i) * n_phys + j) * r_right + b``.

   .. py:method:: __init__(r_left: int, m_phys: int, n_phys: int, r_right: int)

      Allocate a zero-filled core.

   .. py:attribute:: r_left: int (read-only)

      Left bond dimension.

   .. py:attribute:: m_phys: int (read-only)

      Row mode size.

   .. py:attribute:: n_phys: int (read-only)

      Column mode size.

   .. py:attribute:: r_right: int (read-only)

      Right bond dimension.

   .. py:attribute:: dims: tuple[int, int, int, int] (read-only)

      Packed ``(r_left, m_phys, n_phys, r_right)``.

   .. py:attribute:: size: int (read-only)

      Total number of stored ``double`` elements.

   .. py:method:: data_view() -> numpy.ndarray

      Read-only 4D numpy view of shape ``(r_left, m_phys, n_phys, r_right)``.
      Zero-copy.

   .. py:method:: fill(src: numpy.ndarray)

      Copy data from a 4D numpy array into this core.
      ``src`` must have shape ``(r_left, m_phys, n_phys, r_right)``.

TT (Vector) Class
-----------------

.. py:class:: tt

   Tensor-train container: an ordered list of ``tt_core`` objects
   representing a d-dimensional tensor.  Boundary ranks ``r_0 = r_d = 1``
   are enforced at construction.

   .. py:attribute:: shape: list[int] (read-only)

      Mode sizes ``[n_0, ..., n_{d-1}]``.

   .. py:attribute:: ranks: list[int] (read-only)

      Bond ranks ``[r_0, r_1, ..., r_d]`` (length d+1).
      ``r_0 = r_d = 1``.

   .. py:attribute:: d: int (read-only)

      Number of modes (cores).  0 for a default-constructed TT.

   .. py:attribute:: max_rank: int (read-only)

      Maximum bond rank across all bonds.

   .. py:attribute:: num_params: int (read-only)

      Total number of stored double-precision values.
      Equal to ``sum_k r_k * n_k * r_{k+1}``.

   .. py:method:: to_dense() -> numpy.ndarray

      Reconstruct the full dense tensor as a 1D row-major numpy array
      of length ``prod(shape)``.  The multi-index ``(i_0, ..., i_{d-1})``
      maps to position ``(((i_0*n_1)+i_1)*n_2+...)+i_{d-1}``.

   .. py:method:: core(k: int) -> tt_core

      Return a **copy** of core k (0 <= k < d).

   .. py:method:: cores() -> list[numpy.ndarray]

      Return a list of read-only 3D numpy views, one per core.
      Zero-copy: the views share a single TT copy kept alive until
      all arrays are garbage-collected.

TT-matrix (MPO) Class
----------------------

.. py:class:: tt_matrix

   Matrix product operator: an ordered list of ``tt_matrix_core`` objects
   representing a linear operator of shape
   ``(prod(row_shape), prod(col_shape))``.

   .. py:attribute:: d: int (read-only)

      Number of modes.

   .. py:attribute:: row_shape: list[int] (read-only)

      Row mode sizes ``[m_0, ..., m_{d-1}]``.

   .. py:attribute:: col_shape: list[int] (read-only)

      Column mode sizes ``[n_0, ..., n_{d-1}]``.

   .. py:attribute:: ranks: list[int] (read-only)

      Bond ranks (length d+1, ``r_0 = r_d = 1``).

   .. py:attribute:: max_rank: int (read-only)

      Maximum bond rank.

   .. py:attribute:: total_rows: int (read-only)

      Total row extent ``= prod(row_shape)``.

   .. py:attribute:: total_cols: int (read-only)

      Total column extent ``= prod(col_shape)``.

   .. py:attribute:: num_params: int (read-only)

      Total stored doubles.

   .. py:method:: to_dense() -> numpy.ndarray

      Reconstruct the full dense matrix as a 1D row-major numpy array
      of length ``total_rows * total_cols``.

   .. py:method:: core(k: int) -> tt_matrix_core

      Return a **copy** of core k (0 <= k < d).

   .. py:method:: cores() -> list[numpy.ndarray]

      Return a list of read-only 4D numpy views, one per core.
      Zero-copy.

TT Construction
---------------

.. py:function:: from_dense(data: numpy.ndarray, shape: list[int], eps: float) -> tt
   :no-index:

   Compress a dense tensor to TT form via TT-SVD with tolerance ``eps``.
   ``data`` must be a 1D row-major array of length ``prod(shape)``.

.. py:function:: from_dense(data: numpy.ndarray, shape: list[int], opts: from_dense_options) -> tt
   :no-index:

   Compress a dense tensor using ``from_dense_options`` (supports SVD
   and QTT).

.. py:function:: zeros(shape: list[int]) -> tt

   All-zero rank-1 TT.  Every core is ``(1, n_k, 1)`` filled with zeros.

.. py:function:: ones(shape: list[int]) -> tt

   Constant-1 rank-1 TT (outer product of ones vectors).  Every core is
   ``(1, n_k, 1)`` with every entry equal to 1.

.. py:function:: canonical_unit(shape: list[int], idx: list[int]) -> tt

   Canonical basis TT with exactly one non-zero entry at multi-index
   ``idx``.  Each core k is ``(1, n_k, 1)`` with a 1 at ``idx[k]``.

.. py:function:: random(shape: list[int], max_rank: int, seed: int) -> tt

   Random TT with N(0,1) entries.  Bond ranks are set to the largest
   feasible values bounded by ``max_rank`` (forward+backward pass).

.. py:function:: from_cores(cores: list[numpy.ndarray]) -> tt

   Build a TT from a list of 3D C-contiguous numpy arrays.
   Core k must have shape ``(r_k, n_k, r_{k+1})`` with ``r_0 = r_d = 1``
   and must be C-contiguous (use ``np.ascontiguousarray`` if needed).

TT Algebra
----------

All operations return a new ``tt`` instance.  Ranks grow; follow with
``round()`` to control rank.

.. py:function:: add(a: tt, b: tt) -> tt

   ``a + b``.  Ranks add: ``r_k = r_a[k] + r_b[k]``.

.. py:function:: sub(a: tt, b: tt) -> tt

   ``a - b``.  Ranks add.

.. py:function:: scale(a: tt, alpha: float) -> tt

   ``alpha * a``.  Ranks unchanged (multiplies first core only).

.. py:function:: axpy(alpha: float, a: tt, b: tt) -> tt

   ``alpha * a + b``.  Ranks add.

.. py:function:: axpby(alpha: float, a: tt, beta: float, b: tt) -> tt

   ``alpha * a + beta * b``.  Ranks add.

.. py:function:: hadamard(a: tt, b: tt) -> tt

   Elementwise (Hadamard) product ``a .* b``.  Ranks **multiply**:
   ``r_k = r_a[k] * r_b[k]``.  Round aggressively after this call.

.. py:function:: neg(a: tt) -> tt

   ``-a``.  Ranks unchanged.  Equivalent to ``scale(a, -1.0)``.

TT Inner Products & Evaluation
------------------------------

.. py:function:: inner(a: tt, b: tt) -> float

   Frobenius inner product: ``sum a[i] * b[i]`` over all multi-indices.
   Mode sizes must match.

.. py:function:: norm(a: tt) -> float

   Frobenius norm: ``sqrt(inner(a, a))``.

.. py:function:: eval_at(a: tt, idx: list[int]) -> float

   Evaluate the TT at a single multi-index ``idx``.
   ``idx`` must have length ``a.d`` with ``0 <= idx[k] < a.shape[k]``.

.. py:function:: eval_batch(a: tt, idx: numpy.ndarray, M: int) -> numpy.ndarray

   Evaluate the TT at ``M`` multi-indices stored in a flat row-major
   buffer ``idx`` of shape ``(M, d)``.  Returns a 1D array of length ``M``.

TT Compression
--------------

.. py:function:: round(a: tt, opts: round_options) -> tt
   :no-index:

   Compress ``a`` using the method in ``opts``.  Supports SVD, ALS, DMRG.
   Returns a new compressed TT.

.. py:function:: round(a: tt, opts: round_options, warm: tt | None) -> tt
   :no-index:

   Compress with an optional warm-start TT.  Pass ``warm=None`` for cold
   start.

.. py:function:: round_ex(a: tt, opts: round_options) -> tuple[tt, round_result]
   :no-index:

   Compress and return ``(compressed_tt, diagnostics)``.  The
   ``round_result`` contains ``iters_run``, ``final_resid``, and
   ``converged``.

.. py:function:: round_ex(a: tt, opts: round_options, warm: tt | None) -> tuple[tt, round_result]
   :no-index:

   Compress with optional warm-start, returning diagnostics.

TT Cross-Approximation
----------------------

.. py:function:: from_samples(func: callable, shape: list[int], opts: from_samples_options = from_samples_options()) -> tt

   Build a TT by evaluating ``func`` at adaptively chosen multi-indices,
   without ever materialising the full dense tensor.
   ``func(tuple[int, ...])`` receives a tuple of d integers and must
   return a ``float``.
   Supports ``dmrg_cross``, ``tt_cross``, ``als_cross``, ``amen_cross``,
   and their QTT variants (set via ``opts.method``).

TT-matrix Construction
----------------------

.. py:function:: matrix_from_dense(data: numpy.ndarray, row_shape: list[int], col_shape: list[int], eps: float) -> tt_matrix
   :no-index:

   Compress a dense matrix to TT-matrix form via TT-SVD with tolerance
   ``eps``.  ``data`` must be a 1D row-major array of length
   ``prod(row_shape) * prod(col_shape)``.

.. py:function:: matrix_from_dense(data: numpy.ndarray, row_shape: list[int], col_shape: list[int], opts: from_dense_options) -> tt_matrix
   :no-index:

   Compress using ``from_dense_options``.  QTT method is not yet
   implemented for ``tt_matrix``.

.. py:function:: zeros_matrix(row_shape: list[int], col_shape: list[int]) -> tt_matrix

   All-zero rank-1 TT-matrix.  Each core is ``(1, m_k, n_k, 1)``.

.. py:function:: identity_matrix(shape: list[int]) -> tt_matrix

   Block-diagonal identity TT-matrix (rank 1).  Each core is
   ``(1, n_k, n_k, 1)`` with ``delta[i][j]``.

.. py:function:: random_matrix(row_shape: list[int], col_shape: list[int], max_rank: int, seed: int) -> tt_matrix

   Random TT-matrix with N(0,1) entries.

.. py:function:: diag_from_tt(diag: tt) -> tt_matrix

   Diagonal TT-matrix from a TT vector.  Each core k becomes
   ``(r_k, n_k, n_k, r_{k+1})`` with the vector entry on the diagonal.

.. py:function:: from_matrix_cores(cores: list[numpy.ndarray]) -> tt_matrix

   Build a TT-matrix from a list of 4D C-contiguous numpy arrays.
   Core k must have shape ``(r_k, m_k, n_k, r_{k+1})`` with boundary
   ranks 1.

.. py:function:: from_samples_matrix(func: callable, row_shape: list[int], col_shape: list[int], opts: from_samples_options = from_samples_options()) -> tt_matrix

   Cross-approximation for TT-matrices.  ``func(row_tuple, col_tuple)``
   receives two ``tuple[int, ...]`` and must return ``float``.
   QTT methods are not yet implemented for ``tt_matrix``.

TT-matrix Algebra
-----------------

All operations return a new ``tt_matrix`` instance.  Ranks grow;
follow with ``round_matrix()``.

.. py:function:: add_matrix(a: tt_matrix, b: tt_matrix) -> tt_matrix

   ``A + B``.  Ranks add.

.. py:function:: sub_matrix(a: tt_matrix, b: tt_matrix) -> tt_matrix

   ``A - B``.  Ranks add.

.. py:function:: scale_matrix(a: tt_matrix, alpha: float) -> tt_matrix

   ``alpha * A``.  Ranks unchanged.

.. py:function:: axpy_matrix(alpha: float, a: tt_matrix, b: tt_matrix) -> tt_matrix

   ``alpha * A + B``.  Ranks add.

.. py:function:: axpby_matrix(alpha: float, a: tt_matrix, beta: float, b: tt_matrix) -> tt_matrix

   ``alpha * A + beta * B``.  Ranks add.

.. py:function:: neg_matrix(a: tt_matrix) -> tt_matrix

   ``-A``.  Ranks unchanged.

TT-matrix Inner Products
-------------------------

.. py:function:: frob_inner(a: tt_matrix, b: tt_matrix) -> float

   Frobenius inner product: ``sum A[i,j] * B[i,j]`` over all entries.

.. py:function:: frob_norm_matrix(a: tt_matrix) -> float

   Frobenius norm: ``sqrt(frob_inner(a, a))``.

TT-matrix Apply (Raw)
---------------------

Raw apply multiplies ranks.  Results **must** be compressed with
``matvec_round`` or ``round`` before further use.  For inner loops,
prefer the fused ``matvec_round`` / ``matmat_round`` entry points.

.. py:function:: matvec(a: tt_matrix, x: tt) -> tt

   ``A * x``.  Result ranks are ``r_A[k] * r_x[k]``.

.. py:function:: matmat(a: tt_matrix, b: tt_matrix) -> tt_matrix

   ``A * B``.  Result ranks are ``r_A[k] * r_B[k]``.

TT-matrix Compression
---------------------

.. py:function:: round_matrix(a: tt_matrix, opts: round_options) -> tt_matrix
   :no-index:

   Compress a TT-matrix.  Internally packs 4-axis cores to 3-axis,
   applies TT ``round``, then unpacks.

.. py:function:: round_matrix(a: tt_matrix, opts: round_options, warm: tt_matrix | None) -> tt_matrix
   :no-index:

   Compress with optional warm-start TT-matrix.

.. py:function:: round_matrix_ex(a: tt_matrix, opts: round_options) -> tuple[tt_matrix, round_result]
   :no-index:

   Compress with diagnostics.

.. py:function:: round_matrix_ex(a: tt_matrix, opts: round_options, warm: tt_matrix | None) -> tuple[tt_matrix, round_result]
   :no-index:

   Compress with warm-start and diagnostics.

.. py:function:: matvec_round(a: tt_matrix, x: tt, opts: round_options) -> tt
   :no-index:

   Fused ``A * x`` + round.  No intermediate full-rank product
   materialised.  Prefer over raw ``matvec`` in all iterative codes.

.. py:function:: matvec_round(a: tt_matrix, x: tt, opts: round_options, warm: tt | None) -> tt
   :no-index:

   Fused matvec+round with optional warm-start.

.. py:function:: matvec_round_ex(a: tt_matrix, x: tt, opts: round_options) -> tuple[tt, round_result]
   :no-index:

   Fused matvec+round with diagnostics.

.. py:function:: matvec_round_ex(a: tt_matrix, x: tt, opts: round_options, warm: tt | None) -> tuple[tt, round_result]
   :no-index:

   Fused matvec+round with warm-start and diagnostics.

.. py:function:: matmat_round(a: tt_matrix, b: tt_matrix, opts: round_options) -> tt_matrix
   :no-index:

   Fused ``A * B`` + round.  No intermediate full-rank product
   materialised.  Prefer over raw ``matmat`` in all iterative codes.

.. py:function:: matmat_round(a: tt_matrix, b: tt_matrix, opts: round_options, warm: tt_matrix | None) -> tt_matrix
   :no-index:

   Fused matmat+round with optional warm-start.

.. py:function:: matmat_round_ex(a: tt_matrix, b: tt_matrix, opts: round_options) -> tuple[tt_matrix, round_result]
   :no-index:

   Fused matmat+round with diagnostics.

.. py:function:: matmat_round_ex(a: tt_matrix, b: tt_matrix, opts: round_options, warm: tt_matrix | None) -> tuple[tt_matrix, round_result]
   :no-index:

   Fused matmat+round with warm-start and diagnostics.

Gauge / Orthogonalization
-------------------------

All gauge functions operate on copies (the original is not modified,
unlike the C++ in-place versions).

.. py:function:: right_orthogonalize(a: tt) -> tt

   Return a copy with cores[1..d-1] right-orthogonal (each core's
   left-unfold has orthonormal columns).

.. py:function:: left_orthogonalize(a: tt) -> tt

   Return a copy with cores[0..d-2] left-orthogonal (each core's
   right-unfold has orthonormal rows).

.. py:function:: right_orthogonalize_matrix(a: tt_matrix) -> tt_matrix

   Return a copy with cores[1..d-1] right-orthogonal.

Other
-----

.. py:function:: soft_threshold(a: tt, tau: float) -> tt

   Singular-value soft-thresholding: replaces each singular value SV
   with ``max(SV - tau, 0)``.  This is the proximal operator for the
   nuclear norm.  When ``tau <= 0``, returns a deep copy (no-op).

.. py:function:: frob_norm_apply_matvec(a: tt_matrix, x: tt) -> float

   ``||A * x||_F`` computed in one fused pass without storing the
   intermediate rank-multiplying product.

.. py:function:: frob_norm_apply_matmat(a: tt_matrix, b: tt_matrix) -> float

   ``||A * B||_F`` computed in one fused pass without storing the
   intermediate rank-multiplying product.
