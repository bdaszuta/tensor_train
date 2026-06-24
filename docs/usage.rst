Usage Demo: 2D Function Compression
===================================

The script ``bind_py/usage/demo_2d_compress.py`` demonstrates
tensor-train compression of a non-separable 2D function on a
:math:`256 \times 256` grid.  It runs a sweep over three
truncation tolerances :math:`\varepsilon \in \{10^{-2}, 10^{-4}, 10^{-6}\}`,
compresses the dense :math:`256 \times 256` buffer with
:py:func:`from_dense`, and produces a side-by-side plot per
:math:`\varepsilon`:

- **Left:** Original function :math:`f(x,y)`.
- **Middle:** TT cores 0 and 1, reshaped to :math:`(N_x, r)` and
  :math:`(r, N_y)`, showing how physical nodes (support nodes) on
  each axis couple to the bond indices :math:`a_1` that connect
  the two cores.
- **Right:** Pointwise absolute reconstruction error
  :math:`|f - \tilde{f}|` on a log colour scale.

The suptitle shows the compression ratio
:math:`N_x N_y \,/\, (N_x r + r N_y)`.

To run (requires ``numpy``, ``matplotlib``, and the built bindings):

.. code-block:: bash

   python3 bind_py/usage/demo_2d_compress.py

Output is saved to ``figs/``.

Example at :math:`\varepsilon = 10^{-6}`
-----------------------------------------

.. figure:: ../figs/demo_2d_compress_eps_1e-06.png
   :alt: TT compression of a 2D function at eps=1e-6
   :align: center

   Diagonal ridge + two isotropic Gaussians, compressed at
   :math:`\varepsilon = 10^{-6}` to TT-rank 23 (5.6x compression,
   relative :math:`\ell_2` error :math:`7.5 \times 10^{-7}`).

Full sweep results
------------------

.. list-table::
   :header-rows: 1

   * - :math:`\varepsilon`
     - Rank :math:`r`
     - Parameters
     - Compression ratio
     - Rel. :math:`\ell_2` error
   * - :math:`10^{-2}`
     - 12
     - 6,144
     - 10.7x
     - :math:`8.6 \times 10^{-3}`
   * - :math:`10^{-4}`
     - 18
     - 9,216
     - 7.1x
     - :math:`8.9 \times 10^{-5}`
   * - :math:`10^{-6}`
     - 23
     - 11,776
     - 5.6x
     - :math:`7.5 \times 10^{-7}`

The function is

.. math::

   f(x,y) = \exp\!\left(-\frac{(x-y)^2}{2 \cdot 0.08^2}\right)
          + 0.3\,\exp\!\left(-\frac{(x-0.25)^2 + (y-0.75)^2}{0.02}\right)
          + 0.2\,\exp\!\left(-\frac{(x-0.7)^2 + (y-0.3)^2}{0.03}\right).

The diagonal ridge ensures the function is non-separable, so TT
ranks grow with the required accuracy rather than being bounded by a
low analytic rank.
