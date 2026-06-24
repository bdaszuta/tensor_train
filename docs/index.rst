tensor_train
============

Header-only C++20 tensor-train (TT) numerical library.
Namespace: ``mva::tensor_train``.
Source: `github.com/bdaszuta/tensor_train <https://github.com/bdaszuta/tensor_train>`_.

.. toctree::
   :maxdepth: 1
   :caption: C++ API Reference

   api/types
   api/public
   api/detail

.. toctree::
   :maxdepth: 1
   :caption: Python Bindings

   api/python

.. toctree::
   :maxdepth: 1
   :caption: Usage

   usage

What is a tensor train?
-----------------------

A tensor train (TT) decomposes a :math:`d`-dimensional tensor :math:`T` into a chain
of :math:`d` 3-index cores. Each core :math:`G_k` has shape
:math:`(r_k, n_k, r_{k+1})` with boundary ranks
:math:`r_0 = r_d = 1`. Storage falls from :math:`\mathcal{O}(n^d)`
to :math:`\mathcal{O}(d n r^2)` where :math:`r` is the
maximal bond rank.

A TT-matrix (MPO) generalises this to linear operators with per-core shape
:math:`(r_k, m_k, n_k, r_{k+1})`.

Quick start
-----------

.. code-block:: cpp

   #include "tensor_train.hpp"
   using namespace mva::tensor_train;

   std::vector<int> shape = {2, 3, 4};
   tt a = svd(dense.data(), shape, 1e-10);
   tt b = random(shape, 4, 42);
   tt c = add(a, b);
   c = round(c, round_options{.eps = 1e-8});

Build
-----

.. code-block:: bash

   make test
   make docs

Requires g++ (C++20), Eigen vendored at ``libs/ext_eigen``.
