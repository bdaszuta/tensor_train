/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Dense-buffer to TT construction API
*/
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../types/tt.hpp"
#include "../types/tt_matrix.hpp"
#include "tt_matrix_factory.hpp"
#include "tt_svd.hpp"

namespace mva
{
namespace tensor_train
{

enum class from_dense_method
{
  svd,  // standard TT-SVD compression
  qtt   // quantized tensor train (validates mode sizes are powers of base)
};

struct from_dense_options
{
  double eps               = 1.0e-10;
  int max_rank             = 0;
  from_dense_method method = from_dense_method::svd;
  int qtt_base             = 2;
  bool validate            = true;
};

namespace detail
{

inline bool validate_qtt_shape(const std::vector<int>& shape,
                               int base,
                               const char* caller)
{
  if (base <= 1)
  {
    std::fprintf(stderr, "%s: qtt_base %d must be > 1\n", caller, base);
    std::abort();
  }
  // QTT digit extraction uses bitmask (base-1); only works for
  // power-of-2 bases.
  if ((base & (base - 1)) != 0)
  {
    std::fprintf(stderr,
                 "%s: qtt_base %d must be a power of 2\n",
                 caller, base);
    std::abort();
  }
  for (std::size_t k = 0; k < shape.size(); ++k)
  {
    int n = shape[k];
    if (n <= 0)
    {
      std::fprintf(
        stderr, "%s: mode %zu has non-positive size %d\n", caller, k, n);
      std::abort();
    }
    while (n > 1)
    {
      if (n % base != 0)
      {
        std::fprintf(stderr,
                     "%s: mode %zu size %d is not a power of %d\n",
                     caller,
                     k,
                     shape[k],
                     base);
        std::abort();
      }
      n /= base;
    }
  }
  return true;
}

/**
 * QTT compression: reshape dense data from original shape to QTT binary
 * shape, then apply standard TT-SVD.
 *
 * For each original mode n_k = base^{b_k}, split into b_k binary modes
 * of size base.  The dense data is reordered accordingly (the QTT
 * multi-index is iterated lexicographically; for each position, the
 * original flat index is reconstructed via bit-extraction).
 *
 * Cost: O(total * d_orig) for the reshape + TT-SVD cost.  For large
 * grids, prefer from_samples with QTT cross-approximation instead.
 *
 * @param data  Dense data buffer (row-major over original shape).
 * @param shape Original mode-size shape before QTT splitting.
 * @param base  Quantization base (typically 2).
 * @param eps   SVD truncation epsilon.
 * @param max_rank  Max bond rank (0 = no cap).
 * @return QTT-format compressed TensorTrain.
 */
inline tt qtt_svd(const double* data,
                  const std::vector<int>& shape,
                  int base,
                  double eps,
                  int max_rank)
{
  const int d_orig = static_cast<int>(shape.size());

  // 1. Compute bits per dimension and build QTT shape.
  std::vector<int> bits(static_cast<std::size_t>(d_orig));
  std::vector<int> qtt_shape;
  int total_bits = 0;
  for (int k = 0; k < d_orig; ++k)
  {
    int n = shape[static_cast<std::size_t>(k)];
    int b = 0;
    while (n > 1)
    {
      n /= base;
      ++b;
    }
    bits[static_cast<std::size_t>(k)] = b;
    total_bits += b;
    for (int i = 0; i < b; ++i)
      qtt_shape.push_back(base);
  }

  // Edge case: all mode sizes are 1 -> empty QTT shape.
  const int d_qtt = static_cast<int>(qtt_shape.size());
  if (d_qtt == 0)
  {
    // All modes have size 1: tensor is a single scalar.
    // Return a d=1 TT with one mode-size-1 core.
    tt_core c(1, 1, 1);
    std::memcpy(c.data(), data, sizeof(double));
    std::vector<tt_core> cores;
    cores.push_back(std::move(c));
    return tt(std::move(cores));
  }

  // 2. Compute original row-major strides.
  std::vector<int64_t> orig_stride(static_cast<std::size_t>(d_orig), 1);
  for (int k = d_orig - 2; k >= 0; --k)
    orig_stride[static_cast<std::size_t>(k)] =
      orig_stride[static_cast<std::size_t>(k + 1)] *
      static_cast<int64_t>(shape[static_cast<std::size_t>(k + 1)]);

  // 3. Compute total QTT elements and allocate.
  int64_t total = 1;
  for (int s : qtt_shape)
    total *= static_cast<int64_t>(s);
  std::vector<double> qtt_data(static_cast<std::size_t>(total));

  // 4. Reshape: iterate over QTT multi-indices, reconstruct original
  //    flat index, copy value.
  std::vector<int> qtt_idx(static_cast<std::size_t>(d_qtt), 0);
  for (int64_t pos = 0; pos < total; ++pos)
  {
    // Reconstruct original multi-index from QTT binary bits.
    int bit_pos = 0;
    int64_t orig_pos = 0;
    for (int dim = 0; dim < d_orig; ++dim)
    {
      int64_t val = 0;
      const int nb = bits[static_cast<std::size_t>(dim)];
      for (int b = 0; b < nb; ++b, ++bit_pos)
      {
        val = val * base +
              qtt_idx[static_cast<std::size_t>(bit_pos)];
      }
      orig_pos += val * orig_stride[static_cast<std::size_t>(dim)];
    }
    qtt_data[static_cast<std::size_t>(pos)] =
      data[static_cast<std::size_t>(orig_pos)];

    // Increment QTT multi-index lexicographically (fastest-varying last).
    for (int i = d_qtt - 1; i >= 0; --i)
    {
      ++qtt_idx[static_cast<std::size_t>(i)];
      if (qtt_idx[static_cast<std::size_t>(i)] < base)
        break;
      qtt_idx[static_cast<std::size_t>(i)] = 0;
    }
  }

  // 5. Standard TT-SVD on QTT-shaped data.
  return svd(qtt_data.data(), qtt_shape, eps, max_rank);
}

}  // namespace detail

/**
 * @brief Compress a dense tensor to TT form.
 *
 * Convenience overloads accepting double eps or from_dense_options.
 * The options struct supports svd and qtt methods; qtt uses a
 * quantised TT layout with base-2 mode splitting.
 *
 * @param data    Flat row-major dense buffer.
 * @param shape   Mode sizes n_0, ..., n_{d-1}.
 * @param opts    from_dense_options (method, eps, max_rank, qtt_base).
 * @return TT approximant.
 * @note  In the QTT path, the validate flag checks that qtt_base is a
 *        power of 2 and mode sizes are compatible.
 */

inline tt from_dense(const double* data,
                     const std::vector<int>& shape,
                     const from_dense_options& opts)
{
  if (opts.method == from_dense_method::qtt)
  {
    if (opts.validate)
      detail::validate_qtt_shape(shape, opts.qtt_base, "from_dense(qtt)");
    return detail::qtt_svd(
      data, shape, opts.qtt_base, opts.eps, opts.max_rank);
  }
  return detail::svd(data, shape, opts.eps, opts.max_rank);
}

inline tt from_dense(const double* data,
                     const std::vector<int>& shape,
                     double eps)
{
  from_dense_options opts;
  opts.eps = eps;
  return from_dense(data, shape, opts);
}

// -------- tt_matrix (operator) --------

/**
 * @brief Compress a dense matrix to TT-matrix form via TT-SVD.
 *
 * Supports the SVD method only (QTT is not yet implemented for
 * tt_matrix).
 *
 * @param data      Flat row-major dense buffer.
 * @param row_shape Row mode sizes m_0, ..., m_{d-1}.
 * @param col_shape Column mode sizes n_0, ..., n_{d-1}.
 * @param opts      from_dense_options (method, eps, max_rank).
 * @return TT-matrix approximant.
 * @note  The QTT method triggers abort() -- use method='svd'.
 */
inline tt_matrix from_dense(const double* data,
                            const std::vector<int>& row_shape,
                            const std::vector<int>& col_shape,
                            const from_dense_options& opts)
{
  if (opts.method == from_dense_method::qtt)
  {
    std::fprintf(stderr,
                 "from_dense(tt_matrix, qtt): not yet implemented.  "
                 "Aborting.\n");
    std::abort();
  }
  return detail::matrix_from_dense(
    data, row_shape, col_shape, opts.eps, opts.max_rank);
}

/** @brief Convenience overload taking only eps; uses default SVD method. */
inline tt_matrix from_dense(const double* data,
                            const std::vector<int>& row_shape,
                            const std::vector<int>& col_shape,
                            double eps)
{
  from_dense_options opts;
  opts.eps = eps;
  return from_dense(data, row_shape, col_shape, opts);
}

}  // namespace tensor_train
}  // namespace mva
//
// :D
//
