/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Factory routines for tt_matrix. Mirrors core/tt_factory.hpp for the TT-matrix case (zeros, identity, diag_from_tt, random, matrix_from_dense)
*/
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <limits>

#include "../types/tt.hpp"
#include "../types/tt_core.hpp"
#include "../types/tt_matrix.hpp"
#include "tt_svd.hpp"

namespace mva
{
namespace tensor_train
{
/**
 * @brief All-zero rank-1 TT-matrix.
 * Each core is shape (1, m_k, n_k, 1) filled with zeros.
 * @param row_shape Row mode sizes m_0, ..., m_{d-1}.
 * @param col_shape Column mode sizes n_0, ..., n_{d-1}.
 * @return Zero TT-matrix.
 */

inline tt_matrix zeros(const std::vector<int>& row_shape,
                       const std::vector<int>& col_shape)
{
  const int d = static_cast<int>(row_shape.size());
  for (int k = 0; k < d; ++k)
  {
    if (row_shape[k] <= 0 || col_shape[k] <= 0)
    {
      std::fprintf(stderr,
                   "zeros_matrix: mode size (%d,%d) at k=%d "
                   "is not positive\n",
                   row_shape[k], col_shape[k], k);
      std::abort();
    }
  }
  std::vector<tt_matrix_core> cs;
  cs.reserve(static_cast<std::size_t>(d));
  for (int k = 0; k < d; ++k)
  {
    tt_matrix_core c(1, row_shape[k], col_shape[k], 1);
    c.zero_clear();
    cs.push_back(std::move(c));
  }
  return tt_matrix(std::move(cs));
}

/**
 * @brief Block-diagonal identity TT-matrix (rank 1).
 * Each core k is a (1, n_k, n_k, 1) tensor with delta[i][j].  The
 * full matrix is \f$ I_{n_0} \otimes I_{n_1} \otimes \cdots \otimes I_{n_{d-1}} \f$.
 * @param shape Mode sizes (square per mode).
 * @return Identity TT-matrix.
 */
inline tt_matrix identity(const std::vector<int>& shape)
{
  const int d = static_cast<int>(shape.size());
  for (int k = 0; k < d; ++k)
  {
    if (shape[k] <= 0)
    {
      std::fprintf(stderr,
                   "identity: mode size %d is not positive\n",
                   shape[k]);
      std::abort();
    }
  }
  std::vector<tt_matrix_core> cs;
  cs.reserve(static_cast<std::size_t>(d));
  for (int k = 0; k < d; ++k)
  {
    const int n = shape[k];
    tt_matrix_core c(1, n, n, 1);
    c.zero_clear();
    for (int i = 0; i < n; ++i)
      c(0, i, i, 0) = 1.0;
    cs.push_back(std::move(c));
  }
  return tt_matrix(std::move(cs));
}

/**
 * @brief Diagonal TT-matrix from a TT vector.
 * Each core (r, n, r') becomes (r, n, n, r') with the vector entry
 * placed on the diagonal.  Bond ranks are preserved from the input TT.
 * @param v TT vector to place on the diagonal.
 * @return Diagonal TT-matrix.
 */
inline tt_matrix diag_from_tt(const tt& v)
{
  const int d = v.d();
  std::vector<tt_matrix_core> cs;
  cs.reserve(static_cast<std::size_t>(d));
  for (int k = 0; k < d; ++k)
  {
    const tt_core& vc = v.core(k);
    const int rL      = vc.r_left();
    const int n       = vc.n_phys();
    const int rR      = vc.r_right();
    tt_matrix_core c(rL, n, n, rR);
    c.zero_clear();
    for (int a = 0; a < rL; ++a)
      for (int i = 0; i < n; ++i)
        for (int b = 0; b < rR; ++b)
          c(a, i, i, b) = vc(a, i, b);
    cs.push_back(std::move(c));
  }
  return tt_matrix(std::move(cs));
}

/**
 * @brief Random TT-matrix with N(0,1) entries.
 * @param row_shape Row mode sizes.
 * @param col_shape Column mode sizes.
 * @param max_rank  Cap on per-bond rank.
 * @param seed      RNG seed.
 * @return Random TT-matrix.
 */
inline tt_matrix random(const std::vector<int>& row_shape,
                        const std::vector<int>& col_shape,
                        int max_rank,
                        std::uint64_t seed)
{
  const int d = static_cast<int>(row_shape.size());
  std::vector<int> ranks(static_cast<std::size_t>(d + 1), 1);
  for (int k = 0; k < d; ++k)
  {
    long long cap =
      static_cast<long long>(ranks[k]) * row_shape[k] * col_shape[k];
    if (cap > max_rank && max_rank > 0)
      cap = max_rank;
    if (cap > std::numeric_limits<int>::max())
    {
      std::fprintf(stderr,
                   "random: rank cap %lld exceeds INT_MAX at bond %d; "
                   "provide a finite max_rank\n",
                   static_cast<long long>(cap), k + 1);
      std::abort();
    }
    ranks[k + 1] = static_cast<int>(cap);
  }
  for (int k = d - 1; k >= 0; --k)
  {
    long long cap =
      static_cast<long long>(ranks[k + 1]) * row_shape[k] * col_shape[k];
    if (cap > max_rank && max_rank > 0)
      cap = max_rank;
    if (cap > std::numeric_limits<int>::max())
    {
      std::fprintf(stderr,
                   "random: rank cap %lld exceeds INT_MAX at bond %d; "
                   "provide a finite max_rank\n",
                   static_cast<long long>(cap), k);
      std::abort();
    }
    if (cap < ranks[k])
      ranks[k] = static_cast<int>(cap);
  }
  ranks[0] = 1;
  ranks[d] = 1;

  std::mt19937_64 rng(seed);
  std::normal_distribution<double> nd(0.0, 1.0);

  std::vector<tt_matrix_core> cs;
  cs.reserve(static_cast<std::size_t>(d));
  for (int k = 0; k < d; ++k)
  {
    tt_matrix_core c(ranks[k], row_shape[k], col_shape[k], ranks[k + 1]);
    double* p         = c.data();
    const std::size_t sz = c.size();
    for (std::size_t i = 0; i < sz; ++i)
      p[i] = nd(rng);
    cs.push_back(std::move(c));
  }
  return tt_matrix(std::move(cs));
}

namespace detail
{

/**
 * @brief Compress a dense matrix to TT-matrix form via TT-SVD.
 * @param dense      Flat row-major dense buffer.
 * @param row_shape Row mode sizes.
 * @param col_shape Column mode sizes.
 * @param eps       Relative Frobenius tolerance.
 * @param max_rank  Hard cap on bond rank (0=unlimited).
 * @return TT-matrix approximant.
 */
inline tt_matrix matrix_from_dense(const double* dense,
                                   const std::vector<int>& row_shape,
                                   const std::vector<int>& col_shape,
                                   double eps   = 1.0e-10,
                                   int max_rank = 0)
{
  const int d = static_cast<int>(row_shape.size());
  if (d != static_cast<int>(col_shape.size()))
  {
    std::fprintf(stderr,
                 "matrix_from_dense: row_shape.size()=%d != "
                 "col_shape.size()=%d; aborting.\n",
                 d, static_cast<int>(col_shape.size()));
    std::abort();
  }

  std::vector<int> mn_shape(static_cast<std::size_t>(d));
  std::size_t M_total = 1, N_total = 1;
  for (int k = 0; k < d; ++k)
  {
    const int64_t prod =
      static_cast<int64_t>(row_shape[k]) * col_shape[k];
    if (prod > std::numeric_limits<int>::max())
    {
      std::fprintf(stderr,
                   "matrix_from_dense: mn product %ld exceeds INT_MAX "
                   "at axis %d; aborting.\n",
                   static_cast<long>(prod), k);
      std::abort();
    }
    mn_shape[k] = static_cast<int>(prod);
    M_total *= static_cast<std::size_t>(row_shape[k]);
    N_total *= static_cast<std::size_t>(col_shape[k]);
  }
  const std::size_t total = M_total * N_total;

  // Permute (I, J) -> per-axis k-major.
  std::vector<double> permuted(total);
  std::vector<std::size_t> mstride(d), nstride(d), kstride(d);
  {
    std::size_t mp = 1, np = 1, kp = 1;
    for (int k = d - 1; k >= 0; --k)
    {
      mstride[k] = mp;
      nstride[k] = np;
      kstride[k] = kp;
      mp *= static_cast<std::size_t>(row_shape[k]);
      np *= static_cast<std::size_t>(col_shape[k]);
      kp *= static_cast<std::size_t>(mn_shape[k]);
    }
  }
  std::vector<int> ii(d, 0), jj(d, 0);
  while (true)
  {
    std::size_t flat_in = 0;
    std::size_t flat_I  = 0;
    std::size_t flat_J  = 0;
    for (int k = 0; k < d; ++k)
    {
      const std::size_t ka =
        static_cast<std::size_t>(ii[k]) * col_shape[k] + jj[k];
      flat_in += ka * kstride[k];
      flat_I += static_cast<std::size_t>(ii[k]) * mstride[k];
      flat_J += static_cast<std::size_t>(jj[k]) * nstride[k];
    }
    permuted[flat_in] = dense[flat_I * N_total + flat_J];
    int axis          = d - 1;
    while (axis >= 0)
    {
      ++jj[axis];
      if (jj[axis] < col_shape[axis])
        break;
      jj[axis] = 0;
      ++ii[axis];
      if (ii[axis] < row_shape[axis])
        break;
      ii[axis] = 0;
      --axis;
    }
    if (axis < 0)
      break;
  }

  // Run core TT-SVD on the permuted buffer.
  tt t = detail::svd(permuted.data(), mn_shape, eps, max_rank);

  // Repackage each tt_core (a, k, b) into tt_matrix_core (a, i, j, b).
  // The storage is identical: contiguous row-major, k = i * n + j.
  std::vector<tt_matrix_core> mcs;
  mcs.reserve(static_cast<std::size_t>(d));
  for (int k = 0; k < d; ++k)
  {
    const tt_core& tc = t.core(k);
    const int rL      = tc.r_left();
    const int rR      = tc.r_right();
    tt_matrix_core mc(rL, row_shape[k], col_shape[k], rR);
    std::memcpy(mc.data(),
                tc.data(),
                sizeof(double) * static_cast<std::size_t>(tc.size()));
    mcs.push_back(std::move(mc));
  }
  return tt_matrix(std::move(mcs));
}

}  // namespace detail
}  // namespace tensor_train
}  // namespace mva
//
// :D
//
