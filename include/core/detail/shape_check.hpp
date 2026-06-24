/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Shared shape-validation helpers and small numeric utilities
*/
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../types/tt.hpp"
#include "../types/tt_matrix.hpp"

namespace mva
{
namespace tensor_train
{
namespace detail
{

// Verify that two TTs have identical mode-size sequences.  Aborts on
// mismatch.  ``op`` is included in the diagnostic message.
inline void check_same_shape(const tt& a, const tt& b, const char* op)
{
  if (a.d() != b.d())
  {
    std::fprintf(
      stderr, "tt::%s: rank-d mismatch (a.d=%d, b.d=%d)\n", op, a.d(), b.d());
    std::abort();
  }
  const auto sa = a.shape();
  const auto sb = b.shape();
  for (int k = 0; k < a.d(); ++k)
  {
    if (sa[k] != sb[k])
    {
      std::fprintf(stderr,
                   "tt::%s: mode-size mismatch at axis %d "
                   "(a=%d, b=%d)\n",
                   op,
                   k,
                   sa[k],
                   sb[k]);
      std::abort();
    }
  }
}

// Verify that two tt_matrix objects have identical (m, n) mode sequence
// per site.  Aborts on mismatch.
inline void check_same_matrix_shape(const tt_matrix& a,
                                    const tt_matrix& b,
                                    const char* op)
{
  if (a.d() != b.d())
  {
    std::fprintf(stderr,
                 "tt_matrix::%s: rank-d mismatch (a.d=%d, b.d=%d)\n",
                 op,
                 a.d(),
                 b.d());
    std::abort();
  }
  for (int k = 0; k < a.d(); ++k)
  {
    if (a.core(k).m_phys() != b.core(k).m_phys() ||
        a.core(k).n_phys() != b.core(k).n_phys())
    {
      std::fprintf(stderr,
                   "tt_matrix::%s: mode mismatch at axis %d "
                   "(a=%dx%d, b=%dx%d)\n",
                   op,
                   k,
                   a.core(k).m_phys(),
                   a.core(k).n_phys(),
                   b.core(k).m_phys(),
                   b.core(k).n_phys());
      std::abort();
    }
  }
}

// Verify that a tt_matrix and a tt are compatible for matvec:
// a.col_shape == x.shape.
inline void check_matvec_compatible(const tt_matrix& a,
                                    const tt& x,
                                    const char* op)
{
  if (a.d() != x.d())
  {
    std::fprintf(stderr,
                 "%s: d mismatch (a.d=%d, x.d=%d)\n",
                 op,
                 a.d(),
                 x.d());
    std::abort();
  }
  for (int k = 0; k < a.d(); ++k)
  {
    if (a.core(k).n_phys() != x.core(k).n_phys())
    {
      std::fprintf(stderr,
                   "%s: mode-size mismatch at axis %d "
                   "(a.n=%d, x.n=%d)\n",
                   op,
                   k,
                   a.core(k).n_phys(),
                   x.core(k).n_phys());
      std::abort();
    }
  }
}

// Verify that two tt_matrix objects are compatible for matmat:
// a.n == b.m for each core (matrix multiply convention).
inline void check_matmat_compatible(const tt_matrix& a,
                                    const tt_matrix& b,
                                    const char* op)
{
  if (a.d() != b.d())
  {
    std::fprintf(stderr,
                 "%s: d mismatch (a.d=%d, b.d=%d)\n",
                 op,
                 a.d(),
                 b.d());
    std::abort();
  }
  for (int k = 0; k < a.d(); ++k)
  {
    if (a.core(k).n_phys() != b.core(k).m_phys())
    {
      std::fprintf(stderr,
                   "%s: inner-dimension mismatch at axis %d "
                   "(a.n=%d, b.m=%d)\n",
                   op,
                   k,
                   a.core(k).n_phys(),
                   b.core(k).m_phys());
      std::abort();
    }
  }
}

// Product of mode sizes (number of dense entries).  Returns 1 for an
// empty shape.
inline std::size_t mode_product(const std::vector<int>& shape)
{
  std::size_t p = 1;
  for (int n : shape)
  {
    p *= static_cast<std::size_t>(n);
  }
  return p;
}

// Frobenius norm of a flat buffer of length n.
inline double frob_norm_buffer(const double* p, std::size_t n)
{
  double s2 = 0.0;
  for (std::size_t i = 0; i < n; ++i)
  {
    s2 += p[i] * p[i];
  }
  return std::sqrt(s2);
}

// Rank-1 all-zero TT with per-site mode sizes from ``shape``.
// Used when a TT rounds to zero (e.g. norm == 0.0 after orthogonalisation).
inline tt make_rank1_zero_tt(const std::vector<int>& shape)
{
  const int d = static_cast<int>(shape.size());
  std::vector<tt_core> z;
  z.reserve(static_cast<std::size_t>(d));
  for (int k = 0; k < d; ++k)
  {
    tt_core c(1, shape[k], 1);
    c.zero_clear();
    z.push_back(std::move(c));
  }
  return tt(std::move(z));
}

}  // namespace detail
}  // namespace tensor_train
}  // namespace mva
//
// :D
//
