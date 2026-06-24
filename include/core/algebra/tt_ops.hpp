/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Basic TT-vs-TT operators (scale, neg, add, sub, axpy, axpby, hadamard). All return fresh tt instances; ranks add for linear ops, multiply for Hadamard.
*/
#pragma once

#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>
#include <limits>

#include "../detail/block_concat.hpp"
#include "../detail/shape_check.hpp"
#include "../types/tt.hpp"
#include "../types/tt_core.hpp"

namespace mva
{
namespace tensor_train
{

/**
 * @brief Multiply a TT by a scalar.
 * Multiplies the first core's buffer by alpha; all other cores
 * are deep-copied.  Ranks unchanged.
 * @param a     Input TT.
 * @param alpha Scalar multiplier.
 * @return alpha * a as a new TT.
 */
inline tt scale(const tt& a, double alpha)
{
  std::vector<tt_core> cores(a.cores().begin(), a.cores().end());
  if (!cores.empty())
  {
    double* p          = cores[0].data();
    const std::size_t sz = cores[0].size();
    for (std::size_t i = 0; i < sz; ++i)
    {
      p[i] *= alpha;
    }
  }
  return tt(std::move(cores));
}

/**
 * @brief Negate a TT.  Equivalent to scale(a, -1.0).
 * @param a Input TT.
 * @return -a as a new TT.
 */
inline tt neg(const tt& a)
{
  return scale(a, -1.0);
}

/**
 * @brief alpha * a + beta * b in TT format.
 * Ranks add on every interior bond: r_k = r_a[k] + r_b[k].
 * Boundary ranks r_0 = r_d = 1 are preserved.
 * @param alpha Scalar multiplier for a.
 * @param a     First TT operand.
 * @param beta  Scalar multiplier for b.
 * @param b     Second TT operand.
 * @return alpha*a + beta*b as a new TT.
 * @note  Mode sizes of a and b must match (enforced by check).
 *        Round after this call to control rank growth.
 */
inline tt axpby(double alpha, const tt& a, double beta, const tt& b)
{
  detail::check_same_shape(a, b, "axpby");
  const int d = a.d();
  std::vector<tt_core> out;
  out.reserve(d);

  if (d == 0)
  {
    return tt(std::move(out));
  }

  if (d == 1)
  {
    // Single-core case: elementwise alpha*a + beta*b on the (1,n,1) core.
    const tt_core& ca = a.core(0);
    const tt_core& cb = b.core(0);
    const int n       = ca.n_phys();
    tt_core c(1, n, 1);
    double* dst      = c.data();
    const double* pa = ca.data();
    const double* pb = cb.data();
    for (int i = 0; i < n; ++i)
    {
      dst[i] = alpha * pa[i] + beta * pb[i];
    }
    out.push_back(std::move(c));
    return tt(std::move(out));
  }

  // d >= 2.  Fold both scalars into the head core; later cores use 1.
  for (int k = 0; k < d; ++k)
  {
    const double a_k = (k == 0) ? alpha : 1.0;
    const double b_k = (k == 0) ? beta : 1.0;
    out.push_back(detail::axpby_core(k, d, a.core(k), b.core(k), a_k, b_k));
  }
  return tt(std::move(out));
}

/**
 * @brief Add two TTs.  Equivalent to axpby(1.0, a, 1.0, b).
 * @param a First TT operand.
 * @param b Second TT operand.
 * @return a + b as a new TT.
 * @note  Ranks add: r_k = r_a[k] + r_b[k].  Round afterwards.
 */
inline tt add(const tt& a, const tt& b)
{
  return axpby(1.0, a, 1.0, b);
}

/**
 * @brief Subtract two TTs.  Equivalent to axpby(1.0, a, -1.0, b).
 * @param a First TT operand.
 * @param b Second TT operand.
 * @return a - b as a new TT.
 * @note  Ranks add: r_k = r_a[k] + r_b[k].  Round afterwards.
 */
inline tt sub(const tt& a, const tt& b)
{
  return axpby(1.0, a, -1.0, b);
}

/**
 * @brief Fused AXPY: alpha * a + b.
 * Equivalent to axpby(alpha, a, 1.0, b).
 * @param alpha Scalar multiplier for a.
 * @param a     First TT operand.
 * @param b     Second TT operand.
 * @return alpha*a + b as a new TT.
 * @note  Ranks add.  Round afterwards.
 */
inline tt axpy(double alpha, const tt& a, const tt& b)
{
  return axpby(alpha, a, 1.0, b);
}

/**
 * @brief Elementwise (Hadamard) product a .* b.
 * Each result core k is the Kronecker product of core k from a
 * and b along the rank axes.  Ranks multiply:
 * r_k = r_a[k] * r_b[k].
 * @param a First TT operand.
 * @param b Second TT operand.
 * @return Elementwise product as a new TT.
 * @note  Rank growth is multiplicative; round aggressively after.
 */
inline tt hadamard(const tt& a, const tt& b)
{
  detail::check_same_shape(a, b, "hadamard");
  const int d = a.d();
  std::vector<tt_core> out;
  out.reserve(d);

  for (int k = 0; k < d; ++k)
  {
    const tt_core& ca = a.core(k);
    const tt_core& cb = b.core(k);
    const int n       = ca.n_phys();
    const int raL     = ca.r_left();
    const int raR     = ca.r_right();
    const int rbL     = cb.r_left();
    const int rbR     = cb.r_right();
    const int64_t rL64 = static_cast<int64_t>(raL) * rbL;
    const int64_t rR64 = static_cast<int64_t>(raR) * rbR;
    if (rL64 > std::numeric_limits<int>::max() ||
        rR64 > std::numeric_limits<int>::max())
    {
      std::fprintf(stderr,
                   "hadamard: rank product exceeds INT_MAX "
                   "(rL=%ld, rR=%ld); aborting.\n",
                   static_cast<long>(rL64),
                   static_cast<long>(rR64));
      std::abort();
    }
    const int rL = static_cast<int>(rL64);
    const int rR = static_cast<int>(rR64);
    tt_core c(rL, n, rR);
    double* dst      = c.data();
    const double* pa = ca.data();
    const double* pb = cb.data();
    // dst[(I*n + j)*rR + J] = ca(i_a, j, i_a') * cb(i_b, j, i_b'),
    // with I = i_a * rbL + i_b, J = i_a' * rbR + i_b'.
    for (int ia = 0; ia < raL; ++ia)
    {
      for (int ib = 0; ib < rbL; ++ib)
      {
        const int I = ia * rbL + ib;
        for (int j = 0; j < n; ++j)
        {
          for (int ip = 0; ip < raR; ++ip)
          {
            const double av = pa[(ia * n + j) * raR + ip];
            for (int iq = 0; iq < rbR; ++iq)
            {
              const int J               = ip * rbR + iq;
              const double bv           = pb[(ib * n + j) * rbR + iq];
              const std::size_t idx =
                (static_cast<std::size_t>(I) * n + j) * rR + J;
              dst[idx] = av * bv;
            }
          }
        }
      }
    }
    out.push_back(std::move(c));
  }

  return tt(std::move(out));
}

}  // namespace tensor_train
}  // namespace mva
//
// :D
//
