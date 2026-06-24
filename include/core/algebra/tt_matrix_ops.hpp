/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Affine combinations of tt_matrix instances: scale, neg, add, sub, axpy, axpby
*/
#pragma once

#include <utility>
#include <vector>

#include "../detail/matrix_block_concat.hpp"
#include "../detail/shape_check.hpp"
#include "../types/tt_matrix.hpp"

namespace mva
{
namespace tensor_train
{
/**
 * @brief Multiply a TT-matrix by a scalar.  Ranks unchanged.
 * @param a     Input TT-matrix.
 * @param alpha Scalar multiplier.
 * @return alpha * a.
 */

inline tt_matrix scale(const tt_matrix& a, double alpha)
{
  std::vector<tt_matrix_core> cores(a.cores().begin(), a.cores().end());
  if (!cores.empty())
  {
    double* p            = cores[0].data();
    const std::size_t sz = cores[0].size();
    for (std::size_t i = 0; i < sz; ++i)
      p[i] *= alpha;
  }
  return tt_matrix(std::move(cores));
}
/**
 * @brief Negate a TT-matrix.  Equivalent to scale(a, -1.0).
 * @param a Input TT-matrix.
 * @return -a.
 */

inline tt_matrix neg(const tt_matrix& a)
{
  return scale(a, -1.0);
}
/**
 * @brief alpha * A + beta * B for TT-matrices.
 * Ranks add on every interior bond.
 * @param alpha Scalar for a.
 * @param a     First TT-matrix.
 * @param beta  Scalar for b.
 * @param b     Second TT-matrix.
 * @return alpha*A + beta*B.
 * @note  Round afterwards to control rank growth.
 */

inline tt_matrix axpby(double alpha,
                       const tt_matrix& a,
                       double beta,
                       const tt_matrix& b)
{
  detail::check_same_matrix_shape(a, b, "axpby");
  const int d = a.d();
  std::vector<tt_matrix_core> out;
  out.reserve(static_cast<std::size_t>(d));

  if (d == 0)
    return tt_matrix(std::move(out));

  if (d == 1)
  {
    const tt_matrix_core& ca = a.core(0);
    const tt_matrix_core& cb = b.core(0);
    const int m              = ca.m_phys();
    const int n              = ca.n_phys();
    tt_matrix_core c(1, m, n, 1);
    double* dst      = c.data();
    const double* pa = ca.data();
    const double* pb = cb.data();
    const Eigen::Index sz = static_cast<Eigen::Index>(m) * n;
    for (Eigen::Index i = 0; i < sz; ++i)
      dst[i] = alpha * pa[i] + beta * pb[i];
    out.push_back(std::move(c));
    return tt_matrix(std::move(out));
  }

  for (int k = 0; k < d; ++k)
  {
    const double a_k = (k == 0) ? alpha : 1.0;
    const double b_k = (k == 0) ? beta : 1.0;
    out.push_back(
      detail::axpby_matrix_core(k, d, a.core(k), b.core(k), a_k, b_k));
  }
  return tt_matrix(std::move(out));
}
/**
 * @brief Add two TT-matrices.  Ranks add; round afterwards.
 * @param a First TT-matrix.
 * @param b Second TT-matrix.
 * @return a + b.
 */

inline tt_matrix add(const tt_matrix& a, const tt_matrix& b)
{
  return axpby(1.0, a, 1.0, b);
}
/**
 * @brief Subtract two TT-matrices.  Ranks add; round afterwards.
 * @param a First TT-matrix.
 * @param b Second TT-matrix.
 * @return a - b.
 */

inline tt_matrix sub(const tt_matrix& a, const tt_matrix& b)
{
  return axpby(1.0, a, -1.0, b);
}
/**
 * @brief Fused AXPY for TT-matrices: alpha * A + B.
 * @param alpha Scalar for a.
 * @param a     First TT-matrix.
 * @param b     Second TT-matrix.
 * @return alpha*A + B.  Ranks add; round afterwards.
 */

inline tt_matrix axpy(double alpha, const tt_matrix& a, const tt_matrix& b)
{
  return axpby(alpha, a, 1.0, b);
}

}  // namespace tensor_train
}  // namespace mva
//
// :D
//
