/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Frobenius inner product <a, b> = sum_i a[i] * b[i] of two TTs via GEMM-accelerated network contraction
*/
#pragma once

#include <cmath>

#include "../detail/contract.hpp"
#include "../detail/core_view.hpp"
#include "../detail/shape_check.hpp"
#include "../types/tt.hpp"
#include "../types/tt_eigen_bridge.hpp"

namespace mva
{
namespace tensor_train
{

/**
 * @brief Frobenius inner product of two TTs.
 * Contracts all cores from left to right via GEMM-accelerated
 * tensor network contraction.
 * @param a First TT operand.
 * @param b Second TT operand.
 * @return \f$ \sum_{i_0\ldots i_{d-1}} a[i] \cdot b[i] \f$.
 * @note  Mode sizes must match.
 */
inline double inner(const tt& a, const tt& b)
{
  detail::check_same_shape(a, b, "inner");
  const int d = a.d();
  if (d == 0)
    return 0.0;
  // E starts as 1 x 1 identity; two GEMMs per core advance it.
  detail::row_matrix E(1, 1);
  E(0, 0) = 1.0;
  for (int k = 0; k < d; ++k)
  {
    E = detail::contract_inner_step(E, a.core(k), b.core(k));
  }
  // Final E is 1 x 1.
  return E(0, 0);
}

/**
 * @brief Frobenius norm of a TT.  \f$ \sqrt{\langle a, a \rangle} \f$.
 * @param a Input TT.
 * @return \f$ \sqrt{\sum a[i]^2} \f$.
 */
inline double norm(const tt& a)
{
  const double s = inner(a, a);
  return std::sqrt(std::max(s, 0.0));
}

}  // namespace tensor_train
}  // namespace mva
//
// :D
//
