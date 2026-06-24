/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Fused Frobenius norm of A*x or A*B without materialising the full rank-multiplying apply result. Fuses the apply and inner product in one pass.
*/
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "../detail/contract.hpp"
#include "../detail/shape_check.hpp"
#include "../types/tt.hpp"
#include "../types/tt_core.hpp"
#include "../types/tt_matrix.hpp"
#include "../apply/tt_matrix_apply.hpp"

namespace mva
{
namespace tensor_train
{

/**
 * @brief Frobenius norm of A*x without materializing the full product.
 * Uses a fused contraction that avoids storing the intermediate
 * rank-multiplying apply result.
 * @param a TT-matrix operator.
 * @param x TT vector.
 * @return \f$ \|A \cdot x\|_F \f$.
 */
inline double frob_norm_apply(const tt_matrix& a, const tt& x)
{
  detail::check_matvec_compatible(a, x, "frob_norm_apply(matvec)");
  const int d = a.d();
  if (d == 0)
    return 0.0;
  detail::row_matrix E(1, 1);
  E(0, 0) = 1.0;
  for (int k = 0; k < d; ++k)
  {
    tt_core y_k = detail::matvec_core(a.core(k), x.core(k));
    E           = detail::contract_inner_step(E, y_k, y_k);
  }
  const double s = E(0, 0);
  return std::sqrt(std::max(s, 0.0));
}

// ||A * B||_F  --  matrix apply.
/**
 * @brief Frobenius norm of A*B without materializing the full product.
 * Packs the matmat_core result as a 3-axis TT core (the row-major
 * buffer is bit-identical) and contracts via the standard inner-product
 * step, mirroring the pack-as-tt trick used by frob_inner.
 * @param a Left TT-matrix operand.
 * @param b Right TT-matrix operand.
 * @return \f$ \|A \cdot B\|_F \f$.
 */
inline double frob_norm_apply(const tt_matrix& a, const tt_matrix& b)
{
  detail::check_matmat_compatible(a, b, "frob_norm_apply(matmat)");
  const int d = a.d();
  if (d == 0)
    return 0.0;
  detail::row_matrix E(1, 1);
  E(0, 0) = 1.0;
  for (int k = 0; k < d; ++k)
  {
    tt_matrix_core c_k = detail::matmat_core(a.core(k), b.core(k));
    const int64_t mn64 =
      static_cast<int64_t>(c_k.m_phys()) * c_k.n_phys();
    if (mn64 > std::numeric_limits<int>::max() || mn64 < 0)
    {
      std::fprintf(stderr,
                   "frob_norm_apply: m*n product exceeds INT_MAX "
                   "(m=%d * n=%d = %ld)\n",
                   c_k.m_phys(), c_k.n_phys(),
                   static_cast<long>(mn64));
      std::abort();
    }
    const int mn = static_cast<int>(mn64);  // guarded by overflow check L74-84
    tt_core y_k(c_k.r_left(), mn, c_k.r_right());
    std::memcpy(y_k.data(),
                c_k.data(),
                sizeof(double) * static_cast<std::size_t>(y_k.size()));
    E = detail::contract_inner_step(E, y_k, y_k);
  }
  const double s = E(0, 0);
  return std::sqrt(std::max(s, 0.0));
}

}  // namespace tensor_train
}  // namespace mva
//
// :D
//
