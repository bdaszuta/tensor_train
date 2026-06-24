/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Frobenius inner product and norm for tt_matrix. Implemented by packing 4-axis cores to 3-axis and delegating to the TT inner product.
*/
#pragma once

#include <cmath>

#include "../detail/matrix_pack.hpp"
#include "../detail/shape_check.hpp"
#include "../types/tt.hpp"
#include "../types/tt_core.hpp"
#include "../types/tt_matrix.hpp"
#include "tt_inner.hpp"

namespace mva
{
namespace tensor_train
{
/**
 * @brief Frobenius inner product of two TT-matrices.
 * Packs each 4-axis core to 3-axis and contracts via the TT inner.
 * @param a First TT-matrix.
 * @param b Second TT-matrix.
 * @return \f$ \sum_{i,j} A[i,j] \cdot B[i,j] \f$.
 */
inline double frob_inner(const tt_matrix& a, const tt_matrix& b)
{
  detail::check_same_matrix_shape(a, b, "frob_inner");
  return inner(tt(detail::pack_matrix_cores(a)),
               tt(detail::pack_matrix_cores(b)));
}

/**
 * @brief Frobenius norm of a TT-matrix.
 * \f$ \sqrt{\langle A, A \rangle_F} \f$.
 * @param a Input TT-matrix.
 * @return \f$ \sqrt{\sum A[i,j]^2} \f$.
 */
inline double frob_norm(const tt_matrix& a)
{
  const double s = frob_inner(a, a);
  return std::sqrt(std::max(s, 0.0));
}

}  // namespace tensor_train
}  // namespace mva
//
// :D
//
