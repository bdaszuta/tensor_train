/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: TT orthogonalization sweeps (right_orthogonalize, left_orthogonalize)
*/
#pragma once

#include "../detail/orthogonalize.hpp"
#include "../types/tt.hpp"
#include "../types/tt_core.hpp"

namespace mva
{
namespace tensor_train
{

/**
 * @brief Right-orthogonalize a TT in-place (R->L QR sweep).
 * After this call, cores[1..d-1] satisfy sum_{i_k,a_{k+1}}
 * G_k(a_k,i_k,a_{k+1}) * G_k(a_k',i_k,a_{k+1}) = delta[a_k][a_k'].
 * @param a TT to orthogonalize (modified in-place).  No-op for d <= 1.
 */
inline void right_orthogonalize(tt& a)
{
  const int d = a.d();
  if (d <= 1)
    return;
  auto& cs = a.cores();
  for (int k = d - 1; k >= 1; --k)
  {
    detail::right_qr_step(cs[k], cs[k - 1]);
  }
}

namespace detail
{

/**
 * @brief Left-orthogonalize a TT in-place (L->R QR sweep).
 * After this call, cores[0..d-2] satisfy sum_{a_k,i_k}
 * G_k(a_k,i_k,a_{k+1}) * G_k(a_k,i_k,a_{k+1}') = delta[a_{k+1}][a_{k+1}'].
 * @param a TT to orthogonalize (modified in-place).  No-op for d <= 1.
 */
inline void left_orthogonalize(tt& a)
{
  const int d = a.d();
  if (d <= 1)
    return;
  auto& cs = a.cores();
  for (int k = 0; k < d - 1; ++k)
  {
    detail::left_qr_step(cs[k], cs[k + 1]);
  }
}

}  // namespace detail

}  // namespace tensor_train
}  // namespace mva
//
// :D
//
