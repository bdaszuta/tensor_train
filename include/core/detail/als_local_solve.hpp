/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: ALS local update for mixed-canonical TT rounding
*/
#pragma once

#include <Eigen/Core>

#include "../types/tt_core.hpp"
#include "../types/tt_eigen_bridge.hpp"
#include "core_view.hpp"

namespace mva
{
namespace tensor_train
{
namespace detail
{

// Returns a fresh tt_core of shape (L.rows, n_phys(Y_k), R.cols).
inline tt_core als_local_update(const row_matrix& L,
                                const tt_core& Y_k,
                                const row_matrix& R)
{
  const int rXl = static_cast<int>(L.rows());
  const int rYl = Y_k.r_left();
  const int n   = Y_k.n_phys();
  const int rYr = Y_k.r_right();
  const int rXr = static_cast<int>(R.cols());
  (void)rYl;  // checked implicitly by Eigen GEMM dims

  // T1 (rXl, n*rYr) = L (rXl, rYl) @ right_unfold(Y_k) (rYl, n*rYr).
  const Eigen::Index t1_cols = static_cast<Eigen::Index>(n) * rYr;
  row_matrix T1(rXl, t1_cols);
  T1.noalias() = L * right_unfold(Y_k);

  // Reinterpret T1's row-major buffer as (rXl*n, rYr) -- same bytes.
  const Eigen::Index ta_rows = static_cast<Eigen::Index>(rXl) * n;
  Eigen::Map<const row_matrix> T1_as(T1.data(), ta_rows, rYr);

  // X_k_lu (rXl*n, rXr) = T1_as @ R (rYr, rXr); = left_unfold(X_k).
  tt_core X_k(rXl, n, rXr);
  left_unfold(X_k).noalias() = T1_as * R;
  return X_k;
}

}  // namespace detail
}  // namespace tensor_train
}  // namespace mva
//
// :D
//
