/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Two-site local-solve helpers for DMRG-style TT rounding
*/
#pragma once

#include <Eigen/Core>
#include <utility>

#include "../types/tt_core.hpp"
#include "../types/tt_eigen_bridge.hpp"
#include "core_view.hpp"
#include "truncate.hpp"

namespace mva
{
namespace tensor_train
{
namespace detail
{

// Build the merged-pair target Theta as a (rXl * n_k, n_{k+1} * rXr)
// row-major matrix.  Three logical contractions, two GEMMs:
//
//   T1 = L @ right_unfold(Y_k)
//          shape (rXl, n_k * rYm),  reshape to (rXl * n_k, rYm).
//   T2 = left_unfold(Y_{k+1}) @ R
//          shape (rYm * n_{k+1}, rXr),  reshape to (rYm, n_{k+1} * rXr).
//   Theta = T1_resh @ T2_resh
//          shape (rXl * n_k, n_{k+1} * rXr).
inline row_matrix merge_y_block(const row_matrix& L,
                                const tt_core& Y_k,
                                const tt_core& Y_kp1,
                                const row_matrix& R)
{
  const int rYl  = Y_k.r_left();
  const int n_k  = Y_k.n_phys();
  const int rYm  = Y_k.r_right();
  const int n_kp = Y_kp1.n_phys();
  const int rYr  = Y_kp1.r_right();
  (void)rYl;
  (void)rYr;

  const int rXl = static_cast<int>(L.rows());
  const int rXr = static_cast<int>(R.cols());

  // T1 (rXl, n_k * rYm) = L (rXl, rYl) @ right_unfold(Y_k) (rYl, n_k*rYm).
  const Eigen::Index t1_cols = static_cast<Eigen::Index>(n_k) * rYm;
  row_matrix T1(rXl, t1_cols);
  T1.noalias() = L * right_unfold(Y_k);

  // T2 (rYm * n_kp, rXr) = left_unfold(Y_kp1) (rYm*n_kp, rYr) @ R (rYr, rXr).
  const Eigen::Index t2_rows = static_cast<Eigen::Index>(rYm) * n_kp;
  row_matrix T2(t2_rows, rXr);
  T2.noalias() = left_unfold(Y_kp1) * R;

  // Reinterpret T1 as (rXl * n_k, rYm) and T2 as (rYm, n_kp * rXr).
  const Eigen::Index tv_rows = static_cast<Eigen::Index>(rXl) * n_k;
  const Eigen::Index tv_cols = static_cast<Eigen::Index>(n_kp) * rXr;
  Eigen::Map<const row_matrix> T1_v(T1.data(), tv_rows, rYm);
  Eigen::Map<const row_matrix> T2_v(T2.data(), rYm, tv_cols);

  row_matrix Theta(tv_rows, tv_cols);
  Theta.noalias() = T1_v * T2_v;
  return Theta;
}

// L->R split: SVD(Theta) = U s V^T, truncate to r_new <=
// truncate_eps_rank(s, delta, max_rank).  X_k_out gets U.leftCols(r_new)
// (left-orthogonal); X_kp1_out gets diag(s_head) * Vt.topRows(r_new)
// (orth center moves to k+1).  Returns r_new.  When ``s_head_out`` is
// non-null, it receives the truncated singular-value head (length
// r_new); used by dmrg_engine for sweep-internal convergence checks.
inline int split_two_site_lr(const row_matrix& Theta,
                             int rXl,
                             int n_k,
                             int n_kp,
                             int rXr,
                             double delta,
                             int max_rank,
                             tt_core& X_k_out,
                             tt_core& X_kp1_out,
                             eigen_bridge::col_vector* s_head_out = nullptr)
{
  eigen_bridge::row_matrix U;
  eigen_bridge::col_vector s;
  eigen_bridge::row_matrix Vt;
  const Eigen::Index svd_rows = static_cast<Eigen::Index>(rXl) * n_k;
  const Eigen::Index svd_cols = static_cast<Eigen::Index>(n_kp) * rXr;
  detail::svd_thin(Theta.data(), svd_rows, svd_cols, U, s, Vt);
  const int r_new = detail::truncate_eps_rank(s, delta, max_rank);

  // X_k_out: left_unfold = U.leftCols(r_new), shape (rXl*n_k, r_new).
  tt_core new_k(rXl, n_k, r_new);
  left_unfold(new_k) = U.leftCols(r_new);
  X_k_out            = std::move(new_k);

  // X_kp1_out: right_unfold = diag(s_head) * Vt.topRows(r_new),
  //            shape (r_new, n_kp * rXr).
  eigen_bridge::row_matrix F = diag_scale_rows(Vt, s, r_new);
  tt_core new_kp(r_new, n_kp, rXr);
  right_unfold(new_kp) = F;
  X_kp1_out            = std::move(new_kp);
  if (s_head_out)
  {
    *s_head_out = s.head(r_new);
  }
  return r_new;
}

// R->L split: same SVD, but fold s into U so the orth center moves to
// the LEFT core.  X_k_out gets U.leftCols(r_new) * diag(s_head) (orth
// center); X_kp1_out gets Vt.topRows(r_new) (right-orthogonal).
// Returns r_new.
inline int split_two_site_rl(const row_matrix& Theta,
                             int rXl,
                             int n_k,
                             int n_kp,
                             int rXr,
                             double delta,
                             int max_rank,
                             tt_core& X_k_out,
                             tt_core& X_kp1_out)
{
  eigen_bridge::row_matrix U;
  eigen_bridge::col_vector s;
  eigen_bridge::row_matrix Vt;
  detail::svd_thin(Theta.data(),
                   static_cast<Eigen::Index>(rXl) * n_k,
                   static_cast<Eigen::Index>(n_kp) * rXr,
                   U, s, Vt);
  const int r_new = detail::truncate_eps_rank(s, delta, max_rank);

  // X_k_out: left_unfold = U.leftCols(r_new) * diag(s_head),
  //          shape (rXl*n_k, r_new).
  eigen_bridge::row_matrix Us = diag_scale_cols(U, s, r_new);
  tt_core new_k(rXl, n_k, r_new);
  left_unfold(new_k) = Us;
  X_k_out            = std::move(new_k);

  // X_kp1_out: right_unfold = Vt.topRows(r_new), shape (r_new, n_kp*rXr).
  tt_core new_kp(r_new, n_kp, rXr);
  right_unfold(new_kp) = Vt.topRows(r_new);
  X_kp1_out            = std::move(new_kp);
  return r_new;
}

}  // namespace detail
}  // namespace tensor_train
}  // namespace mva
//
// :D
//
