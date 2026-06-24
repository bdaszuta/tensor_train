/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Left-to-right and right-to-left truncated-SVD steps used by TT-rounding and TT-SVD
*/
#pragma once

#include <Eigen/Core>

#include "../types/tt_core.hpp"
#include "../types/tt_eigen_bridge.hpp"
#include "contract.hpp"
#include "core_view.hpp"

namespace mva
{
namespace tensor_train
{
namespace detail
{

// Scales the top r rows of M by diag(s[0..r-1]), returns the scaled
// block.  Used to compute s_head * Vt_head in SVD truncation steps.
inline eigen_bridge::row_matrix diag_scale_rows(
  const eigen_bridge::row_matrix& M,
  const eigen_bridge::col_vector& s,
  int r)
{
  eigen_bridge::row_matrix F = M.topRows(r);
  for (int i = 0; i < r; ++i)
  {
    F.row(i) *= s[i];
  }
  return F;
}

// Scales the first r columns of M by diag(s[0..r-1]).  Used in
// right-to-left truncation steps: builds U_head * diag(s_head).
inline eigen_bridge::row_matrix diag_scale_cols(
  const eigen_bridge::row_matrix& M,
  const eigen_bridge::col_vector& s,
  int r)
{
  eigen_bridge::row_matrix F = M.leftCols(r);
  for (int j = 0; j < r; ++j)
  {
    F.col(j) *= s[j];
  }
  return F;
}

// In-place: truncate c_cur via SVD; absorb (s * Vt) into c_next.
// Returns the new bond rank r_new.
//
// For tall left-unfolds (rows >= 2*cols) the SVD is QR-condensed:
//   M = Q R,  R = Ur s Vt_r,  U = Q Ur,  Vt = Vt_r,
// turning a (rows, cols) BDCSVD into a (cols, cols) one plus a thin QR.
inline int svd_step(tt_core& c_cur,
                    tt_core& c_next,
                    double delta,
                    int max_rank = 0)
{
  const int r    = c_cur.r_left();
  const int n    = c_cur.n_phys();
  const int rp   = c_cur.r_right();
  const Eigen::Index rows = static_cast<Eigen::Index>(r) * n;
  const int cols = rp;
  // left_unfold(c_cur) is (rows, cols) row-major in the buffer.
  eigen_bridge::row_matrix U;
  eigen_bridge::col_vector s;
  eigen_bridge::row_matrix Vt;

  if (rows >= 2 * static_cast<Eigen::Index>(cols) && cols > 0)
  {
    // Tall: QR-condense to a (cols, cols) SVD.
    eigen_bridge::row_matrix Q, R;
    detail::qr_thin(c_cur.data(), rows, cols, Q, R);
    eigen_bridge::row_matrix Ur, Vt_r;
    detail::svd_thin(R.data(), cols, cols, Ur, s, Vt_r);
    U  = Q * Ur;           // (rows, cols)
    Vt = std::move(Vt_r);  // (cols, cols)
  }
  else
  {
    detail::svd_thin(c_cur.data(), rows, cols, U, s, Vt);
  }
  const int r_new = detail::truncate_eps_rank(s, delta, max_rank);

  // New c_cur: left_unfold(new_cur) = U.leftCols(r_new), shape (r*n, r_new).
  tt_core new_cur(r, n, r_new);
  left_unfold(new_cur) = U.leftCols(r_new);
  c_cur                = std::move(new_cur);

  c_next = apply_left_factor(diag_scale_rows(Vt, s, r_new), c_next);
  return r_new;
}

// Right-to-left counterpart: truncate c_cur on its LEFT-rank axis via
// the SVD of right_unfold(c_cur) = (r, n*rp) row-major.  Let
//   right_unfold(c_cur) = U s V^T,  U (r,k), Vt (k, n*rp).
// Pick r_new = truncate_eps_rank(s, delta, max_rank).  Set
//   right_unfold(new c_cur) = Vt.topRows(r_new),  shape (r_new, n, rp);
// absorb F = U.leftCols(r_new) * diag(s_head) (shape (r, r_new)) into
// the right-rank axis of c_prev: c_prev <- apply_right_factor(c_prev, F).
// Returns r_new.
//
// QR-condense paths:
//   wide (cols >= 2*rows): M^T = Q R, then svd(R^T) gives U directly
//     and Vt = Vt_r * Q^T.  Common case in TT R->L sweeps.
//   tall (rows >= 2*cols): M = Q R, svd(R) then U = Q*Ur, Vt = Vt_r.
inline int svd_step_right(tt_core& c_cur,
                          tt_core& c_prev,
                          double delta,
                          int max_rank = 0)
{
  const int r    = c_cur.r_left();
  const int n    = c_cur.n_phys();
  const int rp   = c_cur.r_right();
  const int rows = r;
  const Eigen::Index cols = static_cast<Eigen::Index>(n) * rp;
  // right_unfold(c_cur) is (rows, cols) row-major in the buffer.
  eigen_bridge::row_matrix U;
  eigen_bridge::col_vector s;
  eigen_bridge::row_matrix Vt;

  if (cols >= 2 * static_cast<Eigen::Index>(rows) && rows > 0)
  {
    // Wide: M^T = Q R, with Q (cols, rows), R (rows, rows).
    // Build M^T as a row-major (cols, rows) matrix.
    Eigen::Map<const eigen_bridge::row_matrix> M(c_cur.data(), rows, cols);
    eigen_bridge::row_matrix M_T = M.transpose();
    eigen_bridge::row_matrix Q, R;
    detail::qr_thin(M_T.data(), cols, rows, Q, R);
    // svd(R^T) = Ur s Vt_r;  M = Ur s Vt_r Q^T.
    eigen_bridge::row_matrix Rt = R.transpose();
    eigen_bridge::row_matrix Ur, Vt_r;
    detail::svd_thin(Rt.data(), rows, rows, Ur, s, Vt_r);
    U  = std::move(Ur);         // (rows, rows)
    Vt = Vt_r * Q.transpose();  // (rows, cols)
  }
  else if (rows >= 2 * cols && cols > 0)
  {
    eigen_bridge::row_matrix Q, R;
    detail::qr_thin(c_cur.data(), rows, cols, Q, R);
    eigen_bridge::row_matrix Ur, Vt_r;
    detail::svd_thin(R.data(), cols, cols, Ur, s, Vt_r);
    U  = Q * Ur;
    Vt = std::move(Vt_r);
  }
  else
  {
    detail::svd_thin(c_cur.data(), rows, cols, U, s, Vt);
  }
  const int r_new = detail::truncate_eps_rank(s, delta, max_rank);

  // New c_cur: right_unfold(new_cur) = Vt.topRows(r_new), shape (r_new, n*rp).
  tt_core new_cur(r_new, n, rp);
  right_unfold(new_cur) = Vt.topRows(r_new);
  c_cur                 = std::move(new_cur);

  c_prev = apply_right_factor(c_prev, diag_scale_cols(U, s, r_new));
  return r_new;
}

}  // namespace detail
}  // namespace tensor_train
}  // namespace mva
//
// :D
//
