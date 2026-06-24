/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Left-to-right and right-to-left QR sweep steps used by TT-rounding gauge conditioning
*/
#pragma once

#include <Eigen/Core>
#include <Eigen/QR>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

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

// In-place: left-orthogonalise c_cur and absorb R into c_next.
//
// left_unfold(c_cur) is row-major (r*n, rp) -- directly suitable for a
// thin row-major QR.  Let M = Q R with Q (r*n, k) orthonormal columns,
// R (k, rp) upper triangular, k = min(r*n, rp).  Replace c_cur by Q
// reshaped to (r, n, k); pre-multiply c_next's left-rank axis by R.
inline void left_qr_step(tt_core& c_cur, tt_core& c_next)
{
  if (c_cur.r_right() != c_next.r_left())
  {
    std::fprintf(stderr,
                 "left_qr_step: rank mismatch c_cur.r_right=%d "
                 "!= c_next.r_left=%d\n",
                 c_cur.r_right(),
                 c_next.r_left());
    std::abort();
  }
  const int r  = c_cur.r_left();
  const int n  = c_cur.n_phys();
  const int rp = c_cur.r_right();
  const Eigen::Index m = static_cast<Eigen::Index>(r) * n;
  const int k  = static_cast<int>(std::min(m, static_cast<Eigen::Index>(rp)));

  eigen_bridge::row_matrix Q;
  eigen_bridge::row_matrix R;
  detail::qr_thin(c_cur.data(), m, rp, Q, R);

  tt_core new_cur(r, n, k);
  left_unfold(new_cur) = Q;
  c_next               = apply_left_factor(R, c_next);
  c_cur                = std::move(new_cur);
}

// In-place: right-orthogonalise c_cur and absorb L (row-major R-factor) into c_prev.
inline void right_qr_step(tt_core& c_cur, tt_core& c_prev)
{
  if (c_cur.r_left() != c_prev.r_right())
  {
    std::fprintf(stderr,
                 "right_qr_step: rank mismatch c_cur.r_left=%d "
                 "!= c_prev.r_right=%d\n",
                 c_cur.r_left(),
                 c_prev.r_right());
    std::abort();
  }
  const int r  = c_cur.r_left();
  const int n  = c_cur.n_phys();
  const int rp = c_cur.r_right();
  const Eigen::Index m = static_cast<Eigen::Index>(n) * rp;

  // M^T (m, r) col-major view of c_cur's row-major buffer (r, m).
  using col_matrix_d =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
  Eigen::Map<const col_matrix_d> Mt_cm(c_cur.data(), m, r);
  Eigen::HouseholderQR<col_matrix_d> qr(Mt_cm);
  const int r_eff = static_cast<int>(std::min(m, static_cast<Eigen::Index>(r)));

  // New c_cur: (r_eff, n, rp).  Its row-major (r_eff, m) buffer ==
  // col-major (m, r_eff): write thin-Q directly there.
  tt_core new_cur(r_eff, n, rp);
  Eigen::Map<col_matrix_d> Q_dest(new_cur.data(), m, r_eff);
  Q_dest.setIdentity();
  Q_dest.applyOnTheLeft(qr.householderQ());

  // R (r_eff, r) col-major; its bytes equal row-major (r, r_eff) = L.
  col_matrix_d R_cm =
    qr.matrixQR().topRows(r_eff).template triangularView<Eigen::Upper>();
  Eigen::Map<eigen_bridge::row_matrix> L(R_cm.data(), r, r_eff);
  c_prev = apply_right_factor(c_prev, L);

  c_cur = std::move(new_cur);
}

}  // namespace detail
}  // namespace tensor_train
}  // namespace mva
//
// :D
//
