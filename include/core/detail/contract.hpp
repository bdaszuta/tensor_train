/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Core-level contraction helpers built on the row-major unfolding views
*/
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>

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

// Replace ``c`` with M @ right_unfold(c) reshaped to (M.rows, n, r_right).
// Equivalently: pre-multiply the rank-left axis of ``c`` by ``M``.
//
// ``c`` must have left rank == M.cols().  Returns a fresh core whose
// left rank equals M.rows(), with the same n_phys and r_right as ``c``.
inline tt_core apply_left_factor(const row_matrix& M, const tt_core& c)
{
  const int n   = c.n_phys();
  const int rR  = c.r_right();
  if (M.cols() != static_cast<Eigen::Index>(c.r_left()))
  {
    std::fprintf(stderr,
                 "apply_left_factor: M.cols=%ld != c.r_left=%d\n",
                 static_cast<long>(M.cols()),
                 c.r_left());
    std::abort();
  }
  // rNL is always a TT bond rank, bounded by max_rank << INT_MAX.
  const int rNL = static_cast<int>(M.rows());
  // M @ right_unfold(c) (r_left, n*rR) -> (rNL, n*rR);
  // reinterpreted as (rNL, n, rR) by tt_core layout
  tt_core out(rNL, n, rR);
  right_unfold(out).noalias() = M * right_unfold(c);
  return out;
}

// Replace ``c`` with left_unfold(c) @ M reshaped to (r_left, n,
// M.cols()).  Equivalently: post-multiply the rank-right axis by ``M``.
//
// ``c`` must have right rank == M.rows().  Returns a fresh core whose
// right rank equals M.cols(), with the same r_left and n_phys as ``c``.
inline tt_core apply_right_factor(const tt_core& c, const row_matrix& M)
{
  const int rL  = c.r_left();
  const int n   = c.n_phys();
  if (M.rows() != static_cast<Eigen::Index>(c.r_right()))
  {
    std::fprintf(stderr,
                 "apply_right_factor: M.rows=%ld != c.r_right=%d\n",
                 static_cast<long>(M.rows()),
                 c.r_right());
    std::abort();
  }
  // rNR is always a TT bond rank, bounded by max_rank << INT_MAX.
  const int rNR = static_cast<int>(M.cols());
  tt_core out(rL, n, rNR);
  // left_unfold view of out is (rL*n, rNR); = left_unfold(c) (rL*n, rR) @ M.
  left_unfold(out).noalias() = left_unfold(c) * M;
  return out;
}

// One step of the right-to-left inner-product contraction for <a, b>.
// Given current ``F`` of shape (r_a', r_b') (right environment from
// position k+1), produce F' of shape (r_a, r_b) where r_a, r_b are the
// LEFT ranks of cores at position k:
//
//   F'[a, b] = sum_{a', j, b'}  ca(a, j, a') * cb(b, j, b') * F[a', b'].
//
// Implementation: two GEMMs mirroring contract_inner_step.
//   T1 (r_a*n, r_b') = left_unfold(ca) (r_a*n, r_a') @ F (r_a', r_b').
//   reinterpret T1 as (r_a, n*r_b') (same row-major bytes).
//   F' (r_a, r_b) = T1_as @ right_unfold(cb)^T (n*r_b' x r_b).
inline row_matrix contract_inner_step_right(const tt_core& ca,
                                            const tt_core& cb,
                                            const row_matrix& F)
{
  const int ral = ca.r_left();
  const int n   = ca.n_phys();
  const int rar = ca.r_right();
  const int rbl = cb.r_left();
  const int rbr = cb.r_right();

  // Validate dimension consistency (release-build safety).
  if (ca.n_phys() != cb.n_phys())
  {
    std::fprintf(stderr,
                 "contract_inner_step_right: n_phys mismatch "
                 "(%d vs %d)\n",
                 ca.n_phys(), cb.n_phys());
    std::abort();
  }
  if (F.rows() != static_cast<Eigen::Index>(rar))
  {
    std::fprintf(stderr,
                 "contract_inner_step_right: F.rows=%ld != ca.r_right=%d\n",
                 static_cast<long>(F.rows()),
                 rar);
    std::abort();
  }
  if (F.cols() != rbr)
  {
    std::fprintf(stderr,
                 "contract_inner_step_right: F.cols=%ld != cb.r_right=%d\n",
                 static_cast<long>(F.cols()),
                 rbr);
    std::abort();
  }

  // T1: (ral * n, F.cols) = left_unfold(ca) (ral*n, rar) @ F (rar, rbr).
  const Eigen::Index rows_t1 = static_cast<Eigen::Index>(ral) * n;
  row_matrix T1(rows_t1, F.cols());
  T1.noalias() = left_unfold(ca) * F;

  // Reinterpret T1 as (ral, n * F.cols) row-major; same bytes.
  Eigen::Map<const row_matrix> T1_as(
    T1.data(), ral, static_cast<Eigen::Index>(n) * F.cols());

  // F_out (ral, rbl) = T1_as (ral, n*rbr) @ right_unfold(cb)^T (n*rbr, rbl).
  row_matrix F_out(ral, rbl);
  F_out.noalias() = T1_as * right_unfold(cb).transpose();
  return F_out;
}

// One step of the left-to-right inner-product contraction for <a, b>.
// Given current ``E`` of shape (r_a, r_b), produce E' of shape
// (r_a', r_b') where r_a' = r_right(ca), r_b' = r_right(cb):
//
//   E'[p, q] = sum_{aa, j, bb}  E[aa, bb] * ca(aa, j, p) * cb(bb, j, q).
//
// Implementation: multiply R_env by left_unfold(ca) then transpose.
// Equivalent to E^T @ left_factor_view_a but simpler as two GEMMs.
//   tmp[bb, j, p] = sum_aa E[aa, bb] * ca(aa, j, p)
//                 reshape ca right_unfold (r_a, n*r_a') -> tmp_mat
//                 = E^T (r_b x r_a) @ ca_right_unfold (r_a x n*r_a')
//                 result (r_b x n*r_a').
//   E'[p, q] = sum_{bb, j} tmp[bb, j, p] * cb(bb, j, q)
//            = (tmp reshaped (r_b*n, r_a'))^T @ cb_left_unfold (r_b*n, r_b')
inline row_matrix contract_inner_step(const row_matrix& E,
                                      const tt_core& ca,
                                      const tt_core& cb)
{
  const int n   = ca.n_phys();
  const int ral = ca.r_left();
  const int rap = ca.r_right();
  const int rb  = cb.r_left();
  const int rbp = cb.r_right();

  // Validate dimension consistency (release-build safety).
  // E must have shape (ca.r_left, cb.r_left).
  if (ca.n_phys() != cb.n_phys())
  {
    std::fprintf(stderr,
                 "contract_inner_step: n_phys mismatch "
                 "(%d vs %d)\n",
                 ca.n_phys(), cb.n_phys());
    std::abort();
  }
  if (E.rows() != static_cast<Eigen::Index>(ral))
  {
    std::fprintf(stderr,
                 "contract_inner_step: E.rows=%ld != ca.r_left=%d\n",
                 static_cast<long>(E.rows()),
                 ral);
    std::abort();
  }
  if (E.cols() != static_cast<Eigen::Index>(rb))
  {
    std::fprintf(stderr,
                 "contract_inner_step: E.cols=%ld != cb.r_left=%d\n",
                 static_cast<long>(E.cols()),
                 rb);
    std::abort();
  }

  // tmp_mat: (rb, n * rap) = E^T (rb x ra) @ right_unfold(ca) (ra x n*rap)
  const Eigen::Index n_rap = static_cast<Eigen::Index>(n) * rap;
  row_matrix tmp_mat(rb, n_rap);
  tmp_mat.noalias() = E.transpose() * right_unfold(ca);

  // Reinterpret tmp_mat's buffer as (rb*n, rap) row-major (no copy).
  const Eigen::Index rb_n = static_cast<Eigen::Index>(rb) * n;
  Eigen::Map<const row_matrix> tmp_as(tmp_mat.data(), rb_n, rap);

  // E_new (rap, rbp) = tmp_as^T (rap x rb*n) @ left_unfold(cb) (rb*n x rbp)
  row_matrix E_new(rap, rbp);
  E_new.noalias() = tmp_as.transpose() * left_unfold(cb);
  return E_new;
}

}  // namespace detail
}  // namespace tensor_train
}  // namespace mva
//
// :D
//
