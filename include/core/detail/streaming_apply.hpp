/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Streaming apply-and-round for tt_matrix * tt-vector and tt_matrix * tt-matrix. Interleaves GEMM with per-core truncation to avoid materialising the full rank-multiplied intermediate.
*/
#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "../apply/tt_matrix_apply.hpp"
#include "../gauge/tt_matrix_orthogonalize.hpp"
#include "../gauge/tt_orthogonalize.hpp"
#include "../types/tt.hpp"
#include "../types/tt_core.hpp"
#include "../types/tt_eigen_bridge.hpp"
#include "../types/tt_matrix.hpp"
#include "core_view.hpp"
#include "fused_apply_core.hpp"
#include "shape_check.hpp"
#include "truncate.hpp"

namespace mva
{
namespace tensor_train
{
namespace detail
{

// Left-to-right streaming matvec sweep: right-orthogonalize inputs,
// compute per-step delta, iterate cores with fused_matvec_core,
// SVD-truncate intermediate cores, propagate F environment forward.
inline tt build_streaming_matvec(const tt_matrix& a_in,
                                 const tt& x_in,
                                 double eps,
                                 int max_rank)
{
  detail::check_matvec_compatible(a_in, x_in,
                                   "build_streaming_matvec");
  const int d = a_in.d();
  if (d == 0)
  {
    std::fprintf(stderr,
                 "build_streaming_matvec: d=0 not supported\n");
    std::abort();
  }

  // Local mutable copies; right-orthogonalise without touching inputs.
  tt_matrix A = a_in;
  tt x        = x_in;
  right_orthogonalize(A);
  right_orthogonalize(x);

  const double norm_A = frob_norm_buffer(A.core(0).data(), A.core(0).size());
  const double norm_x = frob_norm_buffer(x.core(0).data(), x.core(0).size());
  const double norm   = norm_A * norm_x;
  if (norm == 0.0)
  {
    return make_rank1_zero_tt(a_in.row_shape());
  }
  const double delta =
    eps * norm / std::sqrt(static_cast<double>(std::max(d - 1, 1)));

  using row_mat = eigen_bridge::row_matrix;

  // F_0: shape (1, rL_A * rL_x) = (1, 1).
  row_mat F(1, 1);
  F(0, 0) = 1.0;

  std::vector<tt_core> out_cores;
  out_cores.reserve(static_cast<std::size_t>(d));

  for (int k = 0; k < d; ++k)
  {
    tt_core Cj      = fused_matvec_core(F, A.core(k), x.core(k));
    const int alpha = Cj.r_left();
    const int mk    = Cj.n_phys();
    const int br    = Cj.r_right();

    if (k == d - 1)
    {
      out_cores.push_back(std::move(Cj));
      break;
    }

    // SVD of left_unfold(Cj) shape (alpha * mk, br).
    const Eigen::Index rows0 = static_cast<Eigen::Index>(alpha) * mk;
    eigen_bridge::row_matrix U;
    eigen_bridge::col_vector s;
    eigen_bridge::row_matrix Vt;
    detail::svd_thin(Cj.data(), rows0, br, U, s, Vt);
    const int r_new = detail::truncate_eps_rank(s, delta, max_rank);

    tt_core new_core(alpha, mk, r_new);
    left_unfold(new_core) = U.leftCols(r_new);
    out_cores.push_back(std::move(new_core));

    // F_{k+1} = diag(s_head) * Vt_head, shape (r_new, br).
    row_mat F_next = diag_scale_rows(Vt, s, r_new);
    F              = std::move(F_next);
  }

  return tt(std::move(out_cores));
}

// Left-to-right streaming matmat sweep: right-orthogonalize inputs,
// compute per-step delta, iterate cores with fused_matmat_core,
// SVD-truncate intermediate cores, propagate F environment forward.
inline tt_matrix build_streaming_matmat(const tt_matrix& a_in,
                                        const tt_matrix& b_in,
                                        double eps,
                                        int max_rank)
{
  detail::check_matmat_compatible(a_in, b_in,
                                   "build_streaming_matmat");
  const int d = a_in.d();
  if (d == 0)
  {
    std::fprintf(stderr,
                 "build_streaming_matmat: d=0 not supported\n");
    std::abort();
  }

  tt_matrix A = a_in;
  tt_matrix B = b_in;
  right_orthogonalize(A);
  right_orthogonalize(B);

  const double norm_A = frob_norm_buffer(A.core(0).data(), A.core(0).size());
  const double norm_B = frob_norm_buffer(B.core(0).data(), B.core(0).size());
  const double norm   = norm_A * norm_B;
  if (norm == 0.0)
  {
    const std::vector<int> rs = a_in.row_shape();
    const std::vector<int> cs = b_in.col_shape();
    std::vector<tt_matrix_core> z;
    z.reserve(static_cast<std::size_t>(d));
    for (int k = 0; k < d; ++k)
    {
      tt_matrix_core mc(1,
                        rs[static_cast<std::size_t>(k)],
                        cs[static_cast<std::size_t>(k)],
                        1);
      mc.zero_clear();
      z.push_back(std::move(mc));
    }
    return tt_matrix(std::move(z));
  }
  const double delta =
    eps * norm / std::sqrt(static_cast<double>(std::max(d - 1, 1)));

  using row_mat = eigen_bridge::row_matrix;

  row_mat F(1, 1);
  F(0, 0) = 1.0;

  std::vector<tt_matrix_core> out_cores;
  out_cores.reserve(static_cast<std::size_t>(d));

  for (int k = 0; k < d; ++k)
  {
    tt_matrix_core Cj = fused_matmat_core(F, A.core(k), B.core(k));
    const int alpha   = Cj.r_left();
    const int mk      = Cj.m_phys();
    const int nk      = Cj.n_phys();
    const int br      = Cj.r_right();

    if (k == d - 1)
    {
      out_cores.push_back(std::move(Cj));
      break;
    }

    // Reinterpret Cj's buffer as a 3-axis (alpha, mk * nk, br) core;
    // SVD left unfold is (alpha * mk * nk, br).
    const Eigen::Index rows = static_cast<Eigen::Index>(alpha) * mk * nk;
    eigen_bridge::row_matrix U;
    eigen_bridge::col_vector s;
    eigen_bridge::row_matrix Vt;
    detail::svd_thin(Cj.data(), rows, br, U, s, Vt);
    const int r_new = detail::truncate_eps_rank(s, delta, max_rank);

    // U.leftCols(r_new) row-major shape (alpha * mk * nk, r_new) is
    // bit-identical to a tt_matrix_core (alpha, mk, nk, r_new) layout
    // (alpha, i, j, b) row-major.
    tt_matrix_core new_core(alpha, mk, nk, r_new);
    Eigen::Map<row_mat> nc_view(new_core.data(), rows, r_new);
    nc_view = U.leftCols(r_new);
    out_cores.push_back(std::move(new_core));

    row_mat F_next = diag_scale_rows(Vt, s, r_new);
    F              = std::move(F_next);
  }

  return tt_matrix(std::move(out_cores));
}

}  // namespace detail
}  // namespace tensor_train
}  // namespace mva
//
// :D
//
