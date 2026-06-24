/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Block-concat helper for the tt_matrix axpby engine
*/
#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include "../types/tt_eigen_bridge.hpp"
#include "../types/tt_matrix.hpp"

namespace mva
{
namespace tensor_train
{
namespace detail
{

// Build the k-th matrix-core of  alpha * a + beta * b  for d >= 2
// tt_matrix inputs.  See block_concat.hpp for the layout convention;
// the only difference here is the 4-axis output shape.
// Rank additions are guarded against signed overflow.
inline tt_matrix_core axpby_matrix_core(int k,
                                        int d,
                                        const tt_matrix_core& ca,
                                        const tt_matrix_core& cb,
                                        double alpha,
                                        double beta)
{
  using mat_t     = eigen_bridge::row_matrix;
  using map_mut   = Eigen::Map<mat_t>;
  using map_const = Eigen::Map<const mat_t>;

  const int m        = ca.m_phys();
  const int n        = ca.n_phys();
  const Eigen::Index mn = static_cast<Eigen::Index>(m) * n;
  const int raL      = ca.r_left();
  const int raR      = ca.r_right();
  const int rbL      = cb.r_left();
  const int rbR      = cb.r_right();

  if (m != cb.m_phys() || n != cb.n_phys())
  {
    std::fprintf(stderr,
                 "axpby_matrix_core: mode size mismatch "
                 "ca.(m=%d,n=%d) != cb.(m=%d,n=%d)\n",
                 m, n, cb.m_phys(), cb.n_phys());
    std::abort();
  }

  const bool is_head = (k == 0);
  const bool is_tail = (k == d - 1);

  // Validate TT boundary-rank invariants (r_0 = 1, r_d = 1).
  if (is_head && (raL != 1 || rbL != 1))
  {
    std::fprintf(stderr,
                 "axpby_matrix_core: head ranks must be 1 "
                 "(got raL=%d, rbL=%d)\n", raL, rbL);
    std::abort();
  }
  if (is_tail && (raR != 1 || rbR != 1))
  {
    std::fprintf(stderr,
                 "axpby_matrix_core: tail ranks must be 1 "
                 "(got raR=%d, rbR=%d)\n", raR, rbR);
    std::abort();
  }

  // Guard rank additions against signed overflow.
  const int64_t rL64 = is_head ? 1
                       : static_cast<int64_t>(raL) + rbL;
  const int64_t rR64 = is_tail ? 1
                       : static_cast<int64_t>(raR) + rbR;
  if (rL64 > std::numeric_limits<int>::max() || rR64 > std::numeric_limits<int>::max())
  {
    std::fprintf(stderr,
                 "axpby_matrix_core: rank sum overflow raL+rbL=%lld, "
                 "raR+rbR=%lld; aborting.\n",
                 static_cast<long long>(rL64),
                 static_cast<long long>(rR64));
    std::abort();
  }
  const int rL = static_cast<int>(rL64);
  const int rR = static_cast<int>(rR64);

  tt_matrix_core out(rL, m, n, rR);
  out.zero_clear();

  // Treat each core as (rL * mn, rR) row-major.
  map_mut out_lu(out.data(), rL * mn, rR);
  map_const a_lu(ca.data(), raL * mn, raR);
  map_const b_lu(cb.data(), rbL * mn, rbR);

  // a-slab.
  for (int aa = 0; aa < raL; ++aa)
  {
    const Eigen::Index out_row_off = static_cast<Eigen::Index>(aa) * mn;
    const Eigen::Index src_row_off = static_cast<Eigen::Index>(aa) * mn;
    if (alpha == 1.0)
      out_lu.block(out_row_off, 0, mn, raR) =
        a_lu.block(src_row_off, 0, mn, raR);
    else
      out_lu.block(out_row_off, 0, mn, raR) =
        alpha * a_lu.block(src_row_off, 0, mn, raR);
  }

  // b-slab.
  const int b_left_offset = is_head ? 0 : raL;
  const int b_col_offset  = is_tail ? 0 : raR;
  for (int bb = 0; bb < rbL; ++bb)
  {
    const Eigen::Index out_row_off =
      static_cast<Eigen::Index>(b_left_offset + bb) * mn;
    const Eigen::Index src_row_off = static_cast<Eigen::Index>(bb) * mn;
    if (beta == 1.0)
      out_lu.block(out_row_off, b_col_offset, mn, rbR) =
        b_lu.block(src_row_off, 0, mn, rbR);
    else
      out_lu.block(out_row_off, b_col_offset, mn, rbR) =
        beta * b_lu.block(src_row_off, 0, mn, rbR);
  }
  return out;
}

}  // namespace detail
}  // namespace tensor_train
}  // namespace mva
//
// :D
//
