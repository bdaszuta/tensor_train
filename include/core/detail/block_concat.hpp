/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Block-concat helper for the TT axpby engine
*/
#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include "../types/tt_core.hpp"
#include "../types/tt_eigen_bridge.hpp"
#include "core_view.hpp"

namespace mva
{
namespace tensor_train
{
namespace detail
{

// Build the k-th core of  alpha * a + beta * b  for d >= 2 TTs.
//
// Output ranks:
//   r_left  = (k == 0     ? 1 : raL + rbL)
//   r_right = (k == d - 1 ? 1 : raR + rbR)
// where raL/raR/rbL/rbR are the bond ranks of ca and cb.
// Rank additions are guarded against signed overflow.
//
// The alpha factor is folded into the a-slab of the leading core (k==0),
// and the beta factor into the b-slab of the leading core, so each
// scalar is multiplied exactly once over the chain.  Pass alpha = beta
// = 1 for non-leading cores (caller's responsibility -- see axpby).
inline tt_core axpby_core(int k,
                          int d,
                          const tt_core& ca,
                          const tt_core& cb,
                          double alpha,
                          double beta)
{
  if (d <= 1)
  {
    std::fprintf(stderr,
                 "axpby_core: requires d >= 2 (got d=%d)\n", d);
    std::abort();
  }
  const int n        = ca.n_phys();
  const int raL      = ca.r_left();
  const int raR      = ca.r_right();
  const int rbL      = cb.r_left();
  const int rbR      = cb.r_right();

  if (n != cb.n_phys())
  {
    std::fprintf(stderr,
                 "axpby_core: mode size mismatch ca.n=%d != cb.n=%d\n",
                 n, cb.n_phys());
    std::abort();
  }

  const bool is_head = (k == 0);
  const bool is_tail = (k == d - 1);

  // Validate TT boundary-rank invariants (r_0 = 1, r_d = 1).
  if (is_head && raL != 1)
  {
    std::fprintf(stderr,
                 "axpby_core: head rank must be 1 (got raL=%d)\n", raL);
    std::abort();
  }
  if (is_head && rbL != 1)
  {
    std::fprintf(stderr,
                 "axpby_core: head rank must be 1 (got rbL=%d)\n", rbL);
    std::abort();
  }
  if (is_tail && raR != 1)
  {
    std::fprintf(stderr,
                 "axpby_core: tail rank must be 1 (got raR=%d)\n", raR);
    std::abort();
  }
  if (is_tail && rbR != 1)
  {
    std::fprintf(stderr,
                 "axpby_core: tail rank must be 1 (got rbR=%d)\n", rbR);
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
                 "axpby_core: rank sum overflow raL+rbL=%lld, raR+rbR=%lld; "
                 "aborting.\n",
                 static_cast<long long>(rL64),
                 static_cast<long long>(rR64));
    std::abort();
  }
  const int rL = static_cast<int>(rL64);
  const int rR = static_cast<int>(rR64);

  tt_core out(rL, n, rR);
  out.zero_clear();
  auto out_lu = left_unfold(out);  // (rL * n, rR)
  auto a_lu   = left_unfold(ca);   // (raL * n, raR)
  auto b_lu   = left_unfold(cb);   // (rbL * n, rbR)

  // a-slab: rows [aa*n, (aa+1)*n) for aa in [0, raL), cols [0, raR).
  // Output left index for that block = aa (head) or aa (interior).
  // For the tail core, raR == rR == 1 still works.
  for (int aa = 0; aa < raL; ++aa)
  {
    const Eigen::Index out_row_off = static_cast<Eigen::Index>(aa) * n;
    const Eigen::Index src_row_off = static_cast<Eigen::Index>(aa) * n;
    if (alpha == 1.0)
      out_lu.block(out_row_off, 0, n, raR) =
        a_lu.block(src_row_off, 0, n, raR);
    else
      out_lu.block(out_row_off, 0, n, raR) =
        alpha * a_lu.block(src_row_off, 0, n, raR);
  }

  // b-slab.  For the head core (rL == 1), b's left index is still 0
  // and the b-slab columns sit at [raR, raR + rbR).  For interior /
  // tail cores, b's left index maps to (raL + bb).
  const int b_left_offset = is_head ? 0 : raL;
  const int b_col_offset  = is_tail ? 0 : raR;
  for (int bb = 0; bb < rbL; ++bb)
  {
    const Eigen::Index out_row_off =
      static_cast<Eigen::Index>(b_left_offset + bb) * n;
    const Eigen::Index src_row_off = static_cast<Eigen::Index>(bb) * n;
    if (beta == 1.0)
    {
      out_lu.block(out_row_off, b_col_offset, n, rbR) =
        b_lu.block(src_row_off, 0, n, rbR);
    }
    else
    {
      out_lu.block(out_row_off, b_col_offset, n, rbR) =
        beta * b_lu.block(src_row_off, 0, n, rbR);
    }
  }
  return out;
}

}  // namespace detail
}  // namespace tensor_train
}  // namespace mva
//
// :D
//
