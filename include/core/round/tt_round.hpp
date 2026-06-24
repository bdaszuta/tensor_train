/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: TT rounding engines (round, round_right_canonical) -- QR + truncated SVD
*/
#pragma once

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "../detail/shape_check.hpp"
#include "../detail/truncate.hpp"
#include "../gauge/tt_orthogonalize.hpp"
#include "../types/tt.hpp"
#include "../types/tt_core.hpp"

namespace mva
{
namespace tensor_train
{
namespace detail
{

// Re-orthogonalise then truncate ``a`` to the smallest TT-ranks
// achieving ||a - round(a)||_F <= eps * ||a||_F.  When ``max_rank > 0``
// each bond rank is also hard-capped to ``max_rank``.
//
// The rvalue overload moves the input cores in-place; the const-ref
// overload deep-copies them first.  All algorithmic work happens in
// the rvalue path.
inline tt round(tt&& acc, double eps = 1.0e-10, int max_rank = 0)
{
  const int d = acc.d();
  if (d <= 1)
  {
    return tt(std::move(acc));
  }

  // ---------------- right-to-left QR sweep ----------------
  // After this sweep, cores[1..d-1] are right-orthogonal and cores[0]
  // absorbs the trailing L factors.
  right_orthogonalize(acc);
  auto& cs = acc.cores();

  // ---------------- norm + delta ----------------
  // Frobenius norm = ||cores[0]|| since right cores are orthonormal.
  const double norm = frob_norm_buffer(cs[0].data(), cs[0].size());
  if (norm == 0.0)
  {
    return make_rank1_zero_tt(acc.shape());
  }
  const double delta = eps * norm / std::sqrt(std::max(d - 1, 1));

  // ---------------- left-to-right truncated SVD ----------------
  for (int k = 0; k < d - 1; ++k)
  {
    detail::svd_step(cs[k], cs[k + 1], delta, max_rank);
  }

  // ---------------- norm balancing (defense-in-depth) ----------------
  // After the L->R SVD sweep, cores[0..d-2] have orthonormal columns
  // under left_unfold (left-orthogonal gauge).  The balancing step
  // below DESTROYS this gauge: columns are scaled to a common geometric
  // mean and reciprocal scaling is applied to the next core's rows.
  // The TT tensor is unchanged (the scalars cancel), but cores[0..d-2]
  // are no longer left-orthogonal after this step.
  //
  // To guarantee a specific gauge, call right_orthogonalize() or
  // left_orthogonalize() after round(), or use round_right_canonical()
  // which preserves right-canonical gauge and skips norm balancing.
  {
    // Phase 1: collect column norms of left_unfold for cores 0..d-2.
    std::vector<std::vector<double>> col_norms(
      static_cast<std::size_t>(d));
    double total_log = 0.0;
    int total_cols   = 0;
    for (int k = 0; k < d - 1; ++k)
    {
      int ncols = cs[k].r_right();
      auto luf = left_unfold(cs[k]);
      col_norms[static_cast<std::size_t>(k)].resize(
        static_cast<std::size_t>(ncols));
      for (int c = 0; c < ncols; ++c)
      {
        double nc = luf.col(c).norm();
        col_norms[static_cast<std::size_t>(k)]
                 [static_cast<std::size_t>(c)] = nc;
        if (nc > 0.0)
        {
          total_log += std::log(nc);
          ++total_cols;
        }
      }
    }

    // Phase 2: scale columns to geometric mean, invert on next core.
    if (total_cols > 0)
    {
      double geo = std::exp(total_log / static_cast<double>(total_cols));
      for (int k = 0; k < d - 1; ++k)
      {
        int ncols = cs[k].r_right();
        auto luf_k   = left_unfold(cs[k]);
        auto ruf_kp1 = right_unfold(cs[k + 1]);
        for (int c = 0; c < ncols; ++c)
        {
          double nc =
            col_norms[static_cast<std::size_t>(k)]
                     [static_cast<std::size_t>(c)];
          if (nc > 0.0)
          {
            double scl = geo / nc;
            luf_k.col(c) *= scl;
            ruf_kp1.row(c) /= scl;
          }
        }
      }
    }
  }

  return tt(std::move(acc));
}

inline tt round(const tt& a, double eps = 1.0e-10, int max_rank = 0)
{
  // Deep-copy then defer to the rvalue overload.
  std::vector<tt_core> cores(a.cores().begin(), a.cores().end());
  return round(tt(std::move(cores)), eps, max_rank);
}

// Right-canonical-output variant of round().  Same numerical contract
// (||a - round_right_canonical(a)||_F <= eps * ||a||_F, with optional
// per-bond cap) but the output gauge is RIGHT-canonical: cores[1..d-1]
// have orthonormal rows under right_unfold (orth center at site 0).
//
// Use case: ALS / DMRG cold inits build a right-canonical X to seed the
// engine's R-environment construction.  The default round() applies a
// norm-balancing step that destroys gauge (cores[0..d-2] are no longer
// left-orthogonal after balancing), so its output has no guaranteed
// gauge.  round_right_canonical preserves right-canonical gauge and
// skips norm balancing, fusing the gauge-fix into the truncation sweep.
//
// Implementation: left-orthogonalise (L->R QR sweep), then truncate
// right-to-left via svd_step_right.  Equivalent flop count to round();
// only the sweep direction and final gauge differ.
inline tt round_right_canonical(tt&& acc,
                                double eps   = 1.0e-10,
                                int max_rank = 0)
{
  const int d = acc.d();
  if (d <= 1)
  {
    return tt(std::move(acc));
  }

  // ---------------- left-to-right QR sweep ----------------
  // After this sweep, cores[0..d-2] are left-orthogonal and cores[d-1]
  // absorbs the trailing R factors.
  left_orthogonalize(acc);
  auto& cs = acc.cores();

  // ---------------- norm + delta ----------------
  // Frobenius norm = ||cores[d-1]|| since left cores are orthonormal.
  const double norm = frob_norm_buffer(cs[d - 1].data(), cs[d - 1].size());
  if (norm == 0.0)
  {
    return make_rank1_zero_tt(acc.shape());
  }
  const double delta = eps * norm / std::sqrt(std::max(d - 1, 1));

  // ---------------- right-to-left truncated SVD ----------------
  for (int k = d - 1; k >= 1; --k)
  {
    detail::svd_step_right(cs[k], cs[k - 1], delta, max_rank);
  }

  return tt(std::move(acc));
}

inline tt round_right_canonical(const tt& a,
                                double eps   = 1.0e-10,
                                int max_rank = 0)
{
  std::vector<tt_core> cores(a.cores().begin(), a.cores().end());
  return round_right_canonical(tt(std::move(cores)), eps, max_rank);
}

}  // namespace detail
}  // namespace tensor_train
}  // namespace mva
//
// :D
//
