/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Soft thresholding for TT bond ranks
*/
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>
#include <limits>

#include "../detail/core_view.hpp"
#include "../detail/truncate.hpp"
#include "../gauge/tt_orthogonalize.hpp"
#include "../types/tt.hpp"
#include "../types/tt_core.hpp"
#include "../types/tt_eigen_bridge.hpp"

namespace mva
{
namespace tensor_train
{

/**
 * @brief Singular-value soft-thresholding of a TT.
 * Performs SVD truncation where the new singular values are
 * max(SV - tau, 0).  This is the proximal operator for the
 * nuclear norm (sum of singular values).
 * @param a   Input TT.
 * @param tau Soft-threshold parameter (default 0 = no-op).
 * @return Soft-thresholded copy of a.
 * @note  When tau <= 0 this is a no-op (returns a deep copy).
 */
inline tt soft_threshold(const tt& a, double tau = 0.0)
{
  const int d = a.d();
  if (d <= 1 || tau <= 0.0)
  {
    // d <= 1: no bonds to threshold.  tau <= 0: no-op.
    return tt(a);  // deep copy
  }

  // Mutable copy; right-canonicalise so the R->L SVD is exact.
  tt acc = a;   // implicit deep-copy via tt copy-ctor
  right_orthogonalize(acc);

  auto& cs   = acc.cores();
  using rm   = eigen_bridge::row_matrix;
  using cv   = eigen_bridge::col_vector;

  for (int k = d - 1; k >= 1; --k)
  {
    tt_core& c_prev = cs[k - 1];
    tt_core& c_cur  = cs[k];

    const int rL  = c_prev.r_left();
    const int n_p = c_prev.n_phys();
    const int n_c = c_cur.n_phys();
    const int rR  = c_cur.r_right();
    // Bond rank between c_prev and c_cur: r = c_prev.r_right() == c_cur.r_left().

    // Guard dimension products against int overflow.
    {
      const int64_t rows64 = static_cast<int64_t>(rL) * n_p;
      const int64_t cols64 = static_cast<int64_t>(n_c) * rR;
      if (rows64 > std::numeric_limits<int>::max() || rows64 < 0
          || cols64 > std::numeric_limits<int>::max() || cols64 < 0)
      {
        std::fprintf(stderr,
                     "soft_threshold: dimension product exceeds INT_MAX "
                     "(rL=%d * n_p=%d = %ld, n_c=%d * rR=%d = %ld)\n",
                     rL, n_p, static_cast<long>(rows64),
                     n_c, rR, static_cast<long>(cols64));
        std::abort();
      }
    }

    // Contract: M = left_unfold(c_prev) * right_unfold(c_cur)
    // shapes: (rL * n_p, r) * (r, n_c * rR) -> (rL * n_p, n_c * rR)
    rm M(static_cast<Eigen::Index>(rL) * n_p,
         static_cast<Eigen::Index>(n_c) * rR);
    {
      const auto lu = detail::left_unfold(c_prev);
      const auto ru = detail::right_unfold(c_cur);
      M.noalias() = lu * ru;
    }

    rm U, Vt;
    cv s_full;
    detail::svd_thin(M.data(),
                     static_cast<Eigen::Index>(rL) * n_p,
                     static_cast<Eigen::Index>(n_c) * rR,
                     U, s_full, Vt);

    // Apply soft threshold to singular values.
    const int n_sv = static_cast<int>(s_full.size());
    cv s_shrunk = s_full;
    int r_new = n_sv;
    {
      for (int j = 0; j < n_sv; ++j)
      {
        const double sv = s_full[j];
        if (sv <= tau)
        {
          s_shrunk[j] = 0.0;
        }
        else
        {
          s_shrunk[j] = sv - tau;
          r_new       = j + 1;   // largest index with non-zero value
        }
      }
    }

    // If every singular value was zeroed, both cores become zero.
    // Skip rank-1 guard: we genuinely want a zero TT here.
    if (s_shrunk.size() > 0 && s_shrunk[0] <= 0.0)
    {
      // All zero: produce two trivial rank-1 zero cores.
      tt_core z_prev(rL, n_p, 1);
      z_prev.zero_clear();
      tt_core z_cur(1, n_c, rR);
      z_cur.zero_clear();
      c_prev = std::move(z_prev);
      c_cur  = std::move(z_cur);
      continue;
    }

    // Retain only non-zero singular values.
    // Trailing zeros beyond r_new carry no energy -- truncate them.
    const int r_keep = r_new;

    // Absorb U * diag(s_shrunk) into c_prev (left core).
    // left_unfold(new_prev) = U[:, :r_keep] * diag(s_shrunk_head)
    if (r_keep < n_sv)
    {
      // Explicitly build the truncated product to drop zero columns.
      tt_core new_prev(rL, n_p, r_keep);
      auto lu_new = detail::left_unfold(new_prev);
      lu_new      = U.leftCols(r_keep);
      for (int j = 0; j < r_keep; ++j)
      {
        if (s_shrunk[j] > 0.0)
          lu_new.col(j) *= s_shrunk[j];
      }
      c_prev = std::move(new_prev);
    }
    else
    {
      tt_core new_prev(rL, n_p, r_keep);
      auto lu_new = detail::left_unfold(new_prev);
      lu_new      = U;
      for (int j = 0; j < r_keep; ++j)
      {
        if (s_shrunk[j] > 0.0)
          lu_new.col(j) *= s_shrunk[j];
      }
      c_prev = std::move(new_prev);
    }

    // Absorb Vt_head into c_cur (right core).
    // right_unfold(new_cur) = Vt.topRows(r_keep)
    tt_core new_cur(r_keep, n_c, rR);
    detail::right_unfold(new_cur) = Vt.topRows(r_keep);
    c_cur                        = std::move(new_cur);
  }

  return acc;
}

}  // namespace tensor_train
}  // namespace mva
//
// :D
//
