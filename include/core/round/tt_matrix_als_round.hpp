/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: ALS-based TT-matrix rounding. Packs the matrix into a tt, delegates to als_round(), then unpacks.
*/
#pragma once

#include <utility>
#include <vector>

#include "../detail/matrix_pack.hpp"
#include "../detail/warm_start.hpp"
#include "../types/tt.hpp"
#include "../types/tt_core.hpp"
#include "../types/tt_matrix.hpp"
#include "tt_als.hpp"

namespace mva
{
namespace tensor_train
{
namespace detail
{

// ALS bond-compression for tt_matrix.  Same purpose as round_matrix
// but uses ALS sweeps instead of an SVD truncation pass.  Packs the
// matrix into a tt, delegates to als_round(), then unpacks.
//
// Honours opts.max_rank (0 = derived from SVD reference when eps > 0);
// opts.eps is used by als_round() for NaN guard + determining max_rank
// from an SVD reference when max_rank is unspecified.
//
// Optional warm-start: if warm_start_matrix is non-null and its
// per-site (m, n) match a, its packed tt form is used as the ALS
// initial guess.  Otherwise als_round falls back to
// round_right_canonical(packed_a, 0.0, max_rank) for cold start.
inline tt_matrix als_round_matrix(const tt_matrix& a,
                                  const als_options& opts,
                                  const tt_matrix* warm_start_matrix = nullptr,
                                  als_result* info                   = nullptr)
{
  std::vector<int> ms, ns;
  tt packed_a = detail::pack_matrix_as_tt(a, ms, ns);

  // Local copy of opts so we can override warm_start without
  // mutating the caller's struct.
  als_options local_opts = opts;

  // If a warm-start matrix was supplied and its per-site (m, n)
  // match a, pack it and point local_opts.warm_start at it.  We need
  // to keep the packed tt alive until als_round returns.
  tt packed_warm;
  const bool have_warm =
    detail::warm_start_matrix_compatible(warm_start_matrix, ms, ns);
  if (have_warm)
  {
    std::vector<int> dummy_ms, dummy_ns;
    packed_warm =
      detail::pack_matrix_as_tt(*warm_start_matrix, dummy_ms, dummy_ns);
    local_opts.warm_start = &packed_warm;
  }
  else
  {
    local_opts.warm_start = nullptr;
  }

  tt packed_x = als_round(packed_a, local_opts, info);
  return detail::unpack_tt_as_matrix(packed_x, ms, ns);
}

}  // namespace detail
}  // namespace tensor_train
}  // namespace mva
//
// :D
//
