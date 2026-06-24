/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Fused matmat + ALS round. Computes C = matmat(A, B) and compresses C using ALS sweeps. Mirrors matvec_als_round but for the matrix-matrix product.
*/
#pragma once

#include <utility>
#include <vector>

#include "../detail/matrix_pack.hpp"
#include "../detail/warm_start.hpp"
#include "../types/tt.hpp"
#include "tt_als.hpp"
#include "../types/tt_core.hpp"
#include "../factory/tt_factory.hpp"
#include "../types/tt_matrix.hpp"
#include "../apply/tt_matrix_apply.hpp"
#include "tt_matrix_round.hpp"

namespace mva
{
namespace tensor_train
{
namespace detail
{

// C ~ round_matrix(A * B, eps, max_rank), via ALS.  If
// warm_start_matrix is non-null and shape-compatible with the
// product, its packed tt form is used as the ALS initial guess;
// otherwise a random tt of opts.max_rank seeded by opts.seed is used
// which avoids the rank-R SVD that the SVD-based path would
// require.  Honours opts.max_rank (0 = derived from SVD reference
// when eps > 0); opts.eps is used by als_round() for NaN guard +
// SVD-ref rank estimate (als_engine itself never reads opts.eps).
inline tt_matrix matmat_als_round(const tt_matrix& a,
                                  const tt_matrix& b,
                                  const als_options& opts,
                                  const tt_matrix* warm_start_matrix = nullptr,
                                  als_result* info                   = nullptr)
{
  if (std::isnan(opts.eps))
  {
    std::fprintf(stderr,
                 "matmat_als_round: eps is NaN; "
                 "must be a valid number\n");
    std::abort();
  }

  // Form the full rank-multiplied product.
  tt_matrix c_full = matmat(a, b);

  // Trivial case: degenerate d -- no bonds to compress.
  if (c_full.d() <= 1)
  {
    if (info)
    {
      info->iters_run    = 0;
      info->final_change = 0.0;
      info->converged    = true;
    }
    return round_matrix(c_full, opts.eps, opts.max_rank);
  }

  // Build effective options.  When max_rank is not specified, derive
  // the target rank from an SVD reference (matrix round) at tolerance eps.
  // Pack the SVD reference as a tt for warm-start if caller did not
  // provide one.
  tt_matrix svd_ref_m;  // lives for duration; packed as svd_ref_tt
  tt        svd_ref_tt;
  als_options effective_opts = opts;
  if (effective_opts.max_rank <= 0)
  {
    if (effective_opts.eps <= 0.0)
    {
      std::fprintf(stderr,
                   "matmat_als_round: max_rank <= 0 and eps <= 0; "
                   "ALS requires either max_rank > 0 or eps > 0 "
                   "to determine compression target\n");
      std::abort();
    }
    svd_ref_m   = round_matrix(c_full, effective_opts.eps, 0);
    effective_opts.max_rank = svd_ref_m.max_rank();
    // Pack SVD reference as tt for warm-start (only if caller did not
    // already provide a warm_start_matrix).
    if (warm_start_matrix == nullptr)
    {
      std::vector<int> dummy_ms, dummy_ns;
      svd_ref_tt = detail::pack_matrix_as_tt(svd_ref_m, dummy_ms, dummy_ns);
      effective_opts.warm_start = &svd_ref_tt;
    }
  }

  // Pack c_full as a tt (deep copy).
  std::vector<int> ms, ns;
  tt packed_c = detail::pack_matrix_as_tt(c_full, ms, ns);

  const int d = packed_c.d();
  std::vector<int> shape(static_cast<std::size_t>(d));
  for (int k = 0; k < d; ++k)
  {
    shape[static_cast<std::size_t>(k)] = packed_c.core(k).n_phys();
  }

  // Build the ALS initial guess.  Warm-start path: validate shape
  // (per-site m and n) against c_full.  Cold-start: random tt of
  // effective_opts.max_rank; honours effective_opts.seed.
  tt x_init;
  const bool used_warm =
    detail::warm_start_matrix_compatible(warm_start_matrix, ms, ns);
  if (used_warm)
  {
    for (int r : warm_start_matrix->ranks())
    {
      if (r > effective_opts.max_rank)
      {
        std::fprintf(stderr,
                     "matmat_als_round: warm_start max rank %d exceeds "
                     "max_rank %d\n",
                     r, effective_opts.max_rank);
        std::abort();
      }
    }
  }
  if (used_warm)
  {
    std::vector<int> dummy_ms, dummy_ns;
    x_init = detail::pack_matrix_as_tt(*warm_start_matrix, dummy_ms, dummy_ns);
  }
  else if (effective_opts.warm_start != nullptr)
  {
    // SVD-derived warm-start (tt format, already packed above).
    x_init = detail::copy_tt(*effective_opts.warm_start);
  }
  else
  {
    x_init = random(shape, effective_opts.max_rank, effective_opts.seed);
  }

  tt packed_x = detail::als_engine(packed_c, std::move(x_init),
                                   effective_opts, info);
  return detail::unpack_tt_as_matrix(packed_x, ms, ns);
}

}  // namespace detail
}  // namespace tensor_train
}  // namespace mva
//
// :D
//
