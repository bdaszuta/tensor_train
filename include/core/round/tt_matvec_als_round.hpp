/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Fused matvec-then-round via ALS. Computes Y = matvec(A, x) and compresses Y to a lower-rank TT using ALS sweeps.
*/
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "../apply/tt_matrix_apply.hpp"
#include "../detail/warm_start.hpp"
#include "../factory/tt_factory.hpp"
#include "../types/tt.hpp"
#include "../types/tt_matrix.hpp"
#include "tt_als.hpp"
#include "tt_round.hpp"

namespace mva
{
namespace tensor_train
{
namespace detail
{

inline tt matvec_als_round(const tt_matrix& A,
                           const tt& x,
                           const als_options& opts,
                           als_result* info = nullptr)
{
  if (std::isnan(opts.eps))
  {
    std::fprintf(stderr,
                 "matvec_als_round: eps is NaN; "
                 "must be a valid number\n");
    std::abort();
  }

  // Form the rank-(r_A * r_x) product Y = A x.  Storage but no SVD.
  tt Y = matvec(A, x);

  const int d = Y.d();

  // Trivial case: degenerate d -- no bonds to compress.
  if (d <= 1)
  {
    tt out = round(Y, opts.eps, opts.max_rank);
    if (info)
    {
      info->iters_run    = 0;
      info->final_change = 0.0;
      info->converged    = true;
    }
    return out;
  }

  // Build effective options.  When max_rank is not specified, derive
  // the target rank from an SVD reference TT at tolerance eps.
  tt svd_ref;  // lives for duration; pointed to by warm_start
  als_options effective_opts = opts;
  if (effective_opts.max_rank <= 0)
  {
    if (effective_opts.eps <= 0.0)
    {
      std::fprintf(stderr,
                   "matvec_als_round: max_rank <= 0 and eps <= 0; "
                   "ALS requires either max_rank > 0 or eps > 0 "
                   "to determine compression target\n");
      std::abort();
    }
    svd_ref = round(Y, effective_opts.eps, 0);
    effective_opts.max_rank = svd_ref.max_rank();
    if (effective_opts.warm_start == nullptr)
      effective_opts.warm_start = &svd_ref;
  }

  // Warm-start path: validate and call engine.
  bool use_warm = detail::warm_start_tt_compatible(effective_opts.warm_start,
                                                    Y);
  if (use_warm)
  {
    for (int r : effective_opts.warm_start->ranks())
    {
      if (r > effective_opts.max_rank)
      {
        std::fprintf(stderr,
                     "matvec_als_round: warm_start max rank %d exceeds "
                     "max_rank %d\n",
                     r, effective_opts.max_rank);
        std::abort();
      }
    }
  }

  tt X_init;
  if (use_warm)
  {
    X_init = detail::copy_tt(*effective_opts.warm_start);
  }
  else
  {
    // Cold start: random rank-max_rank TT.  Avoids round(Y, ...) which
    // would re-introduce the rank-R SVD we are trying to skip.
    std::vector<int> shape(static_cast<std::size_t>(d));
    for (int k = 0; k < d; ++k)
      shape[static_cast<std::size_t>(k)] = Y.core(k).n_phys();
    X_init = random(shape, effective_opts.max_rank, effective_opts.seed);
  }

  return detail::als_engine(Y, std::move(X_init), effective_opts, info);
}

}  // namespace detail
}  // namespace tensor_train
}  // namespace mva
//
// :D
//
