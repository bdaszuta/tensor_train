/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Fused matvec-then-round via two-site DMRG. Computes Y = matvec(A, x) and compresses Y using DMRG bond-compression sweeps.
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
#include "tt_dmrg.hpp"
#include "tt_round.hpp"

namespace mva
{
namespace tensor_train
{
namespace detail
{

inline tt matvec_dmrg_round(const tt_matrix& A,
                            const tt& x,
                            const dmrg_options& opts,
                            dmrg_result* info = nullptr)
{
  if (std::isnan(opts.eps) || opts.eps <= 0.0)
  {
    std::fprintf(stderr,
                 "matvec_dmrg_round: eps must be > 0 for DMRG compression "
                 "(got %.15e); eps=0 means 'no truncation' only for "
                 "the SVD path\n",
                 opts.eps);
    std::abort();
  }

  tt Y = matvec(A, x);

  const int d = Y.d();

  if (d <= 1)
  {
    tt out = round(Y, opts.eps, opts.max_rank);
    if (info)
    {
      info->iters_run    = 0;
      info->final_change = 0.0;
      info->converged    = true;
      int rmax           = 0;
      for (int r : out.ranks())
        if (r > rmax)
          rmax = r;
      info->max_bond_rank = rmax;
    }
    return out;
  }

  bool use_warm = detail::warm_start_tt_compatible(opts.warm_start, Y);

  tt X_init;
  bool init_is_right_canonical = false;
  if (use_warm)
  {
    X_init = detail::copy_tt(*opts.warm_start);
    if (opts.max_rank > 0)
    {
      for (int r : X_init.ranks())
      {
        if (r > opts.max_rank)
        {
          std::fprintf(stderr,
                       "matvec_dmrg_round: warm_start max rank %d"
                       " exceeds max_rank %d\n",
                       r,
                       opts.max_rank);
          std::abort();
        }
      }
    }
  }
  else if (opts.max_rank > 0)
  {
    std::vector<int> shape(static_cast<std::size_t>(d));
    for (int k = 0; k < d; ++k)
      shape[static_cast<std::size_t>(k)] = Y.core(k).n_phys();
    X_init = random(shape, opts.max_rank, opts.seed);
  }
  else if (opts.eps > 0.0)
  {
    X_init                  = round_right_canonical(Y, opts.eps, 0);
    init_is_right_canonical = true;
  }
  else
  {
    X_init                  = round_right_canonical(Y, 0.0, 0);
    init_is_right_canonical = true;
  }

  return detail::dmrg_engine(
    Y, std::move(X_init), opts, info, init_is_right_canonical);
}

}  // namespace detail
}  // namespace tensor_train
}  // namespace mva
//
// :D
//
