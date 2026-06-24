/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Fused matmat + DMRG round. Computes C = matmat(A, B) and compresses C using two-site DMRG sweeps. Mirrors matmat_als_round but uses DMRG.
*/
#pragma once

#include <utility>
#include <vector>

#include "../detail/matrix_pack.hpp"
#include "../detail/warm_start.hpp"
#include "../types/tt.hpp"
#include "../types/tt_core.hpp"
#include "tt_dmrg.hpp"
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

inline tt_matrix matmat_dmrg_round(
  const tt_matrix& a,
  const tt_matrix& b,
  const dmrg_options& opts,
  const tt_matrix* warm_start_matrix = nullptr,
  dmrg_result* info                  = nullptr)
{
  if (std::isnan(opts.eps) || opts.eps <= 0.0)
  {
    std::fprintf(stderr,
                 "matmat_dmrg_round: eps must be > 0 for DMRG compression "
                 "(got %.15e); eps=0 means 'no truncation' only for "
                 "the SVD path\n",
                 opts.eps);
    std::abort();
  }

  tt_matrix c_full = matmat(a, b);

  if (c_full.d() <= 1)
  {
    tt_matrix out = round_matrix(c_full, opts.eps, opts.max_rank);
    if (info)
    {
      info->iters_run     = 0;
      info->final_change  = 0.0;
      info->converged     = true;
      int rmax = 0;
      for (int r : out.ranks())
        if (r > rmax) rmax = r;
      info->max_bond_rank = rmax;
    }
    return out;
  }

  std::vector<int> ms, ns;
  tt packed_c = detail::pack_matrix_as_tt(c_full, ms, ns);

  const int d = packed_c.d();
  std::vector<int> shape(static_cast<std::size_t>(d));
  for (int k = 0; k < d; ++k)
  {
    shape[static_cast<std::size_t>(k)] = packed_c.core(k).n_phys();
  }

  // Build initial guess.
  tt x_init;
  bool init_is_right_canonical = false;
  const bool used_warm =
    detail::warm_start_matrix_compatible(warm_start_matrix, ms, ns);
  if (used_warm)
  {
    std::vector<int> dummy_ms, dummy_ns;
    x_init = detail::pack_matrix_as_tt(*warm_start_matrix, dummy_ms, dummy_ns);
    if (opts.max_rank > 0)
    {
      for (int r : x_init.ranks())
      {
        if (r > opts.max_rank)
        {
          std::fprintf(stderr,
                       "matmat_dmrg_round: warm_start max rank %d"
                       " exceeds max_rank %d\n",
                       r,
                       opts.max_rank);
          std::abort();
        }
      }
    }
  }
  else
  {
    if (opts.max_rank > 0)
    {
      x_init = random(shape, opts.max_rank, opts.seed);
    }
    else if (opts.eps > 0.0)
    {
      x_init                  = round_right_canonical(packed_c, opts.eps, 0);
      init_is_right_canonical = true;
    }
    else
    {
      x_init                  = round_right_canonical(packed_c, 0.0, 0);
      init_is_right_canonical = true;
    }
  }

  tt packed_x = detail::dmrg_engine(
    packed_c, std::move(x_init), opts, info, init_is_right_canonical);
  return detail::unpack_tt_as_matrix(packed_x, ms, ns);
}

}  // namespace detail
}  // namespace tensor_train
}  // namespace mva
//
// :D
//
