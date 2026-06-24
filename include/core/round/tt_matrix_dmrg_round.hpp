/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: DMRG-based TT-matrix rounding. Packs the matrix into a tt, delegates to dmrg_round(), then unpacks.
*/
#pragma once

#include <utility>
#include <vector>

#include "../detail/matrix_pack.hpp"
#include "../detail/warm_start.hpp"
#include "../types/tt.hpp"
#include "../types/tt_core.hpp"
#include "tt_dmrg.hpp"
#include "../types/tt_matrix.hpp"

namespace mva
{
namespace tensor_train
{
namespace detail
{

// DMRG bond-compression for tt_matrix.  Same purpose as round_matrix
// but driven by DMRG sweeps.  Packs the matrix into a tt, delegates
// to dmrg_round(), then unpacks.  opts.eps must be > 0 (enforced by
// dmrg_round).  warm_start_matrix, if compatible, is packed and passed
// as the DMRG initial guess.
inline tt_matrix dmrg_round_matrix(
  const tt_matrix& a,
  const dmrg_options& opts,
  const tt_matrix* warm_start_matrix = nullptr,
  dmrg_result* info                  = nullptr)
{
  std::vector<int> ms, ns;
  tt packed_a = detail::pack_matrix_as_tt(a, ms, ns);

  dmrg_options local_opts = opts;

  tt packed_warm;
  const bool have_warm =
    detail::warm_start_matrix_compatible(warm_start_matrix, ms, ns);
  if (have_warm)
  {
    std::vector<int> dummy_ms, dummy_ns;
    packed_warm =
      detail::pack_matrix_as_tt(*warm_start_matrix, dummy_ms, dummy_ns);
  }
  local_opts.warm_start = have_warm ? &packed_warm : nullptr;

  tt packed_x = dmrg_round(packed_a, local_opts, info);
  return detail::unpack_tt_as_matrix(packed_x, ms, ns);
}

}  // namespace detail
}  // namespace tensor_train
}  // namespace mva
//
// :D
//
