/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: TT-matrix rounding. Delegates to round(tt, eps) via pack/unpack: packs (rL, m, n, rR) cores as (rL, m*n, rR) tt cores via memcpy, calls round(), then unpacks. Same Oseledets Alg.2 algorithm as round(tt, eps) applied to the packed representation.
*/
#pragma once

#include <utility>
#include <vector>

#include "../detail/matrix_pack.hpp"
#include "../types/tt.hpp"
#include "../types/tt_core.hpp"
#include "../types/tt_matrix.hpp"
#include "tt_round.hpp"

namespace mva
{
namespace tensor_train
{
namespace detail
{

inline tt_matrix round_matrix(tt_matrix&& a,
                              double eps   = 1.0e-10,
                              int max_rank = 0)
{
  std::vector<int> ms, ns;
  tt packed = detail::pack_matrix_as_tt(a, ms, ns);
  // Drop the source matrix: its data has been copied into ``packed``.
  (void)a;

  // Reuse the 3-axis rounding algorithm via the rvalue overload.
  tt rounded = round(std::move(packed), eps, max_rank);

  return detail::unpack_tt_as_matrix(rounded, ms, ns);
}

inline tt_matrix round_matrix(const tt_matrix& a,
                              double eps   = 1.0e-10,
                              int max_rank = 0)
{
  // Deep-copy then defer to the rvalue overload.
  std::vector<tt_matrix_core> cores(a.cores().begin(), a.cores().end());
  return round_matrix(tt_matrix(std::move(cores)), eps, max_rank);
}

// Right-canonical-output variant: packs, round_right_canonical, unpacks.
inline tt_matrix round_matrix_right_canonical(tt_matrix&& a,
                                              double eps   = 1.0e-10,
                                              int max_rank = 0)
{
  std::vector<int> ms, ns;
  tt packed = detail::pack_matrix_as_tt(a, ms, ns);
  (void)a;

  tt rounded = round_right_canonical(std::move(packed), eps, max_rank);
  return detail::unpack_tt_as_matrix(rounded, ms, ns);
}

inline tt_matrix round_matrix_right_canonical(const tt_matrix& a,
                                              double eps   = 1.0e-10,
                                              int max_rank = 0)
{
  std::vector<tt_matrix_core> cores(a.cores().begin(), a.cores().end());
  return round_matrix_right_canonical(tt_matrix(std::move(cores)),
                                      eps, max_rank);
}

}  // namespace detail
}  // namespace tensor_train
}  // namespace mva
//
// :D
//
