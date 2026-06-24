/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Fused apply + round: canonical SVD and naive paths for matvec_round and matmat_round in the unified dispatching framework.
*/
#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "../apply/tt_matrix_apply.hpp"
#include "../detail/streaming_apply.hpp"
#include "../types/tt.hpp"
#include "../types/tt_matrix.hpp"
#include "tt_matrix_round.hpp"
#include "tt_round.hpp"

namespace mva
{
namespace tensor_train
{
namespace detail
{

// y = round(A * x, eps).  Naive path: full apply + round.  Default
// matvec_round dispatches here (see matvec_round_streaming for the
// streaming variant).
inline tt matvec_round(const tt_matrix& a,
                       const tt& x,
                       double eps   = 1.0e-10,
                       int max_rank = 0)
{
  return round(matvec(a, x), eps, max_rank);
}

// y ~ round(A * x, eps).  Streaming: never materialise the full
// rank-multiplied apply.  d=1 falls back to the naive path (no
// truncation between cores).  Use when A and x are known to compress
// meaningfully under eps; otherwise prefer matvec_round (naive).
inline tt matvec_round_streaming(const tt_matrix& a,
                                 const tt& x,
                                 double eps   = 1.0e-10,
                                 int max_rank = 0)
{
  if (a.d() != x.d())
  {
    std::fprintf(stderr,
                 "matvec_round_streaming: core count mismatch %d vs %d\n",
                 a.d(),
                 x.d());
    std::abort();
  }
  if (a.d() <= 1)
  {
    return matvec_round(a, x, eps, max_rank);
  }
  return detail::build_streaming_matvec(a, x, eps, max_rank);
}

// C = round(A * B, eps).  Naive path: full apply + round
// (svd_naive in the unified API).
inline tt_matrix matmat_round_naive(const tt_matrix& a,
                                    const tt_matrix& b,
                                    double eps   = 1.0e-10,
                                    int max_rank = 0)
{
  return round_matrix(matmat(a, b), eps, max_rank);
}

// C ~ round_matrix(A * B, eps).  Streaming variant; default for
// matmat_round.  d=1 falls back to the naive path.
inline tt_matrix matmat_round(const tt_matrix& a,
                              const tt_matrix& b,
                              double eps   = 1.0e-10,
                              int max_rank = 0)
{
  if (a.d() != b.d())
  {
    std::fprintf(
      stderr, "matmat_round: core count mismatch %d vs %d\n", a.d(), b.d());
    std::abort();
  }
  if (a.d() <= 1)
  {
    return matmat_round_naive(a, b, eps, max_rank);
  }
  return detail::build_streaming_matmat(a, b, eps, max_rank);
}

}  // namespace detail
}  // namespace tensor_train
}  // namespace mva
//
// :D
//
