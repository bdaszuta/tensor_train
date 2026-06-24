/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Shared helpers for cross-approximation engines
*/
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "truncate.hpp"
#include "../types/tt_eigen_bridge.hpp"

namespace mva
{
namespace tensor_train
{
namespace detail
{

// Rank truncation from singular values (used by all cross engines).
inline int cross_truncate_rank(const eigen_bridge::col_vector& s,
                               double delta,
                               int max_rank,
                               int rmin = 1)
{
  int r = truncate_eps_rank(s, delta, max_rank);
  if (r < rmin)
    r = rmin;
  return r;
}

// Evaluate the two-site supercore M_{a,i,ip,b} = func(idx) on the
// current left/right index sets L_idx[k], R_idx[k+1].
template <typename F>
inline void eval_supercore_cross(F& func,
                                 int k,
                                 int n_k,
                                 int n_kp1,
                                 const std::vector<std::vector<int>>& L_idx,
                                 const std::vector<std::vector<int>>& R_idx,
                                 int d,
                                 double* buf,
                                 int rL,
                                 int rR)
{
  if (k + 1 >= d)
  {
    std::fprintf(stderr,
                 "eval_supercore_cross: k=%d >= d-1=%d\n",
                 k, d - 1);
    std::abort();
  }
  // Allocate the multi-index once; reused across all quadruples.
  std::vector<int> idx(static_cast<std::size_t>(d));
  for (int a = 0; a < rL; ++a)
  {
    for (int i = 0; i < n_k; ++i)
    {
      for (int ip = 0; ip < n_kp1; ++ip)
      {
        for (int b = 0; b < rR; ++b)
        {
          if (k > 0)
          {
            const auto& Lk = L_idx[static_cast<std::size_t>(k)];
            for (int m = 0; m < k; ++m)
              idx[static_cast<std::size_t>(m)] =
                Lk[static_cast<std::size_t>(
                  static_cast<std::size_t>(a) * k + m)];
          }
          idx[static_cast<std::size_t>(k)]     = i;
          idx[static_cast<std::size_t>(k + 1)] = ip;
          if (k + 2 < d)
          {
            const auto& Rkp      = R_idx[static_cast<std::size_t>(k + 1)];
            const int suffix_len = d - k - 2;
            for (int m = 0; m < suffix_len; ++m)
              idx[static_cast<std::size_t>(k + 2 + m)] =
                Rkp[static_cast<std::size_t>(
                  static_cast<std::size_t>(b) * suffix_len + m)];
          }
          const std::size_t off =
            static_cast<std::size_t>(a) * n_k * n_kp1 * rR +
            static_cast<std::size_t>(i) * n_kp1 * rR +
            static_cast<std::size_t>(ip) * rR +
            b;
          buf[off] =
            func(idx.data());
        }
      }
    }
  }
}

}  // namespace detail
}  // namespace tensor_train
}  // namespace mva
//
// :D
//
