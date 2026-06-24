/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Factory routines for constructing tt instances without SVD compression (zeros, ones, canonical_unit, random)
*/
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

#include "../types/tt.hpp"
#include "../types/tt_core.hpp"

namespace mva
{
namespace tensor_train
{
/**
 * @brief All-zero rank-1 TT with given mode sizes.
 * Each core is shape (1, n_k, 1) filled with zeros.
 * @param shape Mode sizes n_0, ..., n_{d-1}.
 * @return Zero TT.
 */

inline tt zeros(const std::vector<int>& shape)
{
  const int d = static_cast<int>(shape.size());
  for (int k = 0; k < d; ++k)
  {
    if (shape[k] <= 0)
    {
      std::fprintf(stderr, "zeros: mode size %d is not positive\n",
                   shape[k]);
      std::abort();
    }
  }
  std::vector<tt_core> cs;
  cs.reserve(static_cast<std::size_t>(d));
  for (int k = 0; k < d; ++k)
  {
    tt_core c(1, shape[k], 1);
    c.zero_clear();
    cs.push_back(std::move(c));
  }
  return tt(std::move(cs));
}
/**
 * @brief All-one rank-1 TT (outer product of ones vectors).
 * Each core (1, n_k, 1) has every entry = 1.
 * @param shape Mode sizes.
 * @return Constant-1 TT.
 */

inline tt ones(const std::vector<int>& shape)
{
  const int d = static_cast<int>(shape.size());
  for (int k = 0; k < d; ++k)
  {
    if (shape[k] <= 0)
    {
      std::fprintf(stderr, "ones: mode size %d is not positive\n",
                   shape[k]);
      std::abort();
    }
  }
  std::vector<tt_core> cs;
  cs.reserve(static_cast<std::size_t>(d));
  for (int k = 0; k < d; ++k)
  {
    tt_core c(1, shape[k], 1);
    for (int n = 0; n < shape[k]; ++n)
      c(0, n, 0) = 1.0;
    cs.push_back(std::move(c));
  }
  return tt(std::move(cs));
}

/**
 * @brief Canonical basis TT with a single 1 at multi-index idx.
 * Each core k is a (1, n_k, 1) matrix with a 1 at position idx[k].
 * @param shape Mode sizes.
 * @param idx   Multi-index (length d, 0 <= idx[k] < n_k).
 * @return Basis TT with exactly one non-zero entry.
 */
inline tt canonical_unit(const std::vector<int>& shape,
                         const std::vector<int>& idx)
{
  const int d = static_cast<int>(shape.size());
  for (int k = 0; k < d; ++k)
  {
    if (shape[k] <= 0)
    {
      std::fprintf(stderr,
                   "canonical_unit: mode size %d is not positive\n",
                   shape[k]);
      std::abort();
    }
  }
  if (static_cast<int>(idx.size()) != d)
  {
    std::fprintf(stderr,
                 "canonical_unit: idx.size()=%zu != shape.size()=%d\n",
                 idx.size(), d);
    std::abort();
  }
  std::vector<tt_core> cs;
  cs.reserve(static_cast<std::size_t>(d));
  for (int k = 0; k < d; ++k)
  {
    if (idx[k] < 0 || idx[k] >= shape[k])
    {
      std::fprintf(stderr,
                   "canonical_unit: idx[%d]=%d out of range [0, %d)\n",
                   k, idx[k], shape[k]);
      std::abort();
    }
    tt_core c(1, shape[k], 1);
    c.zero_clear();
    c(0, idx[k], 0) = 1.0;
    cs.push_back(std::move(c));
  }
  return tt(std::move(cs));
}

/**
 * @brief Random TT with N(0,1) entries.
 * Cores have shape (r_k, n_k, r_{k+1}) with r_k = min(max_rank,
 * feasible upper bound) and entries drawn from N(0,1).
 * @param shape    Mode sizes.
 * @param max_rank Cap on per-bond rank.
 * @param seed     RNG seed for determinism.
 * @return Random TT with r_0 = r_d = 1.
 */
inline tt random(const std::vector<int>& shape,
                 int max_rank,
                 std::uint64_t seed)
{
  const int d = static_cast<int>(shape.size());
  for (int k = 0; k < d; ++k)
  {
    if (shape[k] <= 0)
    {
      std::fprintf(stderr, "random: mode size %d is not positive\n",
                   shape[k]);
      std::abort();
    }
  }
  std::vector<int> ranks(static_cast<std::size_t>(d + 1), 1);
  // Forward pass: r_{k+1} <= r_k * n_k.
  for (int k = 0; k < d; ++k)
  {
    long long cap = static_cast<long long>(ranks[k]) * shape[k];
    if (cap > max_rank && max_rank > 0)
      cap = max_rank;
    if (cap > std::numeric_limits<int>::max())
    {
      std::fprintf(stderr,
                   "random: rank cap %lld exceeds INT_MAX at bond %d; "
                   "provide a finite max_rank\n",
                   static_cast<long long>(cap), k + 1);
      std::abort();
    }
    ranks[k + 1] = static_cast<int>(cap);
  }
  // Backward pass: r_k <= r_{k+1} * n_k.
  for (int k = d - 1; k >= 0; --k)
  {
    long long cap = static_cast<long long>(ranks[k + 1]) * shape[k];
    if (cap > max_rank && max_rank > 0)
      cap = max_rank;
    if (cap > std::numeric_limits<int>::max())
    {
      std::fprintf(stderr,
                   "random: rank cap %lld exceeds INT_MAX at bond %d; "
                   "provide a finite max_rank\n",
                   static_cast<long long>(cap), k);
      std::abort();
    }
    if (cap < ranks[k])
      ranks[k] = static_cast<int>(cap);
  }
  ranks[0] = 1;
  ranks[d] = 1;

  std::mt19937_64 rng(seed);
  std::normal_distribution<double> nd(0.0, 1.0);

  std::vector<tt_core> cs;
  cs.reserve(static_cast<std::size_t>(d));
  for (int k = 0; k < d; ++k)
  {
    tt_core c(ranks[k], shape[k], ranks[k + 1]);
    double* p    = c.data();
    const int sz = c.size();
    for (int i = 0; i < sz; ++i)
      p[i] = nd(rng);
    cs.push_back(std::move(c));
  }
  return tt(std::move(cs));
}

}  // namespace tensor_train
}  // namespace mva
//
// :D
//
