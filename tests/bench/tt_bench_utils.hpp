/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tiny bench harness for tensor_train operations. Header-only, no
*/
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace mva
{
namespace tensor_train
{
namespace bench_utils
{

using clock_t = std::chrono::steady_clock;
using ns_t    = std::chrono::nanoseconds;

// Force the compiler to treat ``v`` as observed.  Defeats DCE on the
// returned values of benchmarked closures.
template <class T>
inline void do_not_optimize(const T& v)
{
  asm volatile("" : : "r,m"(v) : "memory");
}

// Run ``fn`` ``iters`` times after a small warmup; print min and median
// milliseconds per iter, plus a label.
template <class Fn>
inline void run(const char* label, int iters, Fn&& fn)
{
  // Warmup (10% of iters, at least 3, capped at 32).
  int warm = std::max(3, std::min(32, iters / 10));
  for (int i = 0; i < warm; ++i)
  {
    auto v = fn();
    do_not_optimize(v);
  }

  std::vector<long long> ns;
  ns.reserve(static_cast<std::size_t>(iters));
  for (int i = 0; i < iters; ++i)
  {
    auto t0 = clock_t::now();
    auto v  = fn();
    auto t1 = clock_t::now();
    do_not_optimize(v);
    ns.push_back(std::chrono::duration_cast<ns_t>(t1 - t0).count());
  }
  std::sort(ns.begin(), ns.end());
  long long med = ns[ns.size() / 2];
  long long mn  = ns.front();
  std::printf("  %-40s  min=%8.3f ms   med=%8.3f ms   iters=%d\n",
              label,
              mn / 1.0e6,
              med / 1.0e6,
              iters);
  std::fflush(stdout);
}

}  // namespace bench_utils
}  // namespace tensor_train
}  // namespace mva
//
// :D
//
