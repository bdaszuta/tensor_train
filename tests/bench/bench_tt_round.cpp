/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Bench: tt::round on rank-multiplied-shaped tensors
*/
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

#include "tensor_train.hpp"
#include "tt_bench_utils.hpp"

namespace tt_ns  = mva::tensor_train;
namespace tt_dtl = mva::tensor_train::detail;
namespace bu     = mva::tensor_train::bench_utils;

static tt_ns::tt svd_sweep_only(const tt_ns::tt& base, double delta, int mr)
{
  tt_ns::tt t = base;
  auto& cs    = t.cores();
  const int d = t.d();
  for (int k = 0; k < d - 1; ++k)
  {
    tt_dtl::svd_step(cs[k], cs[k + 1], delta, mr);
  }
  return t;
}

static void bench_case(int iters, const char* tag, int d, int n, int r)
{
  std::printf("\n[%s] d=%d  n=%d  r=%d  iters=%d\n", tag, d, n, r, iters);
  std::fflush(stdout);

  std::vector<int> shape(static_cast<std::size_t>(d), n);
  auto t = tt_ns::random(shape, r, 0xD0D0D0ULL);

  tt_ns::round_options o0;
  o0.eps      = 1.0e-10;
  o0.max_rank = 0;
  bu::run(
    "round(eps=1e-10, mr=0)", iters, [&] { return tt_ns::round(t, o0); });

  tt_ns::round_options ohalf = o0;
  ohalf.max_rank             = r / 2 > 0 ? r / 2 : 1;
  bu::run(
    "round(eps=1e-10, mr=r/2)", iters, [&] { return tt_ns::round(t, ohalf); });

  tt_ns::round_options ofull = o0;
  ofull.max_rank             = r;
  bu::run(
    "round(eps=1e-10, mr=r)", iters, [&] { return tt_ns::round(t, ofull); });

  bu::run("right_orthogonalize_only",
          iters,
          [&]
          {
            tt_ns::tt c = t;
            tt_ns::right_orthogonalize(c);
            return c;
          });

  tt_ns::tt base_ro = t;
  tt_ns::right_orthogonalize(base_ro);
  double sum_sq = 0.0;
  {
    const auto& c0  = base_ro.core(0);
    const double* p = c0.data();
    const int sz    = c0.size();
    for (int i = 0; i < sz; ++i)
      sum_sq += p[i] * p[i];
  }
  const double norm = std::sqrt(sum_sq);
  const double delta =
    1.0e-10 * norm / std::sqrt(static_cast<double>(d - 1 > 0 ? d - 1 : 1));

  bu::run("svd_sweep_only(eps=1e-10, mr=0)",
          iters,
          [&] { return svd_sweep_only(base_ro, delta, 0); });
  bu::run("svd_sweep_only(eps=1e-10, mr=r/2)",
          iters,
          [&]
          { return svd_sweep_only(base_ro, delta, r / 2 > 0 ? r / 2 : 1); });
}

int main()
{
  constexpr int iters = 10;
  std::printf("== bench_tt_round ==\n");
  std::printf("compiler: g++  flags: -O2  (see Makefile)\n");
  std::fflush(stdout);

  bench_case(iters, "L", /*d*/ 12, /*n*/ 2, /*r*/ 16);  // ~1.0M
  bench_case(iters, "M", /*d*/ 10, /*n*/ 4, /*r*/ 36);  // ~18.7M
  bench_case(iters, "H", /*d*/ 8, /*n*/ 16, /*r*/ 40);  // ~81.9M

  std::printf("\nbench_tt_round: done\n");
  std::fflush(stdout);
  return 0;
}
//
// :D
//
