/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Bench: tt_matrix apply paths across QTT and dense regimes
*/
#include <cstdint>
#include <cstdio>
#include <vector>

#include "tensor_train.hpp"
#include "tt_bench_utils.hpp"

namespace tt_ns = mva::tensor_train;
namespace bu    = mva::tensor_train::bench_utils;

static void bench_case(int iters, const char* tag, int d, int m, int n, int rA)
{
  std::printf(
    "\n[%s] d=%d  m=%d n=%d  rA=%d  iters=%d\n", tag, d, m, n, rA, iters);
  std::fflush(stdout);

  std::vector<int> rs(static_cast<std::size_t>(d), m);
  std::vector<int> cs(static_cast<std::size_t>(d), n);

  auto A = tt_ns::random(rs, cs, rA, 0xA0A0A0ULL);
  auto B = tt_ns::random(cs, rs, rA, 0xB0B0B0ULL);
  auto x = tt_ns::random(cs, rA, 0xC0C0C0ULL);

  tt_ns::round_options svd_opts;
  svd_opts.eps    = 1.0e-10;
  svd_opts.method = tt_ns::round_method::svd;

  bu::run("matvec(A,x)", iters, [&] { return tt_ns::matvec(A, x); });
  bu::run("matmat(A,B)", iters, [&] { return tt_ns::matmat(A, B); });
  bu::run(
    "round(A, svd, 1e-10)", iters, [&] { return tt_ns::round(A, svd_opts); });
  bu::run("frob_inner(A,A)", iters, [&] { return tt_ns::frob_inner(A, A); });
}

int main()
{
  constexpr int iters = 10;
  std::printf("== bench_tt_matrix_apply ==\n");
  std::printf("compiler: g++  flags: -O2  (see Makefile)\n");

  // QTT regime -- long chain, small physical modes.
  bench_case(iters, "QTT L", /*d*/ 12, 2, 2, /*rA*/ 8);   // ~0.2M
  bench_case(iters, "QTT M", /*d*/ 10, 2, 2, /*rA*/ 32);  // ~13.1M
  bench_case(iters, "QTT H", /*d*/ 10, 4, 4, /*rA*/ 24);  // ~22.1M

  // Dense regime -- short chain, larger physical modes.
  bench_case(iters, "DENSE L", /*d*/ 5, 16, 16, /*rA*/ 8);  // ~6.6M
  bench_case(iters, "DENSE M", /*d*/ 4, 32, 32, /*rA*/ 8);  // ~21.0M
  bench_case(iters, "DENSE H", /*d*/ 3, 64, 64, /*rA*/ 8);  // ~62.9M

  std::printf("\nbench_tt_matrix_apply: done\n");
  return 0;
}
//
// :D
//
