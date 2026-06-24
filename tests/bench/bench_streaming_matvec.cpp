/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Bench: naive vs streaming matvec_round comparison
*/
#include <cstdint>
#include <cstdio>
#include <vector>

#include "tensor_train.hpp"
#include "tt_bench_utils.hpp"

namespace tt_ns = mva::tensor_train;
namespace bu    = mva::tensor_train::bench_utils;

static void
bench_case(int iters, const char* tag, int d, int m, int n, int rA, int rx)
{
  std::printf("\n=== %s  d=%d m=%d n=%d rA=%d rx=%d iters=%d ===\n",
              tag,
              d,
              m,
              n,
              rA,
              rx,
              iters);
  std::fflush(stdout);

  std::vector<int> rs(static_cast<std::size_t>(d), m);
  std::vector<int> cs(static_cast<std::size_t>(d), n);
  auto A = tt_ns::random(rs, cs, rA, 0xA1ULL);
  auto x = tt_ns::random(cs, rx, 0xB2ULL);

  tt_ns::round_options svd_opts;
  svd_opts.eps    = 1.0e-10;
  svd_opts.method = tt_ns::round_method::svd;

  tt_ns::round_options str_opts = svd_opts;
  str_opts.method               = tt_ns::round_method::svd_streaming;

  bu::run("naive matvec_round(svd)",
          iters,
          [&] { return tt_ns::matvec_round(A, x, svd_opts); });

  bu::run("streaming matvec_round(svd_streaming)",
          iters,
          [&] { return tt_ns::matvec_round(A, x, str_opts); });

  bu::run("right_orthogonalize(A)+right_orthogonalize(x)",
          iters,
          [&]
          {
            tt_ns::tt_matrix Ac = A;
            tt_ns::tt xc        = x;
            tt_ns::right_orthogonalize(Ac);
            tt_ns::right_orthogonalize(xc);
            return 0.0;
          });

  tt_ns::tt_matrix A_ro = A;
  tt_ns::tt x_ro        = x;
  tt_ns::right_orthogonalize(A_ro);
  tt_ns::right_orthogonalize(x_ro);

  bu::run("frob_norm_apply(A,x) [diagnostic]",
          iters,
          [&] { return tt_ns::frob_norm_apply(A_ro, x_ro); });

  auto y_str = tt_ns::matvec_round(A, x, str_opts);
  std::printf("  ranks(y_str)= ");
  for (int r : y_str.ranks())
    std::printf("%d ", r);
  std::printf("\n");
  std::fflush(stdout);
}

int main()
{
  constexpr int iters = 10;
  std::printf("== bench_streaming_matvec ==\n");

  bench_case(iters, "L", /*d*/ 10, 2, 2, 4, 4);  // ~0.8M
  bench_case(iters, "M", /*d*/ 8, 4, 4, 6, 6);   // ~14.9M
  bench_case(iters, "H", /*d*/ 8, 4, 4, 8, 8);   // ~83.9M

  std::printf("\nbench_streaming_matvec: done\n");
  return 0;
}
//
// :D
//
