/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Bench: SVD vs ALS (one-site) vs DMRG (two-site) across standalone
*/
#include <cstdint>
#include <cstdio>
#include <vector>

#include "tensor_train.hpp"
#include "tt_bench_utils.hpp"

namespace tt_ns = mva::tensor_train;
namespace bu    = mva::tensor_train::bench_utils;

// ---- panel helpers --------------------------------------------------

static void bench_standalone(int iters,
                             const char* tag,
                             int d,
                             int n,
                             int R,
                             int r)
{
  std::printf("\n[%s] STANDALONE  d=%d n=%d R=%d r=%d iters=%d\n",
              tag,
              d,
              n,
              R,
              r,
              iters);
  std::fflush(stdout);

  std::vector<int> shape(static_cast<std::size_t>(d), n);
  auto Y = tt_ns::random(shape, R, 0xA15A11ULL);

  tt_ns::round_options svd_opts;
  svd_opts.max_rank = r;
  svd_opts.method   = tt_ns::round_method::svd;
  bu::run(
    "round(svd, mr=r)", iters, [&] { return tt_ns::round(Y, svd_opts); });

  tt_ns::round_options als_opts;
  als_opts.method    = tt_ns::round_method::als;
  als_opts.max_rank  = r;
  als_opts.max_iters = 8;
  als_opts.tol       = 1.0e-10;
  als_opts.seed      = 0xBEEFULL;
  bu::run("round(als, cold, max_iters=8)",
          iters,
          [&] { return tt_ns::round(Y, als_opts); });

  tt_ns::round_options dmrg_opts;
  dmrg_opts.method    = tt_ns::round_method::dmrg;
  dmrg_opts.max_rank  = r;
  dmrg_opts.eps       = 0.0;
  dmrg_opts.max_iters = 8;
  dmrg_opts.tol       = 1.0e-10;
  bu::run("round(dmrg, cold, max_iters=8)",
          iters,
          [&] { return tt_ns::round(Y, dmrg_opts); });

  auto Y_warm                   = tt_ns::round(Y, svd_opts);
  tt_ns::round_options als_warm = als_opts;
  als_warm.warm_start_tt        = &Y_warm;
  als_warm.max_iters            = 4;
  bu::run("round(als, warm, max_iters=4)",
          iters,
          [&] { return tt_ns::round(Y, als_warm); });

  tt_ns::round_options dmrg_warm = dmrg_opts;
  dmrg_warm.warm_start_tt        = &Y_warm;
  dmrg_warm.max_iters            = 4;
  bu::run("round(dmrg, warm, max_iters=4)",
          iters,
          [&] { return tt_ns::round(Y, dmrg_warm); });
}

static void
bench_matvec(int iters, const char* tag, int d, int n, int rA, int rx, int r)
{
  const int R = rA * rx;
  std::printf("\n[%s] MATVEC  d=%d n=%d rA=%d rx=%d R=%d r=%d iters=%d\n",
              tag,
              d,
              n,
              rA,
              rx,
              R,
              r,
              iters);
  std::fflush(stdout);

  std::vector<int> rs(static_cast<std::size_t>(d), n);
  std::vector<int> cs(static_cast<std::size_t>(d), n);
  auto A = tt_ns::random(rs, cs, rA, 0x1010ULL);
  auto x = tt_ns::random(cs, rx, 0x2020ULL);

  tt_ns::round_options svd_opts;
  svd_opts.max_rank = r;
  svd_opts.method   = tt_ns::round_method::svd;

  bu::run("matvec + round(svd, mr=r)",
          iters,
          [&]
          {
            auto y = tt_ns::matvec(A, x);
            return tt_ns::round(y, svd_opts);
          });

  bu::run("matvec_round(svd, mr=r)",
          iters,
          [&] { return tt_ns::matvec_round(A, x, svd_opts); });

  tt_ns::round_options als_opts;
  als_opts.method    = tt_ns::round_method::als;
  als_opts.max_rank  = r;
  als_opts.max_iters = 8;
  als_opts.tol       = 1.0e-10;
  als_opts.seed      = 0x3030ULL;
  bu::run("matvec_round(als, cold, max_iters=8)",
          iters,
          [&] { return tt_ns::matvec_round(A, x, als_opts); });

  tt_ns::round_options dmrg_opts;
  dmrg_opts.method    = tt_ns::round_method::dmrg;
  dmrg_opts.max_rank  = r;
  dmrg_opts.eps       = 0.0;
  dmrg_opts.max_iters = 8;
  dmrg_opts.tol       = 1.0e-10;
  bu::run("matvec_round(dmrg, cold, max_iters=8)",
          iters,
          [&] { return tt_ns::matvec_round(A, x, dmrg_opts); });

  auto y_warm                   = tt_ns::matvec_round(A, x, svd_opts);
  tt_ns::round_options als_warm = als_opts;
  als_warm.warm_start_tt        = &y_warm;
  als_warm.max_iters            = 4;
  bu::run("matvec_round(als, warm, max_iters=4)",
          iters,
          [&] { return tt_ns::matvec_round(A, x, als_warm); });

  tt_ns::round_options dmrg_warm = dmrg_opts;
  dmrg_warm.warm_start_tt        = &y_warm;
  dmrg_warm.max_iters            = 4;
  bu::run("matvec_round(dmrg, warm, max_iters=4)",
          iters,
          [&] { return tt_ns::matvec_round(A, x, dmrg_warm); });
}

static void
bench_matmat(int iters, const char* tag, int d, int n, int rA, int rB, int r)
{
  const int R = rA * rB;
  std::printf("\n[%s] MATMAT  d=%d n=%d rA=%d rB=%d R=%d r=%d iters=%d\n",
              tag,
              d,
              n,
              rA,
              rB,
              R,
              r,
              iters);
  std::fflush(stdout);

  std::vector<int> rs(static_cast<std::size_t>(d), n);
  std::vector<int> cs(static_cast<std::size_t>(d), n);
  auto A = tt_ns::random(rs, cs, rA, 0x4040ULL);
  auto B = tt_ns::random(rs, cs, rB, 0x5050ULL);

  tt_ns::round_options svd_opts;
  svd_opts.max_rank = r;
  svd_opts.method   = tt_ns::round_method::svd;
  bu::run("matmat_round(svd, mr=r)",
          iters,
          [&] { return tt_ns::matmat_round(A, B, svd_opts); });

  tt_ns::round_options als_opts;
  als_opts.method    = tt_ns::round_method::als;
  als_opts.max_rank  = r;
  als_opts.max_iters = 8;
  als_opts.tol       = 1.0e-10;
  als_opts.seed      = 0x6060ULL;
  bu::run("matmat_round(als, cold, max_iters=8)",
          iters,
          [&] { return tt_ns::matmat_round(A, B, als_opts); });

  tt_ns::round_options dmrg_opts;
  dmrg_opts.method    = tt_ns::round_method::dmrg;
  dmrg_opts.max_rank  = r;
  dmrg_opts.eps       = 0.0;
  dmrg_opts.max_iters = 8;
  dmrg_opts.tol       = 1.0e-10;
  bu::run("matmat_round(dmrg, cold, max_iters=8)",
          iters,
          [&] { return tt_ns::matmat_round(A, B, dmrg_opts); });

  auto C_warm                   = tt_ns::matmat_round(A, B, svd_opts);
  tt_ns::round_options als_warm = als_opts;
  als_warm.warm_start_tt_matrix = &C_warm;
  als_warm.max_iters            = 4;
  bu::run("matmat_round(als, warm, max_iters=4)",
          iters,
          [&] { return tt_ns::matmat_round(A, B, als_warm); });

  tt_ns::round_options dmrg_warm = dmrg_opts;
  dmrg_warm.warm_start_tt_matrix = &C_warm;
  dmrg_warm.max_iters            = 4;
  bu::run("matmat_round(dmrg, warm, max_iters=4)",
          iters,
          [&] { return tt_ns::matmat_round(A, B, dmrg_warm); });
}

// ---- main -----------------------------------------------------------

int main()
{
  constexpr int iters = 10;
  std::printf("== bench_tt_round_methods ==\n");
  std::printf("compiler: g++  flags: -O2  (see Makefile)\n");
  std::fflush(stdout);

  // STANDALONE.
  bench_standalone(iters, "L", /*d*/ 8, /*n*/ 4, /*R*/ 16, /*r*/ 8);  // ~1.3M
  bench_standalone(
    iters, "M", /*d*/ 8, /*n*/ 4, /*R*/ 40, /*r*/ 20);  // ~20.5M
  bench_standalone(
    iters, "H", /*d*/ 10, /*n*/ 4, /*R*/ 56, /*r*/ 28);  // ~70.2M

  // MATVEC.
  bench_matvec(iters, "L", /*d*/ 10, 2, 4, 4, 6);  // ~0.8M
  bench_matvec(iters, "M", /*d*/ 8, 4, 6, 6, 9);   // ~14.9M
  bench_matvec(iters, "H", /*d*/ 8, 4, 8, 8, 12);  // ~83.9M

  // MATMAT.
  bench_matmat(iters, "L", /*d*/ 8, 2, 4, 4, 6);   // ~1.3M
  bench_matmat(iters, "M", /*d*/ 8, 2, 6, 6, 9);   // ~14.9M
  bench_matmat(iters, "H", /*d*/ 8, 2, 8, 8, 10);  // ~83.9M

  std::printf("\nbench_tt_round_methods: done\n");
  std::fflush(stdout);
  return 0;
}
//
// :D
//
