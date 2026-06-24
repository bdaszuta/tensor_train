/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Test: matvec_round for als, dmrg. 4-site random tt_matrix * tt;
*/
#include <cstdint>
#include <cstdio>
#include <vector>

#include "tensor_train.hpp"
#include "tt_test_utils.hpp"

namespace tt_ns = mva::tensor_train;
namespace tu    = mva::tensor_train::test_utils;

static int max_rank(const std::vector<int>& r)
{
  int m = 0;
  for (int v : r)
    if (v > m)
      m = v;
  return m;
}

int main()
{
  int failed = 0;

  const std::vector<int> rs = { 3, 2, 3, 2 };
  const std::vector<int> cs = { 2, 3, 2, 3 };
  const int rA              = 3;
  const int rx              = 3;
  auto A                    = tt_ns::random(rs, cs, rA, 0xA111ULL);
  auto x                    = tt_ns::random(cs, rx, 0xB222ULL);

  auto y_full      = tt_ns::matvec(A, x);
  auto yd_full     = y_full.to_dense();
  const double ny  = tu::frob_norm(yd_full);
  const int R_full = max_rank(y_full.ranks());
  std::printf("[mv ref] R = %d, ||y|| = %.3e\n", R_full, ny);

  auto test_mv =
    [&](tt_ns::round_method method, const char* tag, int& failed_ref)
  {
    // cold at full rank cap
    {
      tt_ns::round_options opts;
      opts.method    = method;
      opts.max_rank  = R_full;
      opts.eps       = 1.0e-12;
      opts.max_iters = 3;
      opts.tol       = 1.0e-12;
      opts.seed      = 0xC0FFEEULL;
      tt_ns::round_result info;
      auto y         = tt_ns::matvec_round(A, x, opts, &info);
      auto yd        = y.to_dense();
      const double e = tu::frob_diff(yd, yd_full) / (ny + 1.0e-300);
      const int rmax = max_rank(y.ranks());
      std::printf(
        "  [%s mv cold] iters=%d converged=%d rmax=%d rel-err=%.3e\n",
        tag,
        info.iters_run,
        static_cast<int>(info.converged),
        rmax,
        e);
      if (e > 1.0e-10)
      {
        std::printf("FAIL: %s mv cold error\n", tag);
        ++failed_ref;
      }
    }

    // warm-start with exact y
    {
      tt_ns::round_options opts;
      opts.method        = method;
      opts.max_rank      = R_full;
      opts.eps           = 1.0e-12;
      opts.max_iters     = 3;
      opts.tol           = 1.0e-12;
      opts.warm_start_tt = &y_full;
      tt_ns::round_result info;
      auto y         = tt_ns::matvec_round(A, x, opts, &info);
      auto yd        = y.to_dense();
      const double e = tu::frob_diff(yd, yd_full) / (ny + 1.0e-300);
      std::printf("  [%s mv warm] iters=%d converged=%d rel-err=%.3e\n",
                  tag,
                  info.iters_run,
                  static_cast<int>(info.converged),
                  e);
      if (info.iters_run > 3)
      {
        std::printf("FAIL: %s mv warm took >3 iters\n", tag);
        ++failed_ref;
      }
      if (e > 1.0e-12)
      {
        std::printf("FAIL: %s mv warm error\n", tag);
        ++failed_ref;
      }
    }

    // warm-start with perturbed guess
    {
      auto noise = tt_ns::random(rs, R_full, 0xD333ULL);
      auto warm  = tt_ns::axpby(1.0, y_full, 1.0e-3, noise);
      tt_ns::round_options opt_warm;
      opt_warm.max_rank = R_full;
      auto warm_r       = tt_ns::round(warm, opt_warm);

      tt_ns::round_options opts;
      opts.method        = method;
      opts.max_rank      = R_full;
      opts.eps           = 1.0e-12;
      opts.max_iters     = 3;
      opts.tol           = 1.0e-12;
      opts.warm_start_tt = &warm_r;
      tt_ns::round_result info;
      auto y         = tt_ns::matvec_round(A, x, opts, &info);
      auto yd        = y.to_dense();
      const double e = tu::frob_diff(yd, yd_full) / (ny + 1.0e-300);
      std::printf("  [%s mv warm-pert] iters=%d converged=%d rel-err=%.3e\n",
                  tag,
                  info.iters_run,
                  static_cast<int>(info.converged),
                  e);
      if (e > 1.0e-10)
      {
        std::printf("FAIL: %s mv warm-pert error\n", tag);
        ++failed_ref;
      }
    }

    // tight rank cap vs matvec_round svd
    {
      const int r_cap = 4;
      tt_ns::round_options opt_svd;
      opt_svd.max_rank = r_cap;
      auto y_round     = tt_ns::matvec_round(A, x, opt_svd);
      auto yd_round    = y_round.to_dense();
      const double e_round =
        tu::frob_diff(yd_round, yd_full) / (ny + 1.0e-300);

      tt_ns::round_options opts;
      opts.method    = method;
      opts.max_rank  = r_cap;
      opts.eps       = 1.0e-12;
      opts.max_iters = 3;
      opts.tol       = 1.0e-12;
      opts.seed      = 0xE444ULL;
      tt_ns::round_result info;
      auto y_iter         = tt_ns::matvec_round(A, x, opts, &info);
      auto yd_iter        = y_iter.to_dense();
      const double e_iter = tu::frob_diff(yd_iter, yd_full) / (ny + 1.0e-300);
      std::printf(
        "  [%s mv cap=%d] e_round=%.3e e_iter=%.3e iters=%d converged=%d\n",
        tag,
        r_cap,
        e_round,
        e_iter,
        info.iters_run,
        static_cast<int>(info.converged));
      if (e_iter > 5.0 * e_round + 1.0e-12)
      {
        std::printf("FAIL: %s mv tight cap > 5x round error\n", tag);
        ++failed_ref;
      }
    }
  };

  test_mv(tt_ns::round_method::als, "als", failed);
  test_mv(tt_ns::round_method::dmrg, "dmrg", failed);

  if (failed == 0)
    std::printf("test_tt_matvec_round: OK\n");
  return failed == 0 ? 0 : 1;
}
//
// :D
//
