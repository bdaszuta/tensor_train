/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Test: round for tt via svd, als, dmrg (round_method). Inflates a
*/
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "tensor_train.hpp"
#include "tt_test_utils.hpp"

namespace tt_ns = mva::tensor_train;
namespace tu    = mva::tensor_train::test_utils;

static tt_ns::tt inflate_zero_pad(const tt_ns::tt& a, int pad)
{
  const int d = a.d();
  std::vector<tt_ns::tt_core> cs;
  cs.reserve(d);
  auto rL = [&](int k)
  {
    if (k == 0)
      return 1;
    return a.core(k).r_left() + pad;
  };
  auto rR = [&](int k)
  {
    if (k == d - 1)
      return 1;
    return a.core(k).r_right() + pad;
  };
  for (int k = 0; k < d; ++k)
  {
    const tt_ns::tt_core& src = a.core(k);
    const int n               = src.n_phys();
    tt_ns::tt_core dst(rL(k), n, rR(k));
    dst.zero_clear();
    for (int i = 0; i < src.r_left(); ++i)
      for (int j = 0; j < n; ++j)
        for (int p = 0; p < src.r_right(); ++p)
          dst(i, j, p) = src(i, j, p);
    cs.push_back(std::move(dst));
  }
  return tt_ns::tt(std::move(cs));
}

int main()
{
  int failed = 0;

  const int n0 = 4, n1 = 5, n2 = 6, n3 = 4;
  std::vector<double> T(n0 * n1 * n2 * n3);
  std::mt19937_64 rng(13);
  std::normal_distribution<double> N(0.0, 1.0);
  for (auto& x : T)
    x = N(rng);

  auto A     = tt_ns::from_dense(T.data(), { n0, n1, n2, n3 }, 1.0e-12);
  auto Ad    = A.to_dense();
  auto rsA   = A.ranks();
  int rmax_A = 0;
  for (int r : rsA)
    if (r > rmax_A)
      rmax_A = r;

  auto B  = inflate_zero_pad(A, 5);
  auto Bd = B.to_dense();
  if (tu::frob_diff(Ad, Bd) > 1.0e-10)
  {
    std::printf("FAIL: inflate changed tensor (diff=%g)\n",
                tu::frob_diff(Ad, Bd));
    ++failed;
  }

  // ---- svd: basic round + max_rank cap ----
  {
    tt_ns::round_options o;
    o.eps     = 1.0e-10;
    auto C    = tt_ns::round(B, o);
    auto Cd   = C.to_dense();
    double er = tu::frob_diff(Ad, Cd) / tu::frob_norm(Ad);
    std::printf("  [svd] rel-err = %.3e\n", er);
    if (er > 1.0e-8)
    {
      std::printf("FAIL: svd round error too large\n");
      ++failed;
    }
    for (std::size_t k = 0; k < rsA.size(); ++k)
    {
      if (C.ranks()[k] > rsA[k])
      {
        std::printf(
          "FAIL: svd rank at k=%zu (%d > %d)\n", k, C.ranks()[k], rsA[k]);
        ++failed;
      }
    }

    tt_ns::round_options o_cap;
    o_cap.eps      = 1.0e-10;
    o_cap.max_rank = 2;
    auto D         = tt_ns::round(B, o_cap);
    for (std::size_t k = 1; k + 1 < D.ranks().size(); ++k)
    {
      if (D.ranks()[k] > 2)
      {
        std::printf("FAIL: svd max_rank cap at k=%zu (%d)\n", k, D.ranks()[k]);
        ++failed;
      }
    }
  }

  // ---- als: cold, warm, vs-svd, warm-pert ----
  auto test_als = [&](int& failed_ref)
  {
    tt_ns::round_options opt;
    opt.method    = tt_ns::round_method::als;
    opt.max_rank  = rmax_A;
    opt.max_iters = 3;
    opt.tol       = 1.0e-12;
    tt_ns::round_result info;
    auto C          = tt_ns::round(B, opt, &info);
    auto Cd         = C.to_dense();
    const double er = tu::frob_diff(Ad, Cd) / tu::frob_norm(Ad);
    std::printf("  [als cold] iters=%d converged=%d rel-err=%.3e\n",
                info.iters_run,
                info.converged ? 1 : 0,
                er);
    if (er > 1.0e-10)
    {
      std::printf("FAIL: als cold error\n");
      ++failed_ref;
    }
    for (std::size_t k = 1; k + 1 < C.ranks().size(); ++k)
    {
      if (C.ranks()[k] > opt.max_rank)
      {
        std::printf("FAIL: als cold rank cap at k=%zu\n", k);
        ++failed_ref;
      }
    }
  };
  test_als(failed);

  {
    tt_ns::round_options opt;
    opt.method        = tt_ns::round_method::als;
    opt.max_rank      = rmax_A;
    opt.max_iters     = 3;
    opt.tol           = 1.0e-12;
    opt.warm_start_tt = &A;
    tt_ns::round_result info;
    auto C          = tt_ns::round(B, opt, &info);
    auto Cd         = C.to_dense();
    const double er = tu::frob_diff(Ad, Cd) / tu::frob_norm(Ad);
    std::printf("  [als warm] iters=%d converged=%d rel-err=%.3e\n",
                info.iters_run,
                info.converged ? 1 : 0,
                er);
    if (er > 1.0e-10)
    {
      std::printf("FAIL: als warm error\n");
      ++failed;
    }
  }

  {
    const int cap = rmax_A;
    tt_ns::round_options opt_svd;
    opt_svd.eps      = 1.0e-10;
    opt_svd.max_rank = cap;
    auto C_round     = tt_ns::round(B, opt_svd);
    tt_ns::round_options opt;
    opt.method    = tt_ns::round_method::als;
    opt.max_rank  = cap;
    opt.max_iters = 3;
    opt.tol       = 1.0e-13;
    auto C_als    = tt_ns::round(B, opt);
    const double e_round =
      tu::frob_diff(Ad, C_round.to_dense()) / tu::frob_norm(Ad);
    const double e_als =
      tu::frob_diff(Ad, C_als.to_dense()) / tu::frob_norm(Ad);
    std::printf("  [als vs svd] e_round=%.3e e_als=%.3e (cap=%d)\n",
                e_round,
                e_als,
                cap);
    if (e_als > 10.0 * std::max(e_round, 1.0e-13))
    {
      std::printf("FAIL: als error vs svd\n");
      ++failed;
    }
  }

  {
    auto Anoise = tt_ns::random({ n0, n1, n2, n3 }, 4, 99);
    auto Bp     = tt_ns::axpby(1.0, A, 1.0e-3, Anoise);
    tt_ns::round_options opt;
    opt.method        = tt_ns::round_method::als;
    opt.max_rank      = rmax_A;
    opt.max_iters     = 3;
    opt.tol           = 1.0e-12;
    opt.warm_start_tt = &A;
    tt_ns::round_result info;
    auto C           = tt_ns::round(Bp, opt, &info);
    auto Cd          = C.to_dense();
    auto Bpd         = Bp.to_dense();
    const double rel = tu::frob_diff(Bpd, Cd) / tu::frob_norm(Bpd);
    std::printf("  [als warm-pert] iters=%d converged=%d rel-err=%.3e\n",
                info.iters_run,
                info.converged ? 1 : 0,
                rel);
    if (info.iters_run > 5)
    {
      std::printf("FAIL: als warm-pert iters (%d)\n", info.iters_run);
      ++failed;
    }
    if (rel > 1.0e-2)
    {
      std::printf("FAIL: als warm-pert error (%.3e)\n", rel);
      ++failed;
    }
  }

  // ---- dmrg: cold, warm, vs-svd, warm-pert, eps-only ----
  {
    tt_ns::round_options opt;
    opt.method    = tt_ns::round_method::dmrg;
    opt.max_rank  = rmax_A;
    opt.eps       = 1.0e-12;
    opt.max_iters = 3;
    opt.tol       = 1.0e-12;
    tt_ns::round_result info;
    auto C          = tt_ns::round(B, opt, &info);
    auto Cd         = C.to_dense();
    const double er = tu::frob_diff(Ad, Cd) / tu::frob_norm(Ad);
    int rmax_C      = 0;
    for (int r : C.ranks())
      if (r > rmax_C)
        rmax_C = r;
    std::printf("  [dmrg cold] iters=%d converged=%d rmax=%d rel-err=%.3e\n",
                info.iters_run,
                info.converged ? 1 : 0,
                rmax_C,
                er);
    if (er > 1.0e-10)
    {
      std::printf("FAIL: dmrg cold error\n");
      ++failed;
    }
    for (std::size_t k = 1; k + 1 < C.ranks().size(); ++k)
    {
      if (C.ranks()[k] > opt.max_rank)
      {
        std::printf("FAIL: dmrg cold rank cap at k=%zu\n", k);
        ++failed;
      }
    }
  }

  {
    tt_ns::round_options opt;
    opt.method        = tt_ns::round_method::dmrg;
    opt.max_rank      = rmax_A;
    opt.eps           = 1.0e-12;
    opt.max_iters     = 3;
    opt.tol           = 1.0e-12;
    opt.warm_start_tt = &A;
    tt_ns::round_result info;
    auto C          = tt_ns::round(B, opt, &info);
    auto Cd         = C.to_dense();
    const double er = tu::frob_diff(Ad, Cd) / tu::frob_norm(Ad);
    int rmax_C      = 0;
    for (int r : C.ranks())
      if (r > rmax_C)
        rmax_C = r;
    std::printf("  [dmrg warm] iters=%d converged=%d rmax=%d rel-err=%.3e\n",
                info.iters_run,
                info.converged ? 1 : 0,
                rmax_C,
                er);
    if (er > 1.0e-10)
    {
      std::printf("FAIL: dmrg warm error\n");
      ++failed;
    }
  }

  {
    const int cap = rmax_A;
    tt_ns::round_options opt_svd;
    opt_svd.eps      = 1.0e-10;
    opt_svd.max_rank = cap;
    auto C_round     = tt_ns::round(B, opt_svd);
    tt_ns::round_options opt;
    opt.method    = tt_ns::round_method::dmrg;
    opt.max_rank  = cap;
    opt.eps       = 1.0e-12;
    opt.max_iters = 3;
    opt.tol       = 1.0e-13;
    auto C_dmrg   = tt_ns::round(B, opt);
    const double e_round =
      tu::frob_diff(Ad, C_round.to_dense()) / tu::frob_norm(Ad);
    const double e_dmrg =
      tu::frob_diff(Ad, C_dmrg.to_dense()) / tu::frob_norm(Ad);
    std::printf("  [dmrg vs svd] e_round=%.3e e_dmrg=%.3e (cap=%d)\n",
                e_round,
                e_dmrg,
                cap);
    if (e_dmrg > 1.1 * e_round + 1.0e-12)
    {
      std::printf("FAIL: dmrg error vs svd\n");
      ++failed;
    }
  }

  {
    auto Anoise = tt_ns::random({ n0, n1, n2, n3 }, 4, 99);
    auto Bp     = tt_ns::axpby(1.0, A, 1.0e-3, Anoise);
    tt_ns::round_options opt;
    opt.method        = tt_ns::round_method::dmrg;
    opt.max_rank      = rmax_A;
    opt.eps           = 1.0e-12;
    opt.max_iters     = 3;
    opt.tol           = 1.0e-12;
    opt.warm_start_tt = &A;
    tt_ns::round_result info;
    auto C           = tt_ns::round(Bp, opt, &info);
    auto Cd          = C.to_dense();
    auto Bpd         = Bp.to_dense();
    const double rel = tu::frob_diff(Bpd, Cd) / tu::frob_norm(Bpd);
    std::printf("  [dmrg warm-pert] iters=%d converged=%d rel-err=%.3e\n",
                info.iters_run,
                info.converged ? 1 : 0,
                rel);
    if (info.iters_run > 5)
    {
      std::printf("FAIL: dmrg warm-pert iters (%d)\n", info.iters_run);
      ++failed;
    }
    if (rel > 1.0e-2)
    {
      std::printf("FAIL: dmrg warm-pert error (%.3e)\n", rel);
      ++failed;
    }
  }

  // dmrg eps-only adaptive-rank
  {
    tt_ns::round_options opt;
    opt.method    = tt_ns::round_method::dmrg;
    opt.max_rank  = 0;
    opt.eps       = 1.0e-10;
    opt.max_iters = 3;
    opt.tol       = 1.0e-12;
    tt_ns::round_result info;
    auto C          = tt_ns::round(B, opt, &info);
    auto Cd         = C.to_dense();
    const double er = tu::frob_diff(Ad, Cd) / tu::frob_norm(Ad);
    int rmax_C      = 0;
    for (int r : C.ranks())
      if (r > rmax_C)
        rmax_C = r;
    std::printf(
      "  [dmrg eps-only] iters=%d converged=%d rmax=%d rel-err=%.3e\n",
      info.iters_run,
      info.converged ? 1 : 0,
      rmax_C,
      er);
    if (er > 1.0e-8)
    {
      std::printf("FAIL: dmrg eps-only error\n");
      ++failed;
    }
    int rmax_B = 0;
    for (int r : B.ranks())
      if (r > rmax_B)
        rmax_B = r;
    if (rmax_C > rmax_B)
    {
      std::printf("FAIL: dmrg eps-only rank %d > %d\n", rmax_C, rmax_B);
      ++failed;
    }
  }

  if (failed == 0)
    std::printf("test_tt_round: OK\n");
  return failed == 0 ? 0 : 1;
}
//
// :D
//
