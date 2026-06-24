/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Test: round for tt_matrix via svd (inflate + recover), als, dmrg
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

// Inflate interior bonds of a tt_matrix by ``pad`` zero rows/cols.
static tt_ns::tt_matrix inflate_zero_pad(const tt_ns::tt_matrix& a, int pad)
{
  const int d = a.d();
  std::vector<tt_ns::tt_matrix_core> cs;
  cs.reserve(static_cast<std::size_t>(d));
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
    const tt_ns::tt_matrix_core& src = a.core(k);
    const int m                      = src.m_phys();
    const int n                      = src.n_phys();
    tt_ns::tt_matrix_core dst(rL(k), m, n, rR(k));
    dst.zero_clear();
    for (int aa = 0; aa < src.r_left(); ++aa)
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
          for (int b = 0; b < src.r_right(); ++b)
            dst(aa, i, j, b) = src(aa, i, j, b);
    cs.push_back(std::move(dst));
  }
  return tt_ns::tt_matrix(std::move(cs));
}

int main()
{
  int failed = 0;

  // ---------- round (tt_matrix) for svd / als / dmrg ----------
  {
    const std::vector<int> rs = { 2, 3, 2, 3 };
    const std::vector<int> cs = { 3, 2, 3, 2 };
    const int rA              = 3;
    auto A                    = tt_ns::random(rs, cs, rA, 0xA001ULL);
    auto Ad                   = A.to_dense();
    const double nA           = tu::frob_norm(Ad);
    const int rmax_A          = max_rank(A.ranks());

    auto noise = tt_ns::random(rs, cs, rmax_A, 0xA002ULL);
    auto B     = tt_ns::axpby(1.0, A, 1.0e-14, noise);

    // svd: inflate a lower-rank matrix then round back (stronger test).
    {
      const std::vector<int> rs_svd = { 2, 3, 2 };
      const std::vector<int> cs_svd = { 3, 2, 3 };
      auto A2   = tt_ns::random(rs_svd, cs_svd, 2, 0xA1B2C3D4ULL);
      auto A2d  = A2.to_dense();
      auto rsA2 = A2.ranks();

      auto B2  = inflate_zero_pad(A2, 3);
      auto B2d = B2.to_dense();
      if (tu::frob_diff(A2d, B2d) > 1.0e-10)
      {
        std::printf("FAIL: svd inflate changed matrix (diff=%g)\n",
                    tu::frob_diff(A2d, B2d));
        ++failed;
      }

      tt_ns::round_options o;
      o.eps     = 1.0e-10;
      auto C    = tt_ns::round(B2, o);
      auto Cd   = C.to_dense();
      double er = tu::frob_diff(A2d, Cd) / tu::frob_norm(A2d);
      std::printf("  [svd round] rel-err = %.3e\n", er);
      if (er > 1.0e-8)
      {
        std::printf("FAIL: svd round error %g\n", er);
        ++failed;
      }
      for (std::size_t k = 0; k < rsA2.size(); ++k)
      {
        if (C.ranks()[k] > rsA2[k])
        {
          std::printf(
            "FAIL: svd rank at k=%zu (%d > %d)\n", k, C.ranks()[k], rsA2[k]);
          ++failed;
        }
      }
      if (C.row_shape() != rs_svd || C.col_shape() != cs_svd)
      {
        std::printf("FAIL: svd round changed shape\n");
        ++failed;
      }

      tt_ns::round_options o_cap;
      o_cap.eps      = 1.0e-10;
      o_cap.max_rank = 1;
      auto D         = tt_ns::round(B2, o_cap);
      for (std::size_t k = 1; k + 1 < D.ranks().size(); ++k)
      {
        if (D.ranks()[k] > 1)
        {
          std::printf(
            "FAIL: svd max_rank cap at k=%zu (%d)\n", k, D.ranks()[k]);
          ++failed;
        }
      }
    }

    // als round
    {
      tt_ns::round_options opts;
      opts.method    = tt_ns::round_method::als;
      opts.max_rank  = rmax_A;
      opts.max_iters = 3;
      opts.tol       = 1.0e-12;
      opts.seed      = 0xA111ULL;
      tt_ns::round_result info;
      auto C         = tt_ns::round(B, opts, &info);
      auto Cd        = C.to_dense();
      const double e = tu::frob_diff(Cd, Ad) / (nA + 1.0e-300);
      std::printf("  [als round cold] iters=%d converged=%d rel-err=%.3e\n",
                  info.iters_run,
                  static_cast<int>(info.converged),
                  e);
      if (e > 1.0e-10)
      {
        std::printf("FAIL: als round cold error\n");
        ++failed;
      }
    }

    {
      tt_ns::round_options opts;
      opts.method               = tt_ns::round_method::als;
      opts.max_rank             = rmax_A;
      opts.max_iters            = 3;
      opts.tol                  = 1.0e-12;
      opts.warm_start_tt_matrix = &A;
      tt_ns::round_result info;
      auto C         = tt_ns::round(B, opts, &info);
      auto Cd        = C.to_dense();
      const double e = tu::frob_diff(Cd, Ad) / (nA + 1.0e-300);
      std::printf("  [als round warm] iters=%d converged=%d rel-err=%.3e\n",
                  info.iters_run,
                  static_cast<int>(info.converged),
                  e);
      if (info.iters_run > 3)
      {
        std::printf("FAIL: als round warm took >3 iters\n");
        ++failed;
      }
      if (e > 1.0e-12)
      {
        std::printf("FAIL: als round warm error\n");
        ++failed;
      }
    }

    {
      const int r_cap = 2;
      tt_ns::round_options opt_svd;
      opt_svd.max_rank     = r_cap;
      auto C_round         = tt_ns::round(A, opt_svd);
      auto Cd_round        = C_round.to_dense();
      const double e_round = tu::frob_diff(Cd_round, Ad) / (nA + 1.0e-300);
      tt_ns::round_options opts;
      opts.method    = tt_ns::round_method::als;
      opts.max_rank  = r_cap;
      opts.max_iters = 3;
      opts.tol       = 1.0e-12;
      opts.seed      = 0xA222ULL;
      tt_ns::round_result info;
      auto C_als         = tt_ns::round(A, opts, &info);
      auto Cd_als        = C_als.to_dense();
      const double e_als = tu::frob_diff(Cd_als, Ad) / (nA + 1.0e-300);
      std::printf("  [als round cap=%d] e_round=%.3e e_als=%.3e iters=%d\n",
                  r_cap,
                  e_round,
                  e_als,
                  info.iters_run);
      if (e_als > 5.0 * e_round + 1.0e-12)
      {
        std::printf("FAIL: als round tight cap > 5x round error\n");
        ++failed;
      }
    }

    // dmrg round
    {
      tt_ns::round_options opts;
      opts.method    = tt_ns::round_method::dmrg;
      opts.max_rank  = rmax_A;
      opts.eps       = 1.0e-12;
      opts.max_iters = 3;
      opts.tol       = 1.0e-12;
      tt_ns::round_result info;
      auto C         = tt_ns::round(B, opts, &info);
      auto Cd        = C.to_dense();
      const double e = tu::frob_diff(Cd, Ad) / (nA + 1.0e-300);
      int rmax_C     = max_rank(C.ranks());
      std::printf(
        "  [dmrg round cold] iters=%d converged=%d rmax=%d rel-err=%.3e\n",
        info.iters_run,
        static_cast<int>(info.converged),
        rmax_C,
        e);
      if (e > 1.0e-10)
      {
        std::printf("FAIL: dmrg round cold error\n");
        ++failed;
      }
    }

    {
      tt_ns::round_options opts;
      opts.method               = tt_ns::round_method::dmrg;
      opts.max_rank             = rmax_A;
      opts.eps                  = 1.0e-12;
      opts.max_iters            = 3;
      opts.tol                  = 1.0e-12;
      opts.warm_start_tt_matrix = &A;
      tt_ns::round_result info;
      auto C         = tt_ns::round(B, opts, &info);
      auto Cd        = C.to_dense();
      const double e = tu::frob_diff(Cd, Ad) / (nA + 1.0e-300);
      std::printf("  [dmrg round warm] iters=%d converged=%d rel-err=%.3e\n",
                  info.iters_run,
                  static_cast<int>(info.converged),
                  e);
      if (info.iters_run > 3)
      {
        std::printf("FAIL: dmrg round warm took >3 iters\n");
        ++failed;
      }
      if (e > 1.0e-12)
      {
        std::printf("FAIL: dmrg round warm error\n");
        ++failed;
      }
    }

    {
      const int r_cap = 2;
      tt_ns::round_options opt_svd;
      opt_svd.max_rank     = r_cap;
      auto C_round         = tt_ns::round(A, opt_svd);
      auto Cd_round        = C_round.to_dense();
      const double e_round = tu::frob_diff(Cd_round, Ad) / (nA + 1.0e-300);
      tt_ns::round_options opts;
      opts.method    = tt_ns::round_method::dmrg;
      opts.max_rank  = r_cap;
      opts.eps       = 1.0e-12;
      opts.max_iters = 3;
      opts.tol       = 1.0e-12;
      tt_ns::round_result info;
      auto C_dmrg         = tt_ns::round(A, opts, &info);
      auto Cd_dmrg        = C_dmrg.to_dense();
      const double e_dmrg = tu::frob_diff(Cd_dmrg, Ad) / (nA + 1.0e-300);
      std::printf("  [dmrg round cap=%d] e_round=%.3e e_dmrg=%.3e iters=%d\n",
                  r_cap,
                  e_round,
                  e_dmrg,
                  info.iters_run);
      if (e_dmrg > 5.0 * e_round + 1.0e-12)
      {
        std::printf("FAIL: dmrg round tight cap > 5x round error\n");
        ++failed;
      }
    }

    // dmrg eps-only adaptive-rank
    {
      tt_ns::round_options opts;
      opts.method    = tt_ns::round_method::dmrg;
      opts.max_rank  = 0;
      opts.eps       = 1.0e-10;
      opts.max_iters = 3;
      opts.tol       = 1.0e-12;
      tt_ns::round_result info;
      auto C         = tt_ns::round(B, opts, &info);
      auto Cd        = C.to_dense();
      const double e = tu::frob_diff(Cd, Ad) / (nA + 1.0e-300);
      int rmax_C     = max_rank(C.ranks());
      std::printf("  [dmrg round eps-only] iters=%d rmax=%d rel-err=%.3e\n",
                  info.iters_run,
                  rmax_C,
                  e);
      if (e > 1.0e-8)
      {
        std::printf("FAIL: dmrg round eps-only error\n");
        ++failed;
      }
    }
  }

  // ---------- matmat_round for als / dmrg ----------
  {
    const std::vector<int> rs = { 2, 3, 2 };
    const std::vector<int> ms = { 3, 2, 3 };
    const std::vector<int> cs = { 2, 2, 2 };
    const int rA              = 2;
    const int rB              = 2;
    auto A_mat                = tt_ns::random(rs, ms, rA, 0xB001ULL);
    auto B_mat                = tt_ns::random(ms, cs, rB, 0xB002ULL);

    auto C_full      = tt_ns::matmat(A_mat, B_mat);
    auto Cd_full     = C_full.to_dense();
    const double nC  = tu::frob_norm(Cd_full);
    const int R_full = max_rank(C_full.ranks());
    std::printf("  [matmat ref] R = %d, ||C|| = %.3e\n", R_full, nC);

    // als matmat_round: cold / warm / tight cap / warm-pert
    auto test_matmat =
      [&](tt_ns::round_method method, const char* tag, int& failed_ref)
    {
      // cold
      {
        tt_ns::round_options opts;
        opts.method    = method;
        opts.max_rank  = R_full;
        opts.eps       = 1.0e-12;
        opts.max_iters = 3;
        opts.tol       = 1.0e-12;
        opts.seed      = 0xB111ULL;
        tt_ns::round_result info;
        auto C         = tt_ns::matmat_round(A_mat, B_mat, opts, &info);
        auto Cd        = C.to_dense();
        const double e = tu::frob_diff(Cd, Cd_full) / (nC + 1.0e-300);
        int rmax_C     = max_rank(C.ranks());
        std::printf(
          "  [%s matmat cold] iters=%d converged=%d rmax=%d rel-err=%.3e\n",
          tag,
          info.iters_run,
          static_cast<int>(info.converged),
          rmax_C,
          e);
        if (e > 1.0e-10)
        {
          std::printf("FAIL: %s matmat cold error\n", tag);
          ++failed_ref;
        }
      }

      // warm
      {
        tt_ns::round_options opts;
        opts.method               = method;
        opts.max_rank             = R_full;
        opts.eps                  = 1.0e-12;
        opts.max_iters            = 3;
        opts.tol                  = 1.0e-12;
        opts.warm_start_tt_matrix = &C_full;
        tt_ns::round_result info;
        auto C         = tt_ns::matmat_round(A_mat, B_mat, opts, &info);
        auto Cd        = C.to_dense();
        const double e = tu::frob_diff(Cd, Cd_full) / (nC + 1.0e-300);
        std::printf("  [%s matmat warm] iters=%d converged=%d rel-err=%.3e\n",
                    tag,
                    info.iters_run,
                    static_cast<int>(info.converged),
                    e);
        if (e > 1.0e-12)
        {
          std::printf("FAIL: %s matmat warm error\n", tag);
          ++failed_ref;
        }
      }

      // tight cap vs svd
      {
        const int r_cap = 2;
        tt_ns::round_options opt_svd;
        opt_svd.max_rank = r_cap;
        auto C_round     = tt_ns::matmat_round(A_mat, B_mat, opt_svd);
        auto Cd_round    = C_round.to_dense();
        const double e_round =
          tu::frob_diff(Cd_round, Cd_full) / (nC + 1.0e-300);
        tt_ns::round_options opts;
        opts.method    = method;
        opts.max_rank  = r_cap;
        opts.eps       = 1.0e-12;
        opts.max_iters = 3;
        opts.tol       = 1.0e-12;
        opts.seed      = 0xB222ULL;
        tt_ns::round_result info;
        auto C_iter  = tt_ns::matmat_round(A_mat, B_mat, opts, &info);
        auto Cd_iter = C_iter.to_dense();
        const double e_iter =
          tu::frob_diff(Cd_iter, Cd_full) / (nC + 1.0e-300);
        std::printf("  [%s matmat cap=%d] e_round=%.3e e_iter=%.3e iters=%d\n",
                    tag,
                    r_cap,
                    e_round,
                    e_iter,
                    info.iters_run);
        if (e_iter > 5.0 * e_round + 1.0e-12)
        {
          std::printf("FAIL: %s matmat tight cap > 5x round error\n", tag);
          ++failed_ref;
        }
      }

      // warm-pert
      {
        auto Anoise       = tt_ns::random(rs, ms, rA, 0xB101ULL);
        auto Bnoise       = tt_ns::random(ms, cs, rB, 0xB102ULL);
        auto Ap           = tt_ns::axpby(1.0, A_mat, 1.0e-3, Anoise);
        auto Bp           = tt_ns::axpby(1.0, B_mat, 1.0e-3, Bnoise);
        auto Cp_full      = tt_ns::matmat(Ap, Bp);
        auto Cpd_full     = Cp_full.to_dense();
        const double nCp  = tu::frob_norm(Cpd_full);
        const int Rp_full = max_rank(Cp_full.ranks());

        tt_ns::round_options opt_warm;
        opt_warm.max_rank = Rp_full;
        auto C_warm       = tt_ns::round(C_full, opt_warm);

        tt_ns::round_options opts;
        opts.method               = method;
        opts.max_rank             = Rp_full;
        opts.eps                  = 1.0e-12;
        opts.max_iters            = 3;
        opts.tol                  = 1.0e-12;
        opts.warm_start_tt_matrix = &C_warm;
        tt_ns::round_result info;
        auto Cp        = tt_ns::matmat_round(Ap, Bp, opts, &info);
        auto Cpd       = Cp.to_dense();
        const double e = tu::frob_diff(Cpd, Cpd_full) / (nCp + 1.0e-300);
        std::printf(
          "  [%s matmat warm-pert] iters=%d converged=%d rel-err=%.3e\n",
          tag,
          info.iters_run,
          static_cast<int>(info.converged),
          e);
        if (e > 1.0e-10)
        {
          std::printf("FAIL: %s matmat warm-pert error\n", tag);
          ++failed_ref;
        }
      }
    };
    test_matmat(tt_ns::round_method::als, "als", failed);
    test_matmat(tt_ns::round_method::dmrg, "dmrg", failed);
  }

  if (failed == 0)
    std::printf("test_tt_matrix_round: OK\n");
  return failed == 0 ? 0 : 1;
}
//
// :D
//
