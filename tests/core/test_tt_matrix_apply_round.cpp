/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Test: matvec_round and matmat_round (canonical fused apply paths)
*/
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

  // 1) matvec_round vs matvec: dense agreement and rank reduction.
  {
    const std::vector<int> rs = { 2, 3, 2 };
    const std::vector<int> cs = { 3, 2, 3 };
    auto A                    = tt_ns::random(rs, cs, 3, 0xA1A1A1ULL);
    auto x                    = tt_ns::random(cs, 3, 0xB2B2B2ULL);

    auto y_full = tt_ns::matvec(A, x);

    tt_ns::round_options opts;
    opts.eps   = 1.0e-10;
    auto y_rnd = tt_ns::matvec_round(A, x, opts);

    auto yd_full = y_full.to_dense();
    auto yd_rnd  = y_rnd.to_dense();

    double er = tu::frob_diff(yd_full, yd_rnd) / tu::frob_norm(yd_full);
    std::printf("[mv_rnd vs mv]      rel-err = %.3e\n", er);
    if (er > 1.0e-9)
    {
      std::printf("FAIL: matvec_round disagrees with matvec\n");
      ++failed;
    }

    int r_full = max_rank(y_full.ranks());
    int r_rnd  = max_rank(y_rnd.ranks());
    std::printf("[mv_rnd]  max-rank: full=%d  rounded=%d\n", r_full, r_rnd);
    if (r_rnd > r_full)
    {
      std::printf("FAIL: rounding increased rank\n");
      ++failed;
    }
  }

  // 2) matmat_round vs matmat.
  {
    const std::vector<int> rs = { 2, 3, 2 };
    const std::vector<int> ps = { 3, 2, 3 };
    const std::vector<int> ns = { 2, 3, 2 };
    auto A                    = tt_ns::random(rs, ps, 2, 0xC3C3C3ULL);
    auto B                    = tt_ns::random(ps, ns, 2, 0xD4D4D4ULL);

    auto C_full = tt_ns::matmat(A, B);

    tt_ns::round_options opts;
    opts.eps   = 1.0e-10;
    auto C_rnd = tt_ns::matmat_round(A, B, opts);

    auto Cd_full = C_full.to_dense();
    auto Cd_rnd  = C_rnd.to_dense();

    double er = tu::frob_diff(Cd_full, Cd_rnd) / tu::frob_norm(Cd_full);
    std::printf("[mm_rnd vs mm]      rel-err = %.3e\n", er);
    if (er > 1.0e-9)
    {
      std::printf("FAIL: matmat_round disagrees with matmat\n");
      ++failed;
    }

    int r_full = max_rank(C_full.ranks());
    int r_rnd  = max_rank(C_rnd.ranks());
    std::printf("[mm_rnd]  max-rank: full=%d  rounded=%d\n", r_full, r_rnd);
    if (r_rnd > r_full)
    {
      std::printf("FAIL: rounding increased rank\n");
      ++failed;
    }
  }

  // 2b) streaming vs naive: agree at 10 * eps * ||y||.
  {
    const std::vector<int> rs = { 2, 3, 2, 3 };
    const std::vector<int> cs = { 3, 2, 3, 2 };
    auto A                    = tt_ns::random(rs, cs, 3, 0xAA11ULL);
    auto x                    = tt_ns::random(cs, 3, 0xBB22ULL);

    const double eps = 1.0e-10;

    tt_ns::round_options opt_str;
    opt_str.method = tt_ns::round_method::svd_streaming;
    opt_str.eps    = eps;
    auto y_str     = tt_ns::matvec_round(A, x, opt_str);

    tt_ns::round_options opt_nai;
    opt_nai.eps = eps;
    auto y_nai  = tt_ns::matvec_round(A, x, opt_nai);

    auto yd_str      = y_str.to_dense();
    auto yd_nai      = y_nai.to_dense();
    const double ny  = tu::frob_norm(yd_nai);
    const double err = tu::frob_diff(yd_str, yd_nai);
    std::printf(
      "[mv_rnd stream vs naive] err=%.3e tol=%.3e\n", err, 10.0 * eps * ny);
    if (err > 10.0 * eps * ny)
    {
      std::printf("FAIL: streaming matvec_round vs naive\n");
      ++failed;
    }
  }
  {
    const std::vector<int> rs = { 2, 3, 2, 3 };
    const std::vector<int> ps = { 3, 2, 3, 2 };
    const std::vector<int> ns = { 2, 3, 2, 3 };
    auto A                    = tt_ns::random(rs, ps, 2, 0xCC33ULL);
    auto B                    = tt_ns::random(ps, ns, 2, 0xDD44ULL);

    const double eps = 1.0e-10;

    tt_ns::round_options opt_str;
    opt_str.eps = eps;
    auto C_str  = tt_ns::matmat_round(A, B, opt_str);

    tt_ns::round_options opt_nai;
    opt_nai.method = tt_ns::round_method::svd_naive;
    opt_nai.eps    = eps;
    auto C_nai     = tt_ns::matmat_round(A, B, opt_nai);

    auto Cd_str      = C_str.to_dense();
    auto Cd_nai      = C_nai.to_dense();
    const double nc  = tu::frob_norm(Cd_nai);
    const double err = tu::frob_diff(Cd_str, Cd_nai);
    std::printf(
      "[mm_rnd svd vs naive] err=%.3e tol=%.3e\n", err, 10.0 * eps * nc);
    if (err > 10.0 * eps * nc)
    {
      std::printf("FAIL: svd matmat_round vs naive\n");
      ++failed;
    }
  }

  // 3) max_rank cap honoured.
  {
    const std::vector<int> rs = { 2, 3, 2, 2 };
    const std::vector<int> cs = { 3, 2, 3, 2 };
    auto A                    = tt_ns::random(rs, cs, 3, 0xE5E5E5ULL);
    auto x                    = tt_ns::random(cs, 3, 0xF6F6F6ULL);

    tt_ns::round_options opts;
    opts.eps      = 1.0e-12;
    opts.max_rank = 2;
    auto y_cap    = tt_ns::matvec_round(A, x, opts);
    int rmax      = max_rank(y_cap.ranks());
    std::printf("[mv_rnd cap=2]      max-rank=%d\n", rmax);
    if (rmax > 2)
    {
      std::printf("FAIL: max_rank cap not honoured\n");
      ++failed;
    }
  }

  if (failed == 0)
    std::printf("test_tt_matrix_apply_round: OK\n");
  return failed == 0 ? 0 : 1;
}
//
// :D
//
