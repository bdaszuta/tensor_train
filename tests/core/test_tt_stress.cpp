/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: d=50 identity TT (all mode sizes 1)
*/
#include <cmath>
#include <cstdio>
#include <vector>

#include "tensor_train.hpp"
#include "tt_test_utils.hpp"

namespace tt_ns = mva::tensor_train;
namespace tu    = mva::tensor_train::test_utils;

int main()
{
  int failed = 0;

  // d=50 identity TT (all mode sizes 1)
  {
    std::vector<int> shape(50, 1);
    auto a   = tt_ns::ones(shape);
    double n = tt_ns::norm(a);
    std::printf("  [d=50 identity] norm=%.6f (expect 1.0)\n", n);
    if (std::abs(n - 1.0) > 1.0e-14)
    {
      std::printf("  FAIL\n");
      ++failed;
    }
    auto rks = a.ranks();
    bool ranks_ok = true;
    for (int r : rks)
      if (r != 1)
        ranks_ok = false;
    std::printf("  [d=50 identity] ranks: ");
    if (ranks_ok)
      std::printf("OK\n");
    else
    {
      std::printf("FAIL (expected all 1s)\n");
      ++failed;
    }
  }

  // Rapid rank-growth: add two random TTs then round
  {
    std::vector<int> shape = {4, 4, 4, 4, 4};
    auto a = tt_ns::random(shape, 6, 42);
    auto b = tt_ns::random(shape, 6, 99);
    auto s = tt_ns::add(a, b);
    tt_ns::round_options ropts;
    ropts.eps      = 1.0e-8;
    ropts.max_rank = 32;
    ropts.method   = tt_ns::round_method::svd;
    auto c         = tt_ns::round(s, ropts);
    double nc      = tt_ns::norm(c);
    std::printf("  [add+round] norm=%.6e (expect > 0)\n", nc);
    if (nc <= 0.0)
    {
      std::printf("  FAIL: zero norm\n");
      ++failed;
    }
    // Verify correctness: round(add(a,b)) == a_dense + b_dense
    auto da = a.to_dense();
    auto db = b.to_dense();
    std::vector<double> ref(da.size());
    for (std::size_t i = 0; i < da.size(); ++i)
      ref[i] = da[i] + db[i];
    double err = tu::frob_diff(c.to_dense(), ref) / tu::frob_norm(ref);
    std::printf("  [add+round] rel-err = %.3e (expect < 1e-5)\n", err);
    if (err > 1.0e-5)
    {
      std::printf("  FAIL: add+round incorrect\n");
      ++failed;
    }
  }

  // Iterated round should be idempotent
  {
    std::vector<int> shape = {3, 3, 3, 3};
    auto a = tt_ns::random(shape, 8, 777);
    tt_ns::round_options ropts;
    ropts.eps    = 1.0e-10;
    ropts.method = tt_ns::round_method::svd;
    auto r1      = tt_ns::round(a, ropts);
    auto r2      = tt_ns::round(r1, ropts);
    auto d1      = r1.to_dense();
    auto d2      = r2.to_dense();
    double diff  = tu::frob_diff(d1, d2);
    double nrm   = tu::frob_norm(d1);
    double rel   = (nrm > 0.0) ? diff / nrm : diff;
    std::printf("  [round idempotent] rel-diff = %.3e\n", rel);
    if (rel > 1.0e-12)
    {
      std::printf("  FAIL\n");
      ++failed;
    }
  }

  // High-rank: compress rank-12 TT with tight eps
  {
    std::vector<int> shape = { 8, 8, 8, 8 };
    auto a = tt_ns::random(shape, 12, 555);
    tt_ns::round_options ropts;
    ropts.eps      = 1.0e-8;
    ropts.max_rank = 64;
    ropts.method   = tt_ns::round_method::svd;
    auto b         = tt_ns::round(a, ropts);
    double nb      = tt_ns::norm(b);
    std::printf("  [high-rank %d->?] norm=%.6e (expect > 0)\n",
                a.max_rank(), nb);
    if (nb <= 0.0) { std::printf("  FAIL: zero norm\n"); ++failed; }
    // Verify round idempotence: round(round(a)) == round(a).
    auto b2  = tt_ns::round(b, ropts);
    double r = tu::frob_diff(b.to_dense(), b2.to_dense());
    double n = tu::frob_norm(b.to_dense());
    double rel = (n > 0.0) ? r / n : r;
    std::printf("  [high-rank idemp] rel-diff = %.3e\n", rel);
    if (rel > 1.0e-10) { std::printf("  FAIL: not idempotent\n"); ++failed; }
  }

  // Large d: 20-mode random TT
  {
    std::vector<int> shape(20, 2);
    auto a = tt_ns::random(shape, 4, 0xABCDu);
    double na = tt_ns::norm(a);
    std::printf("  [d=20 rank-4] norm=%.6e (expect > 0)\n", na);
    if (na <= 0.0) { std::printf("  FAIL\n"); ++failed; }
    auto rks = a.ranks();
    if (rks.front() != 1 || rks.back() != 1) {
      std::printf("  [d=20] boundary rank FAIL\n"); ++failed;
    }
  }

  if (failed == 0)
    std::printf("\ntest_tt_stress: OK\n");
  else
    std::printf("\ntest_tt_stress: %d FAILED\n", failed);

  return failed;
}
//
// :D
//
