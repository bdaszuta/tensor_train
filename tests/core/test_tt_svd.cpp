/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Test: from_dense reconstruction error and rank-1 / rank-2 sanity
*/
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "tensor_train.hpp"
#include "tt_test_utils.hpp"

namespace tt_ns = mva::tensor_train;
namespace tu    = mva::tensor_train::test_utils;

int main()
{
  int failed = 0;

  // ---- Test 1: rank-1 outer product is recovered exactly with rank 1.
  // T[i, j, k] = u[i] * v[j] * w[k].
  {
    const int n0 = 3, n1 = 4, n2 = 5;
    std::vector<double> u = { 1.0, 2.0, 3.0 };
    std::vector<double> v = { 1.0, -1.0, 0.5, 0.25 };
    std::vector<double> w = { 2.0, 0.5, -1.0, 1.0, 3.0 };
    std::vector<double> T(n0 * n1 * n2);
    for (int i = 0; i < n0; ++i)
      for (int j = 0; j < n1; ++j)
        for (int k = 0; k < n2; ++k)
          T[(i * n1 + j) * n2 + k] = u[i] * v[j] * w[k];

    auto tt_rep = tt_ns::from_dense(T.data(), { n0, n1, n2 }, 1.0e-12);
    auto rs     = tt_rep.ranks();
    if (rs.size() != 4 || rs[0] != 1 || rs[3] != 1)
    {
      std::printf("FAIL: rank-1 boundary ranks\n");
      ++failed;
    }
    if (rs[1] != 1 || rs[2] != 1)
    {
      std::printf(
        "FAIL: rank-1 product not recovered (got %d %d)\n", rs[1], rs[2]);
      ++failed;
    }
    auto Tre  = tt_rep.to_dense();
    double er = tu::frob_diff(T, Tre) / tu::frob_norm(T);
    if (er > 1.0e-10)
    {
      std::printf("FAIL: rank-1 reconstruction err = %g\n", er);
      ++failed;
    }
  }

  // ---- Test 2: random tensor, eps=1e-8, error within 10x of eps.
  {
    const int n0 = 4, n1 = 5, n2 = 6;
    std::vector<double> T(n0 * n1 * n2);
    std::mt19937_64 rng(42);
    std::normal_distribution<double> N(0.0, 1.0);
    for (auto& x : T)
      x = N(rng);

    const double eps = 1.0e-8;
    auto tt_rep      = tt_ns::from_dense(T.data(), { n0, n1, n2 }, eps);
    auto Tre         = tt_rep.to_dense();
    double er        = tu::frob_diff(T, Tre) / tu::frob_norm(T);
    if (er > 10.0 * eps)
    {
      std::printf("FAIL: random eps=%g got err=%g\n", eps, er);
      ++failed;
    }
  }

  // ---- Test 3: zero tensor -> rank-1 zeros.
  {
    const int n0 = 2, n1 = 3, n2 = 4;
    std::vector<double> T(n0 * n1 * n2, 0.0);
    auto tt_rep = tt_ns::from_dense(T.data(), { n0, n1, n2 }, 1.0e-10);
    auto rs     = tt_rep.ranks();
    bool ok     = rs.size() == 4;
    for (int r : rs)
      ok = ok && (r == 1);
    if (!ok)
    {
      std::printf("FAIL: zero tensor ranks\n");
      ++failed;
    }
    auto Tre = tt_rep.to_dense();
    for (double v : Tre)
    {
      if (std::fabs(v) > 1e-300)
      {
        std::printf("FAIL: zero recon nonzero %g\n", v);
        ++failed;
        break;
      }
    }
  }

  if (failed == 0)
    std::printf("test_tt_svd: OK\n");
  return failed == 0 ? 0 : 1;
}
//
// :D
//
