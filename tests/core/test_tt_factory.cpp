/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Test: tt_factory routines (zeros, ones, canonical_unit, random)
*/
#include <cstdio>
#include <vector>

#include "tensor_train.hpp"
#include "tt_test_utils.hpp"

namespace tt_ns = mva::tensor_train;
namespace tu    = mva::tensor_train::test_utils;

int main()
{
  bool ok          = true;
  const double tol = 1e-12;

  const std::vector<std::vector<int>> shapes = {
    { 5 }, { 3, 4 }, { 3, 4, 2 }, { 2, 3, 2, 3 }
  };

  for (const auto& shape : shapes)
  {
    const int d = static_cast<int>(shape.size());
    std::printf("--- shape d=%d:", d);
    for (int n : shape)
      std::printf(" %d", n);
    std::printf(" ---\n");

    long long total = 1;
    for (int n : shape)
      total *= n;
    const std::size_t N = static_cast<std::size_t>(total);

    // ---- zeros ----
    {
      tt_ns::tt Z = tt_ns::zeros(shape);
      std::vector<double> want(N, 0.0);
      ok &= tu::check_close("zeros", Z.to_dense(), want, tol);
      const auto rs = Z.ranks();
      bool ranks_ok = (rs.front() == 1) && (rs.back() == 1);
      for (int k = 1; k < d; ++k)
        if (rs[k] != 1)
          ranks_ok = false;
      std::printf("  [zeros] rank check: %s\n", ranks_ok ? "OK" : "FAIL");
      ok &= ranks_ok;
    }

    // ---- ones ----
    {
      tt_ns::tt O = tt_ns::ones(shape);
      std::vector<double> want(N, 1.0);
      ok &= tu::check_close("ones", O.to_dense(), want, tol);
      const auto rs = O.ranks();
      bool ranks_ok = true;
      for (int r : rs)
        if (r != 1)
          ranks_ok = false;
      std::printf("  [ones] rank check: %s\n", ranks_ok ? "OK" : "FAIL");
      ok &= ranks_ok;
    }

    // ---- canonical_unit ----
    {
      // pick idx = floor(n/2) per axis.
      std::vector<int> idx(static_cast<std::size_t>(d));
      for (int k = 0; k < d; ++k)
        idx[static_cast<std::size_t>(k)] = shape[k] / 2;
      tt_ns::tt E = tt_ns::canonical_unit(shape, idx);
      // Build expected linear index in row-major.
      long long li = 0;
      for (int k = 0; k < d; ++k)
        li = li * shape[k] + idx[static_cast<std::size_t>(k)];
      std::vector<double> want(N, 0.0);
      want[static_cast<std::size_t>(li)] = 1.0;
      ok &= tu::check_close("canonical_unit", E.to_dense(), want, tol);

      // eval_at at idx must return 1.
      std::vector<int> idx_buf(idx.begin(), idx.end());
      const double v = tt_ns::eval_at(E, idx_buf.data());
      if (std::fabs(v - 1.0) > tol)
      {
        std::printf("  [canonical_unit] eval_at = %g, want 1\n", v);
        ok = false;
      }
    }

    // ---- random ----
    if (true)  // d=1 is valid for random()
    {
      const int max_rank = 3;
      tt_ns::tt R        = tt_ns::random(shape, max_rank, 0xC0FFEEu + d);
      const auto rs      = R.ranks();
      // Boundary ranks 1, internal ranks <= max_rank.
      bool ranks_ok = (rs.front() == 1) && (rs.back() == 1);
      for (int k = 1; k < d; ++k)
        if (rs[k] > max_rank)
          ranks_ok = false;
      std::printf("  [random] ranks:");
      for (int r : rs)
        std::printf(" %d", r);
      std::printf("  ranks_ok=%s\n", ranks_ok ? "OK" : "FAIL");
      ok &= ranks_ok;

      // Non-trivial: dense should not be all zero, and frob norm
      // should match inner(R, R).
      const auto dR   = R.to_dense();
      const double mx = tu::max_abs(dR);
      if (mx <= 0.0)
      {
        std::printf("  [random] dense is all zero\n");
        ok = false;
      }
      const double fr   = tt_ns::norm(R);
      const double fd   = tu::frob_norm(dR);
      const double err  = std::fabs(fr - fd);
      const double tolf = 1e-10 * (fd > 1.0 ? fd : 1.0);
      std::printf("  [random] frob: tt=%.6g dense=%.6g (err %.2e tol %.2e)\n",
                  fr,
                  fd,
                  err,
                  tolf);
      ok &= (err <= tolf);

      // Rounding a random TT with eps=1e-14 must keep the dense
      // tensor unchanged to high accuracy.
      tt_ns::round_options ro_rr;
      ro_rr.eps      = 1e-14;
      tt_ns::tt Rr   = tt_ns::round(R, ro_rr);
      const auto dRr = Rr.to_dense();
      ok &= tu::check_close("random round-trip", dRr, dR, 1e-9 * (fd + 1.0));

      // Determinism: same seed must give bitwise-identical dense.
      tt_ns::tt R2   = tt_ns::random(shape, max_rank, 0xC0FFEEu + d);
      const auto dR2 = R2.to_dense();
      ok &= tu::check_close("random determinism", dR2, dR, 0.0);
    }
  }

  if (!ok)
  {
    std::printf("test_tt_factory: FAIL\n");
    return 1;
  }
  std::printf("test_tt_factory: OK\n");
  return 0;
}
//
// :D
//
