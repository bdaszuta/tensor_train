/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Test basic TT-vs-TT operators: scale, neg, add, sub, hadamard, axpy
*/
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "tensor_train.hpp"
#include "tt_test_utils.hpp"

namespace tt_ns = mva::tensor_train;
namespace tu    = mva::tensor_train::test_utils;

int main()
{
  const double tol = 1e-10;
  bool ok          = true;

  // -------- shapes to exercise --------
  const std::vector<std::vector<int>> shapes = { { 5 },             // d == 1
                                                 { 3, 4 },          // d == 2
                                                 { 3, 4, 2 },       // d == 3
                                                 { 2, 3, 2, 3 } };  // d == 4

  for (const auto& shape : shapes)
  {
    const int d = static_cast<int>(shape.size());
    std::printf("--- shape d=%d:", d);
    for (int n : shape)
      std::printf(" %d", n);
    std::printf(" ---\n");

    // Two independent random dense tensors of the same shape.
    auto da = tu::make_dense(shape, 0xA11CE0u + d);
    auto db = tu::make_dense(shape, 0xB0BB1Eu + d);

    // Compress them with tight eps so the TT round-trip is essentially
    // exact (within floating-point noise of the SVD).
    tt_ns::tt A = tt_ns::from_dense(da.data(), shape, 1e-13);
    tt_ns::tt B = tt_ns::from_dense(db.data(), shape, 1e-13);

    // Use the TT->dense versions of A and B as our reference, so we
    // are not testing from_dense accuracy here.
    const auto dA = A.to_dense();
    const auto dB = B.to_dense();

    // ---- add ----
    {
      tt_ns::tt C = tt_ns::add(A, B);
      std::vector<double> want(dA.size());
      for (std::size_t i = 0; i < dA.size(); ++i)
        want[i] = dA[i] + dB[i];
      ok &= tu::check_close("add", C.to_dense(), want, tol);

      // Verify rank structure.
      const auto rA = A.ranks();
      const auto rB = B.ranks();
      const auto rC = C.ranks();
      bool ranks_ok = (rC.front() == 1) && (rC.back() == 1);
      for (int k = 1; k < d; ++k)
      {
        const int expected = rA[k] + rB[k];
        if (rC[k] != expected)
        {
          ranks_ok = false;
          std::printf("  [add] rank[%d] = %d, expected %d (rA=%d, rB=%d)\n",
                      k,
                      rC[k],
                      expected,
                      rA[k],
                      rB[k]);
        }
      }
      // Boundary bonds in TT are always 1, but interior must be sum.
      // For d == 1, no interior bonds and the size-1 cores hold the sum.
      if (d == 1)
      {
        ranks_ok = (rC[0] == 1) && (rC[1] == 1);
      }
      std::printf("  [add] rank check: %s\n", ranks_ok ? "OK" : "FAIL");
      ok &= ranks_ok;
    }

    // ---- sub ----
    {
      tt_ns::tt C = tt_ns::sub(A, B);
      std::vector<double> want(dA.size());
      for (std::size_t i = 0; i < dA.size(); ++i)
        want[i] = dA[i] - dB[i];
      ok &= tu::check_close("sub", C.to_dense(), want, tol);

      // Self-cancellation A - A.
      tt_ns::tt Z = tt_ns::sub(A, A);
      std::vector<double> zero(dA.size(), 0.0);
      ok &= tu::check_close("sub(A,A)", Z.to_dense(), zero, tol);
    }

    // ---- hadamard ----
    {
      tt_ns::tt C = tt_ns::hadamard(A, B);
      std::vector<double> want(dA.size());
      for (std::size_t i = 0; i < dA.size(); ++i)
        want[i] = dA[i] * dB[i];
      ok &= tu::check_close("hadamard", C.to_dense(), want, tol);

      // Verify rank structure: r_C = r_A * r_B per bond.
      const auto rA = A.ranks();
      const auto rB = B.ranks();
      const auto rC = C.ranks();
      bool ranks_ok = true;
      for (std::size_t k = 0; k < rC.size(); ++k)
      {
        const int expected = rA[k] * rB[k];
        if (rC[k] != expected)
        {
          ranks_ok = false;
          std::printf(
            "  [hadamard] rank[%zu] = %d, expected %d\n", k, rC[k], expected);
        }
      }
      std::printf("  [hadamard] rank check: %s\n", ranks_ok ? "OK" : "FAIL");
      ok &= ranks_ok;
    }

    // ---- axpy ----
    {
      const double a_coef = -2.5;
      tt_ns::tt C         = tt_ns::axpy(a_coef, A, B);
      std::vector<double> want(dA.size());
      for (std::size_t i = 0; i < dA.size(); ++i)
        want[i] = a_coef * dA[i] + dB[i];
      ok &= tu::check_close("axpy", C.to_dense(), want, tol);
    }

    // ---- scale ----
    {
      const double alpha = 2.5;
      tt_ns::tt C        = tt_ns::scale(A, alpha);
      std::vector<double> want(dA.size());
      for (std::size_t i = 0; i < dA.size(); ++i)
        want[i] = alpha * dA[i];
      ok &= tu::check_close("scale", C.to_dense(), want, tol);
    }

    // ---- neg ----
    {
      tt_ns::tt C = tt_ns::neg(A);
      std::vector<double> want(dA.size());
      for (std::size_t i = 0; i < dA.size(); ++i)
        want[i] = -dA[i];
      ok &= tu::check_close("neg", C.to_dense(), want, tol);
    }

    // ---- round(add(a, a)) should drop ranks back ----
    // add(A, A) has 2x ranks but represents 2 * A which has the same
    // rank as A; rounding with tight eps should recover A's rank.
    if (d >= 2)
    {
      tt_ns::tt S = tt_ns::add(A, A);
      tt_ns::round_options ro_s;
      ro_s.eps    = 1e-12;
      tt_ns::tt R = tt_ns::round(S, ro_s);
      std::vector<double> want(dA.size());
      for (std::size_t i = 0; i < dA.size(); ++i)
        want[i] = 2.0 * dA[i];
      ok &= tu::check_close("round(add(A,A))", R.to_dense(), want, 1e-9);

      const auto rA = A.ranks();
      const auto rR = R.ranks();
      bool ranks_ok = true;
      for (std::size_t k = 0; k < rA.size(); ++k)
      {
        if (rR[k] > rA[k])
        {
          ranks_ok = false;
          std::printf("  [round(add)] rank[%zu] = %d, expected <= %d\n",
                      k,
                      rR[k],
                      rA[k]);
        }
      }
      std::printf("  [round(add)] rank-collapse check: %s\n",
                  ranks_ok ? "OK" : "FAIL");
      ok &= ranks_ok;
    }
  }

  // Reference dA non-empty sanity (should be > 0 for our random inputs).
  {
    auto da = tu::make_dense({ 3, 3 }, 1234u);
    if (tu::max_abs(da) <= 0.0)
    {
      std::printf("sanity: random dense tensor was zero\n");
      ok = false;
    }
  }

  if (!ok)
  {
    std::printf("test_tt_ops: FAIL\n");
    return 1;
  }
  std::printf("test_tt_ops: OK\n");
  return 0;
}
//
// :D
//
