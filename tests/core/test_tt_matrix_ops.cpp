/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Test: tt_matrix ops (scale, neg, add, sub, axpy, axpby)
*/
#include <cmath>
#include <cstdio>
#include <vector>

#include "tensor_train.hpp"
#include "tt_test_utils.hpp"

namespace tt_ns = mva::tensor_train;
namespace tu    = mva::tensor_train::test_utils;

static int fail_count = 0;

#define CHECK(cond, msg)                \
  do                                    \
  {                                     \
    if (!(cond))                        \
    {                                   \
      std::printf("  FAIL: %s\n", msg); \
      ++fail_count;                     \
    }                                   \
  } while (0)

static std::vector<double> dense_axpby(double alpha,
                                       const std::vector<double>& a,
                                       double beta,
                                       const std::vector<double>& b)
{
  std::vector<double> r(a.size());
  for (std::size_t i = 0; i < a.size(); ++i)
    r[i] = alpha * a[i] + beta * b[i];
  return r;
}

int main()
{
  std::printf("=== test_tt_matrix_ops ===\n");

  const std::vector<int> rs = { 2, 3, 2 };
  const std::vector<int> cs = { 3, 2, 3 };

  tt_ns::tt_matrix A = tt_ns::random(rs, cs, 2, 0xAA01u);
  tt_ns::tt_matrix B = tt_ns::random(rs, cs, 3, 0xBB02u);

  const auto dA = A.to_dense();
  const auto dB = B.to_dense();

  // Use lambda dispatch via switch since we have multiple signatures.
  auto run_case = [&](const char* name,
                      const tt_ns::tt_matrix& C,
                      const std::vector<double>& want)
  {
    const auto dC    = C.to_dense();
    const double err = tu::max_abs_diff(dC, want);
    std::printf("  [%s] sup-err = %.3e\n", name, err);
    if (err > 1e-12)
    {
      std::printf("  FAIL: %s\n", name);
      ++fail_count;
    }
    // Rank check: interior ranks should be ranks(A) + ranks(B).
    const auto rA = A.ranks();
    const auto rB = B.ranks();
    const auto rC = C.ranks();
    bool rank_ok  = (rC.front() == 1) && (rC.back() == 1);
    for (std::size_t k = 1; k + 1 < rC.size(); ++k)
      if (rC[k] != rA[k] + rB[k])
        rank_ok = false;
    if (!rank_ok)
    {
      std::printf("  FAIL: %s rank growth (got", name);
      for (int r : rC)
        std::printf(" %d", r);
      std::printf(")\n");
      ++fail_count;
    }
  };

  run_case("add", tt_ns::add(A, B), dense_axpby(1.0, dA, 1.0, dB));
  run_case("sub", tt_ns::sub(A, B), dense_axpby(1.0, dA, -1.0, dB));
  run_case("axpy(2.5)", tt_ns::axpy(2.5, A, B), dense_axpby(2.5, dA, 1.0, dB));
  run_case("axpby(1.3,-0.7)",
           tt_ns::axpby(1.3, A, -0.7, B),
           dense_axpby(1.3, dA, -0.7, dB));

  // ---- scale / neg ----
  {
    const double alpha  = 2.5;
    tt_ns::tt_matrix Cs = tt_ns::scale(A, alpha);
    const auto dCs      = Cs.to_dense();
    auto wantS          = dense_axpby(alpha, dA, 0.0, dA);
    double errS         = tu::max_abs_diff(dCs, wantS);
    std::printf("  [scale] sup-err = %.3e\n", errS);
    CHECK(errS <= 1e-12, "scale");
  }
  {
    tt_ns::tt_matrix Cn = tt_ns::neg(A);
    const auto dCn      = Cn.to_dense();
    auto wantN          = dense_axpby(-1.0, dA, 0.0, dA);
    double errN         = tu::max_abs_diff(dCn, wantN);
    std::printf("  [neg] sup-err = %.3e\n", errN);
    CHECK(errN <= 1e-12, "neg");
  }

  // ---- frob_inner ----
  {
    // frob_inner(A, B) = sum_{I,J} A(I,J) B(I,J) = dot(dA, dB).
    double ref      = 0.0;
    for (std::size_t i = 0; i < dA.size(); ++i)
      ref += dA[i] * dB[i];
    double got     = tt_ns::frob_inner(A, B);
    double err_in  = std::fabs(got - ref);
    std::printf("  [frob_inner] ref=%.10e got=%.10e err=%.3e\n", ref, got, err_in);
    CHECK(err_in <= 1e-12, "frob_inner");
  }

  // ---- d=1 special case ----
  {
    const std::vector<int> rs1 = { 4 };
    const std::vector<int> cs1 = { 3 };
    tt_ns::tt_matrix P         = tt_ns::random(rs1, cs1, 1, 0xCCC3u);
    tt_ns::tt_matrix Q         = tt_ns::random(rs1, cs1, 1, 0xDDD4u);
    const auto dP              = P.to_dense();
    const auto dQ              = Q.to_dense();
    tt_ns::tt_matrix R         = tt_ns::axpby(0.4, P, -1.1, Q);
    const auto dR              = R.to_dense();
    const auto wR              = dense_axpby(0.4, dP, -1.1, dQ);
    const double err           = tu::max_abs_diff(dR, wR);
    std::printf("  [d=1 axpby] sup-err = %.3e\n", err);
    CHECK(err <= 1e-13, "d=1 axpby");
    CHECK(R.ranks() == (std::vector<int>{ 1, 1 }), "d=1 rank");
  }

  if (fail_count == 0)
  {
    std::printf("test_tt_matrix_ops: OK\n");
    return 0;
  }
  std::printf("test_tt_matrix_ops: FAILED (%d)\n", fail_count);
  return 1;
}
//
// :D
//
