/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Test: frob_norm_apply(A, x) and frob_norm_apply(A, B)
*/
#include <cmath>
#include <cstdio>
#include <vector>

#include "tensor_train.hpp"
#include "tt_test_utils.hpp"

namespace tt_ns = mva::tensor_train;

static int check_rel(const char* tag, double got, double ref, double tol)
{
  const double denom = std::fabs(ref) > 0.0 ? std::fabs(ref) : 1.0;
  const double rel   = std::fabs(got - ref) / denom;
  std::printf("  [%s] got=%.15e ref=%.15e rel=%.3e\n", tag, got, ref, rel);
  return rel <= tol ? 0 : 1;
}

int main()
{
  int failed       = 0;
  const double tol = 1.0e-13;

  // matvec norm: small and medium cases.
  {
    const std::vector<int> rs = { 2, 3, 2 };
    const std::vector<int> cs = { 3, 2, 3 };
    auto A                    = tt_ns::random(rs, cs, 3, 0xA10001ULL);
    auto x                    = tt_ns::random(cs, 2, 0x55AA77ULL);
    auto y                    = tt_ns::matvec(A, x);
    double ref                = tt_ns::norm(y);
    double got                = tt_ns::frob_norm_apply(A, x);
    failed += check_rel("matvec d=3 small", got, ref, tol);
  }
  {
    const std::vector<int> rs = { 4, 3, 4, 2 };
    const std::vector<int> cs = { 3, 4, 2, 3 };
    auto A                    = tt_ns::random(rs, cs, 4, 0xBEEF01ULL);
    auto x                    = tt_ns::random(cs, 3, 0xDEAD02ULL);
    auto y                    = tt_ns::matvec(A, x);
    double ref                = tt_ns::norm(y);
    double got                = tt_ns::frob_norm_apply(A, x);
    failed += check_rel("matvec d=4", got, ref, tol);
  }

  // matmat norm.
  {
    const std::vector<int> rs = { 2, 3, 2 };
    const std::vector<int> cs = { 3, 2, 3 };
    auto A                    = tt_ns::random(rs, cs, 3, 0xA10001ULL);
    auto B                    = tt_ns::random(cs, rs, 2, 0xB20002ULL);
    auto C                    = tt_ns::matmat(A, B);
    double ref                = tt_ns::frob_norm(C);
    double got                = tt_ns::frob_norm_apply(A, B);
    failed += check_rel("matmat d=3 small", got, ref, tol);
  }
  {
    const std::vector<int> rs = { 4, 3, 4, 2 };
    const std::vector<int> cs = { 3, 4, 2, 3 };
    auto A                    = tt_ns::random(rs, cs, 3, 0xBEEF01ULL);
    auto B                    = tt_ns::random(cs, rs, 2, 0xDEAD02ULL);
    auto C                    = tt_ns::matmat(A, B);
    double ref                = tt_ns::frob_norm(C);
    double got                = tt_ns::frob_norm_apply(A, B);
    failed += check_rel("matmat d=4", got, ref, tol);
  }

  // Boundary: d=1.
  {
    const std::vector<int> rs = { 5 };
    const std::vector<int> cs = { 4 };
    auto A                    = tt_ns::random(rs, cs, 1, 0x1111ULL);
    auto x                    = tt_ns::random(cs, 1, 0x2222ULL);
    auto y                    = tt_ns::matvec(A, x);
    double ref                = tt_ns::norm(y);
    double got                = tt_ns::frob_norm_apply(A, x);
    failed += check_rel("matvec d=1", got, ref, tol);
  }

  if (failed == 0)
  {
    std::printf("PASS: all frob_norm_apply cross-checks\n");
    return 0;
  }
  std::printf("FAIL: %d frob_norm_apply checks failed\n", failed);
  return 1;
}
//
// :D
//
