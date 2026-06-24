/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Test: matvec(A, x) and matmat(A, B) for tt_matrix
*/
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "tensor_train.hpp"
#include "tt_test_utils.hpp"

namespace tt_ns = mva::tensor_train;
namespace tu    = mva::tensor_train::test_utils;

// Dense matvec: y = A * x with A (M x N) row-major, x length N.
static std::vector<double> dense_matvec(const std::vector<double>& A,
                                        std::size_t M,
                                        std::size_t N,
                                        const std::vector<double>& x)
{
  std::vector<double> y(M, 0.0);
  for (std::size_t i = 0; i < M; ++i)
  {
    double s = 0.0;
    for (std::size_t j = 0; j < N; ++j)
      s += A[i * N + j] * x[j];
    y[i] = s;
  }
  return y;
}

// Dense matmat: C = A * B, A (M x K), B (K x N), all row-major.
static std::vector<double> dense_matmat(const std::vector<double>& A,
                                        std::size_t M,
                                        std::size_t K,
                                        const std::vector<double>& B,
                                        std::size_t N)
{
  std::vector<double> C(M * N, 0.0);
  for (std::size_t i = 0; i < M; ++i)
    for (std::size_t k = 0; k < K; ++k)
    {
      const double a = A[i * K + k];
      for (std::size_t j = 0; j < N; ++j)
        C[i * N + j] += a * B[k * N + j];
    }
  return C;
}

int main()
{
  int failed = 0;

  // 1) Identity test: I * x == x.
  {
    const std::vector<int> shape = { 2, 3, 2 };
    auto I                       = tt_ns::identity(shape);
    auto x                       = tt_ns::random(shape, 3, 0xC0FFEEULL);
    auto y                       = tt_ns::matvec(I, x);
    auto xd                      = x.to_dense();
    auto yd                      = y.to_dense();
    double er                    = tu::frob_diff(xd, yd) / tu::frob_norm(xd);
    std::printf("[I*x]   rel-err = %.3e\n", er);
    if (er > 1.0e-12)
    {
      std::printf("FAIL: I * x != x\n");
      ++failed;
    }
  }

  // 2) Random A * random x vs dense reference.
  {
    const std::vector<int> rs = { 2, 3, 2 };
    const std::vector<int> cs = { 3, 2, 3 };
    auto A                    = tt_ns::random(rs, cs, 3, 0xA10001ULL);
    auto x                    = tt_ns::random(cs, 2, 0x55AA77ULL);
    auto y                    = tt_ns::matvec(A, x);

    auto Ad    = A.to_dense();
    auto xd    = x.to_dense();
    auto yd    = y.to_dense();
    auto y_ref = dense_matvec(Ad, A.total_rows(), A.total_cols(), xd);

    double er = tu::frob_diff(yd, y_ref) / tu::frob_norm(y_ref);
    std::printf("[A*x]   rel-err = %.3e (yd size=%zu, ref size=%zu)\n",
                er,
                yd.size(),
                y_ref.size());
    if (er > 1.0e-12)
    {
      std::printf("FAIL: A * x mismatch\n");
      ++failed;
    }
    // Output mode shape == row_shape(A).
    if (y.shape() != rs)
    {
      std::printf("FAIL: y.shape() != A.row_shape()\n");
      ++failed;
    }
  }

  // 3) Random A * random B vs dense.
  {
    const std::vector<int> rs = { 2, 3, 2 };
    const std::vector<int> ps = { 3, 2, 3 };
    const std::vector<int> ns = { 2, 3, 2 };
    auto A                    = tt_ns::random(rs, ps, 2, 0xAA1001ULL);
    auto B                    = tt_ns::random(ps, ns, 2, 0xBB2002ULL);
    auto C                    = tt_ns::matmat(A, B);

    auto Ad = A.to_dense();
    auto Bd = B.to_dense();
    auto Cd = C.to_dense();
    auto C_ref =
      dense_matmat(Ad, A.total_rows(), A.total_cols(), Bd, B.total_cols());
    double er = tu::frob_diff(Cd, C_ref) / tu::frob_norm(C_ref);
    std::printf("[A*B]   rel-err = %.3e\n", er);
    if (er > 1.0e-12)
    {
      std::printf("FAIL: A * B mismatch\n");
      ++failed;
    }
    if (C.row_shape() != rs || C.col_shape() != ns)
    {
      std::printf("FAIL: C row/col shape\n");
      ++failed;
    }

    // Associativity: (A*B)*x == A*(B*x).
    auto x     = tt_ns::random(ns, 2, 0xCAFEEEULL);
    auto y1    = tt_ns::matvec(C, x);
    auto y2    = tt_ns::matvec(A, tt_ns::matvec(B, x));
    auto y1d   = y1.to_dense();
    auto y2d   = y2.to_dense();
    double er2 = tu::frob_diff(y1d, y2d) / tu::frob_norm(y1d);
    std::printf("[assoc] rel-err = %.3e\n", er2);
    if (er2 > 1.0e-10)
    {
      std::printf("FAIL: associativity\n");
      ++failed;
    }
  }

  if (failed == 0)
    std::printf("test_tt_matrix_apply: OK\n");
  return failed == 0 ? 0 : 1;
}
//
// :D
//
