/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Test the tt_matrix container: construction, dims, ranks, shapes, and dense reconstruction
*/
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "tensor_train.hpp"
#include "tt_test_utils.hpp"

using namespace mva::tensor_train;
namespace tu = mva::tensor_train::test_utils;

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

// --------------------------------------------------------------------
// Build a (1, m, n, 1) "single-core" tt_matrix from an m x n row-major
// buffer.  Dense should equal the buffer.
static void test_single_core()
{
  std::printf("--- single core (1, m, n, 1) ---\n");
  const int m = 3, n = 4;
  std::vector<double> dense(static_cast<std::size_t>(m * n));
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      dense[static_cast<std::size_t>(i * n + j)] = 1.0 + i * 10.0 + j;

  std::vector<tt_matrix_core> cores(1);
  cores[0].allocate(1, m, n, 1);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      cores[0](0, i, j, 0) = dense[static_cast<std::size_t>(i * n + j)];

  tt_matrix A(std::move(cores));
  CHECK(A.d() == 1, "d()");
  CHECK(A.row_shape() == std::vector<int>{ m }, "row_shape");
  CHECK(A.col_shape() == std::vector<int>{ n }, "col_shape");
  CHECK(A.ranks() == (std::vector<int>{ 1, 1 }), "ranks");
  CHECK(A.total_rows() == static_cast<std::size_t>(m), "total_rows");
  CHECK(A.total_cols() == static_cast<std::size_t>(n), "total_cols");

  std::vector<double> got = A.to_dense();
  const double err        = tu::max_abs_diff(got, dense);
  std::printf("  [single] sup-err = %.3e (tol %.3e)\n", err, 1e-15);
  CHECK(err <= 1e-15, "single core to_dense");
}

// --------------------------------------------------------------------
// Build a rank-1 d=2 tt_matrix as outer-product of two single cores.
// A((i_0,i_1), (j_0,j_1)) = U(i_0,j_0) * V(i_1,j_1).
static void test_rank1_d2()
{
  std::printf("--- rank-1 d=2 outer product ---\n");
  const int m0 = 2, n0 = 3, m1 = 3, n1 = 2;

  std::vector<double> U(static_cast<std::size_t>(m0 * n0));
  std::vector<double> V(static_cast<std::size_t>(m1 * n1));
  for (std::size_t k = 0; k < U.size(); ++k)
    U[k] = 0.5 + 0.1 * static_cast<double>(k);
  for (std::size_t k = 0; k < V.size(); ++k)
    V[k] = -0.3 + 0.2 * static_cast<double>(k);

  std::vector<tt_matrix_core> cores(2);
  cores[0].allocate(1, m0, n0, 1);
  cores[1].allocate(1, m1, n1, 1);
  for (int i = 0; i < m0; ++i)
    for (int j = 0; j < n0; ++j)
      cores[0](0, i, j, 0) = U[static_cast<std::size_t>(i * n0 + j)];
  for (int i = 0; i < m1; ++i)
    for (int j = 0; j < n1; ++j)
      cores[1](0, i, j, 0) = V[static_cast<std::size_t>(i * n1 + j)];

  tt_matrix A(std::move(cores));
  CHECK(A.d() == 2, "d()");
  CHECK(A.row_shape() == (std::vector<int>{ m0, m1 }), "row_shape");
  CHECK(A.col_shape() == (std::vector<int>{ n0, n1 }), "col_shape");
  CHECK(A.ranks() == (std::vector<int>{ 1, 1, 1 }), "ranks");

  const std::size_t M_total = static_cast<std::size_t>(m0 * m1);
  const std::size_t N_total = static_cast<std::size_t>(n0 * n1);
  std::vector<double> want(M_total * N_total);
  // Reference: I = i0*m1 + i1, J = j0*n1 + j1.
  for (int i0 = 0; i0 < m0; ++i0)
    for (int i1 = 0; i1 < m1; ++i1)
      for (int j0 = 0; j0 < n0; ++j0)
        for (int j1 = 0; j1 < n1; ++j1)
        {
          const std::size_t I   = static_cast<std::size_t>(i0 * m1 + i1);
          const std::size_t J   = static_cast<std::size_t>(j0 * n1 + j1);
          want[I * N_total + J] = U[static_cast<std::size_t>(i0 * n0 + j0)] *
                                  V[static_cast<std::size_t>(i1 * n1 + j1)];
        }

  std::vector<double> got = A.to_dense();
  const double err        = tu::max_abs_diff(got, want);
  std::printf("  [rank1-d2] sup-err = %.3e (tol %.3e)\n", err, 1e-13);
  CHECK(err <= 1e-13, "rank-1 d=2 to_dense");
}

// --------------------------------------------------------------------
// Build a rank-2 d=2 tt_matrix as a sum of two outer products and
// verify dense reconstruction matches U1 (x) V1 + U2 (x) V2.
static void test_rank2_d2()
{
  std::printf("--- rank-2 d=2 sum of outer products ---\n");
  const int m0 = 2, n0 = 2, m1 = 2, n1 = 3;
  const int r1 = 2;

  // Two single-product matrices.
  std::vector<std::vector<double>> Us(r1), Vs(r1);
  for (int s = 0; s < r1; ++s)
  {
    Us[s].resize(static_cast<std::size_t>(m0 * n0));
    Vs[s].resize(static_cast<std::size_t>(m1 * n1));
    for (std::size_t k = 0; k < Us[s].size(); ++k)
      Us[s][k] = 0.1 * (s + 1) + 0.07 * static_cast<double>(k);
    for (std::size_t k = 0; k < Vs[s].size(); ++k)
      Vs[s][k] = -0.2 * (s + 1) + 0.05 * static_cast<double>(k);
  }

  std::vector<tt_matrix_core> cores(2);
  cores[0].allocate(1, m0, n0, r1);
  cores[1].allocate(r1, m1, n1, 1);
  cores[0].zero_clear();
  cores[1].zero_clear();
  for (int s = 0; s < r1; ++s)
    for (int i = 0; i < m0; ++i)
      for (int j = 0; j < n0; ++j)
        cores[0](0, i, j, s) = Us[s][static_cast<std::size_t>(i * n0 + j)];
  for (int s = 0; s < r1; ++s)
    for (int i = 0; i < m1; ++i)
      for (int j = 0; j < n1; ++j)
        cores[1](s, i, j, 0) = Vs[s][static_cast<std::size_t>(i * n1 + j)];

  tt_matrix A(std::move(cores));
  CHECK(A.ranks() == (std::vector<int>{ 1, r1, 1 }), "ranks");

  const std::size_t M_total = static_cast<std::size_t>(m0 * m1);
  const std::size_t N_total = static_cast<std::size_t>(n0 * n1);
  std::vector<double> want(M_total * N_total, 0.0);
  for (int s = 0; s < r1; ++s)
    for (int i0 = 0; i0 < m0; ++i0)
      for (int i1 = 0; i1 < m1; ++i1)
        for (int j0 = 0; j0 < n0; ++j0)
          for (int j1 = 0; j1 < n1; ++j1)
          {
            const std::size_t I = static_cast<std::size_t>(i0 * m1 + i1);
            const std::size_t J = static_cast<std::size_t>(j0 * n1 + j1);
            want[I * N_total + J] +=
              Us[s][static_cast<std::size_t>(i0 * n0 + j0)] *
              Vs[s][static_cast<std::size_t>(i1 * n1 + j1)];
          }

  std::vector<double> got = A.to_dense();
  const double err        = tu::max_abs_diff(got, want);
  std::printf("  [rank2-d2] sup-err = %.3e (tol %.3e)\n", err, 1e-13);
  CHECK(err <= 1e-13, "rank-2 d=2 to_dense");
}

int main()
{
  std::printf("=== test_tt_matrix ===\n");
  test_single_core();
  test_rank1_d2();
  test_rank2_d2();

  if (fail_count == 0)
  {
    std::printf("test_tt_matrix: OK\n");
    return 0;
  }
  std::printf("test_tt_matrix: FAILED (%d)\n", fail_count);
  return 1;
}
//
// :D
//
