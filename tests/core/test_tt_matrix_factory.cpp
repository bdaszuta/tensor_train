/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Test: tt_matrix factories (zeros, identity, diag_from_tt, random, from_dense)
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

// --------------------------------------------------------------------
static void test_zeros()
{
  std::printf("--- zeros ---\n");
  const std::vector<int> rs = { 2, 3 };
  const std::vector<int> cs = { 3, 2 };
  tt_ns::tt_matrix Z        = tt_ns::zeros(rs, cs);
  CHECK(Z.row_shape() == rs, "row_shape");
  CHECK(Z.col_shape() == cs, "col_shape");
  CHECK(Z.ranks() == (std::vector<int>{ 1, 1, 1 }), "ranks");
  const auto dense = Z.to_dense();
  CHECK(dense.size() == 2 * 3 * 3 * 2, "dense size");
  CHECK(tu::max_abs(dense) == 0.0, "all zero");
}

// --------------------------------------------------------------------
static void test_identity()
{
  std::printf("--- identity ---\n");
  const std::vector<int> shape = { 2, 3, 2 };
  tt_ns::tt_matrix I           = tt_ns::identity(shape);
  CHECK(I.row_shape() == shape, "row_shape");
  CHECK(I.col_shape() == shape, "col_shape");
  CHECK(I.ranks() == (std::vector<int>{ 1, 1, 1, 1 }), "rank-1");

  const std::size_t N = 2 * 3 * 2;
  const auto dense    = I.to_dense();
  CHECK(dense.size() == N * N, "dense size");

  std::vector<double> want(N * N, 0.0);
  for (std::size_t i = 0; i < N; ++i)
    want[i * N + i] = 1.0;
  const double err = tu::max_abs_diff(dense, want);
  std::printf("  [identity] sup-err = %.3e\n", err);
  CHECK(err <= 1e-14, "identity dense");
}

// --------------------------------------------------------------------
static void test_diag_from_tt()
{
  std::printf("--- diag_from_tt ---\n");
  const std::vector<int> shape = { 2, 3, 2 };
  // Build a non-trivial tt vector via random.
  tt_ns::tt v         = tt_ns::random(shape, 2, 0xD1A6u);
  const auto vd       = v.to_dense();
  const std::size_t N = vd.size();

  tt_ns::tt_matrix D = tt_ns::diag_from_tt(v);
  CHECK(D.row_shape() == shape, "row_shape");
  CHECK(D.col_shape() == shape, "col_shape");
  // Ranks of D match ranks of v.
  CHECK(D.ranks() == v.ranks(), "ranks match v");

  const auto dense = D.to_dense();
  CHECK(dense.size() == N * N, "dense size");

  // Verify diag entries == vd, off-diag == 0.
  double sup_off = 0.0;
  double sup_dia = 0.0;
  for (std::size_t i = 0; i < N; ++i)
  {
    for (std::size_t j = 0; j < N; ++j)
    {
      const double a = dense[i * N + j];
      if (i == j)
      {
        const double d = std::fabs(a - vd[i]);
        if (d > sup_dia)
          sup_dia = d;
      }
      else
      {
        const double d = std::fabs(a);
        if (d > sup_off)
          sup_off = d;
      }
    }
  }
  std::printf("  [diag] sup_dia=%.3e sup_off=%.3e\n", sup_dia, sup_off);
  CHECK(sup_dia <= 1e-13, "diagonal matches v");
  CHECK(sup_off <= 1e-13, "off-diagonal zero");
}

// --------------------------------------------------------------------
static void test_random_matrix()
{
  std::printf("--- random ---\n");
  const std::vector<int> rs = { 2, 3, 2 };
  const std::vector<int> cs = { 3, 2, 3 };
  const int max_rank        = 3;
  tt_ns::tt_matrix R        = tt_ns::random(rs, cs, max_rank, 0xBEEF1u);
  const auto ranks          = R.ranks();
  std::printf("  [random] ranks:");
  for (int r : ranks)
    std::printf(" %d", r);
  std::printf("\n");
  CHECK(ranks.front() == 1 && ranks.back() == 1, "boundary ranks");
  for (std::size_t k = 1; k + 1 < ranks.size(); ++k)
    CHECK(ranks[k] <= max_rank, "internal rank cap");

  // Determinism.
  tt_ns::tt_matrix R2 = tt_ns::random(rs, cs, max_rank, 0xBEEF1u);
  const auto d1       = R.to_dense();
  const auto d2       = R2.to_dense();
  CHECK(tu::max_abs_diff(d1, d2) <= 1.0e-14, "determinism");

  // Non-trivial entries.
  CHECK(tu::max_abs(d1) > 0.0, "non-zero");
}

// --------------------------------------------------------------------
static void test_from_dense_round_trip()
{
  std::printf("--- from_dense round-trip ---\n");
  const std::vector<int> rs = { 2, 3 };
  const std::vector<int> cs = { 3, 2 };

  const std::size_t M = 2 * 3;
  const std::size_t N = 3 * 2;
  std::vector<double> dense(M * N);
  // Fill with a smooth deterministic pattern.
  for (std::size_t I = 0; I < M; ++I)
    for (std::size_t J = 0; J < N; ++J)
      dense[I * N + J] = std::sin(0.13 * static_cast<double>(I + 1)) +
                         0.5 * std::cos(0.21 * static_cast<double>(J + 1)) +
                         0.01 * static_cast<double>(I * N + J);

  tt_ns::tt_matrix A = tt_ns::from_dense(dense.data(), rs, cs, 1.0e-12);
  CHECK(A.row_shape() == rs, "row_shape");
  CHECK(A.col_shape() == cs, "col_shape");

  const auto got   = A.to_dense();
  const double err = tu::max_abs_diff(got, dense);
  std::printf("  [from_dense] sup-err = %.3e\n", err);
  CHECK(err <= 1e-10, "round-trip");

  // Also: low-rank dense should compress.  Build a rank-1 outer
  // product u*v^T-style and verify compression keeps r <= 1.
  std::vector<double> u(M), v(N);
  for (std::size_t i = 0; i < M; ++i)
    u[i] = 0.3 + 0.1 * static_cast<double>(i);
  for (std::size_t j = 0; j < N; ++j)
    v[j] = -0.2 + 0.07 * static_cast<double>(j);
  std::vector<double> low(M * N);
  for (std::size_t i = 0; i < M; ++i)
    for (std::size_t j = 0; j < N; ++j)
      low[i * N + j] = u[i] * v[j];

  tt_ns::tt_matrix L = tt_ns::from_dense(low.data(), rs, cs, 1.0e-12);
  const auto Lr      = L.ranks();
  std::printf("  [from_dense low-rank] ranks:");
  for (int r : Lr)
    std::printf(" %d", r);
  std::printf("\n");
  // u*v^T does NOT factor as outer product over (i_0,j_0) x (i_1,j_1)
  // in general, so we don't insist on rank 1 here; we just require
  // the dense round-trip.
  const auto Ld     = L.to_dense();
  const double err2 = tu::max_abs_diff(Ld, low);
  std::printf("  [from_dense low-rank] sup-err = %.3e\n", err2);
  CHECK(err2 <= 1e-10, "low-rank round-trip");
}

int main()
{
  std::printf("=== test_tt_matrix_factory ===\n");
  test_zeros();
  test_identity();
  test_diag_from_tt();
  test_random_matrix();
  test_from_dense_round_trip();

  if (fail_count == 0)
  {
    std::printf("test_tt_matrix_factory: OK\n");
    return 0;
  }
  std::printf("test_tt_matrix_factory: FAILED (%d)\n", fail_count);
  return 1;
}
//
// :D
//
