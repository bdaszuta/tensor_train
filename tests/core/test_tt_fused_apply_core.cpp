/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Test: detail::fused_matvec_core / fused_matmat_core
*/
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "core/detail/fused_apply_core.hpp"
#include "tensor_train.hpp"

namespace tt_ns = mva::tensor_train;
namespace dt    = mva::tensor_train::detail;
using row_mat   = mva::tensor_train::eigen_bridge::row_matrix;

static double max_abs_diff_buf(const double* a, const double* b, std::size_t n)
{
  double m = 0.0;
  for (std::size_t i = 0; i < n; ++i)
  {
    const double d = std::fabs(a[i] - b[i]);
    if (d > m)
      m = d;
  }
  return m;
}

static tt_ns::tt_matrix_core random_matrix_core(int rL,
                                                int m,
                                                int n,
                                                int rR,
                                                std::uint64_t seed)
{
  tt_ns::tt_matrix_core c(rL, m, n, rR);
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (std::size_t i = 0; i < c.size(); ++i)
    c.data()[i] = dist(rng);
  return c;
}

static tt_ns::tt_core random_tt_core(int rL, int n, int rR, std::uint64_t seed)
{
  tt_ns::tt_core c(rL, n, rR);
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (std::size_t i = 0; i < c.size(); ++i)
    c.data()[i] = dist(rng);
  return c;
}

static int check_matvec(const char* tag,
                        int a,
                        int m,
                        int n,
                        int b,
                        int sx,
                        int bx,
                        std::uint64_t seed)
{
  auto A = random_matrix_core(a, m, n, b, seed);
  auto x = random_tt_core(sx, n, bx, seed ^ 0xABCDEF01ULL);

  tt_ns::tt_core ref = dt::matvec_core(A, x);

  // F = identity of size (a*sx).
  const int as       = a * sx;
  row_mat F          = row_mat::Identity(as, as);
  tt_ns::tt_core got = dt::fused_matvec_core(F, A, x);

  if (got.r_left() != as || got.n_phys() != m || got.r_right() != b * bx)
  {
    std::printf("FAIL [%s] shape mismatch got=(%d,%d,%d) want=(%d,%d,%d)\n",
                tag,
                got.r_left(),
                got.n_phys(),
                got.r_right(),
                as,
                m,
                b * bx);
    return 1;
  }
  const double err = max_abs_diff_buf(
    got.data(), ref.data(), static_cast<std::size_t>(ref.size()));
  std::printf("  [%s matvec] sup-err=%.3e\n", tag, err);
  return err <= 1.0e-12 ? 0 : 1;
}

static int check_matmat(const char* tag,
                        int a,
                        int m,
                        int p,
                        int b,
                        int sx,
                        int n,
                        int bx,
                        std::uint64_t seed)
{
  auto A = random_matrix_core(a, m, p, b, seed);
  auto B = random_matrix_core(sx, p, n, bx, seed ^ 0x12345678ULL);

  tt_ns::tt_matrix_core ref = dt::matmat_core(A, B);

  const int as              = a * sx;
  row_mat F                 = row_mat::Identity(as, as);
  tt_ns::tt_matrix_core got = dt::fused_matmat_core(F, A, B);

  auto gd = got.dims();
  if (gd[0] != as || gd[1] != m || gd[2] != n || gd[3] != b * bx)
  {
    std::printf(
      "FAIL [%s] shape mismatch got=(%d,%d,%d,%d) want=(%d,%d,%d,%d)\n",
      tag,
      gd[0],
      gd[1],
      gd[2],
      gd[3],
      as,
      m,
      n,
      b * bx);
    return 1;
  }
  const double err = max_abs_diff_buf(
    got.data(), ref.data(), static_cast<std::size_t>(ref.size()));
  std::printf("  [%s matmat] sup-err=%.3e\n", tag, err);
  return err <= 1.0e-12 ? 0 : 1;
}

int main()
{
  int failed = 0;

  // Boundary: rank-1 left (interior of streaming round at first step).
  failed += check_matvec("rank1L", 1, 3, 4, 2, 1, 2, 0x111ULL);
  // General interior shape.
  failed += check_matvec("interior", 3, 2, 5, 4, 2, 3, 0x222ULL);
  failed += check_matvec("non-square", 4, 3, 2, 5, 3, 4, 0x333ULL);
  // Boundary: rank-1 right.
  failed += check_matvec("rank1R", 3, 4, 2, 1, 2, 1, 0x444ULL);

  failed += check_matmat("rank1L", 1, 2, 3, 2, 1, 4, 2, 0x555ULL);
  failed += check_matmat("interior", 3, 2, 4, 3, 2, 3, 4, 0x666ULL);
  failed += check_matmat("rank1R", 2, 3, 2, 1, 3, 4, 1, 0x777ULL);

  // NOTE: The fused kernels are currently only tested with F=Identity.
  // Testing with non-identity F requires constructing a reference that
  // matches the internal contraction order (F is interleaved, not a
  // simple post-multiplication of matvec_core).  This is a known
  // coverage gap.

  if (failed == 0)
  {
    std::printf("PASS: all fused-apply identity checks\n");
    return 0;
  }
  std::printf("FAIL: %d fused-apply checks failed\n", failed);
  return 1;
}
//
// :D
//
