/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Test: right_orthogonalize preserves the dense tensor and produces
*/
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "tensor_train.hpp"
#include "tt_test_utils.hpp"

namespace tt_ns = mva::tensor_train;
namespace tu    = mva::tensor_train::test_utils;

// Check that right_unfold(c) (shape r_left x n_phys*r_right) has rows
// orthonormal: G = M M^T = I_{r_left}.  Returns sup |G - I|.
static double right_orth_residual(const tt_ns::tt_core& c)
{
  const int r  = c.r_left();
  const int n  = c.n_phys();
  const int rp = c.r_right();
  const int sp = n * rp;
  // M(i, j) = c.data()[i * sp + j]; G(a,b) = sum_j M(a,j) * M(b,j).
  const double* d = c.data();
  double err      = 0.0;
  for (int a = 0; a < r; ++a)
  {
    for (int b = 0; b < r; ++b)
    {
      double s = 0.0;
      for (int j = 0; j < sp; ++j)
      {
        s += d[a * sp + j] * d[b * sp + j];
      }
      const double want = (a == b) ? 1.0 : 0.0;
      const double e    = std::fabs(s - want);
      if (e > err)
        err = e;
    }
  }
  return err;
}

int main()
{
  int failed = 0;

  // ---------------- tt path ----------------
  {
    const std::vector<int> shape = { 4, 5, 6, 4 };
    auto T                       = tu::make_dense(shape, 7);
    auto a              = tt_ns::from_dense(T.data(), shape, 1.0e-12);
    auto a_dense_before = a.to_dense();

    tt_ns::right_orthogonalize(a);

    auto a_dense_after     = a.to_dense();
    const double dense_err = tu::frob_diff(a_dense_before, a_dense_after) /
                             tu::frob_norm(a_dense_before);
    std::printf("  [tt dense] rel-err = %.3e\n", dense_err);
    if (dense_err > 1.0e-12)
    {
      std::printf("FAIL: tt right_orthogonalize changed dense tensor\n");
      ++failed;
    }

    // Cores 1..d-1 must be right-orthogonal.
    const int d = a.d();
    for (int k = 1; k < d; ++k)
    {
      const double e = right_orth_residual(a.core(k));
      std::printf("  [tt core %d] orth-resid = %.3e\n", k, e);
      if (e > 1.0e-12)
      {
        std::printf("FAIL: tt core %d not right-orthogonal\n", k);
        ++failed;
      }
    }
  }

  // ---------------- tt left_orthogonalize ----------------
  {
    const std::vector<int> shape = { 4, 5, 6, 4 };
    auto T                       = tu::make_dense(shape, 7);
    auto a              = tt_ns::from_dense(T.data(), shape, 1.0e-12);
    auto a_dense_before = a.to_dense();

    tt_ns::detail::left_orthogonalize(a);

    auto a_dense_after     = a.to_dense();
    const double dense_err = tu::frob_diff(a_dense_before, a_dense_after) /
                             tu::frob_norm(a_dense_before);
    std::printf("  [tt left-orth dense] rel-err = %.3e\n", dense_err);
    if (dense_err > 1.0e-12)
    {
      std::printf("FAIL: tt left_orthogonalize changed dense tensor\n");
      ++failed;
    }

    // Cores 0..d-2 must be left-orthogonal (columns of left_unfold
    // are orthonormal: V^T V = I_{r_right}).
    const int d = a.d();
    for (int k = 0; k < d - 1; ++k)
    {
      const tt_ns::tt_core& c = a.core(k);
      const int r  = c.r_left();
      const int n  = c.n_phys();
      const int rp = c.r_right();
      const int sp = r * n;
      const double* cd = c.data();
      double err_o = 0.0;
      for (int ca = 0; ca < rp; ++ca)
      {
        for (int cb = 0; cb < rp; ++cb)
        {
          double s = 0.0;
          for (int row = 0; row < sp; ++row)
            s += cd[row * rp + ca] * cd[row * rp + cb];
          const double want = (ca == cb) ? 1.0 : 0.0;
          const double e    = std::fabs(s - want);
          if (e > err_o) err_o = e;
        }
      }
      std::printf("  [tt left-orth core %d] orth-resid = %.3e\n", k, err_o);
      if (err_o > 1.0e-12)
      {
        std::printf("FAIL: tt core %d not left-orthogonal\n", k);
        ++failed;
      }
    }
  }

  // ---------------- tt_matrix path ----------------
  {
    const std::vector<int> rs = { 3, 4, 3 };
    const std::vector<int> cs = { 2, 3, 4 };
    auto A                    = tt_ns::random(rs, cs, 3, 13);
    auto A_dense_before       = A.to_dense();

    tt_ns::right_orthogonalize(A);

    auto A_dense_after     = A.to_dense();
    const double dense_err = tu::frob_diff(A_dense_before, A_dense_after) /
                             tu::frob_norm(A_dense_before);
    std::printf("  [tt_matrix dense] rel-err = %.3e\n", dense_err);
    if (dense_err > 1.0e-12)
    {
      std::printf("FAIL: tt_matrix right_orthogonalize changed matrix\n");
      ++failed;
    }

    // After packing, matrix-cores k=1..d-1 viewed as 3-axis cores of
    // mode size m*n must be right-orthogonal.
    const int d = A.d();
    for (int k = 1; k < d; ++k)
    {
      const auto& mc = A.core(k);
      tt_ns::tt_core c(mc.r_left(), mc.m_phys() * mc.n_phys(), mc.r_right());
      std::memcpy(c.data(),
                  mc.data(),
                  sizeof(double) * static_cast<std::size_t>(c.size()));
      const double e = right_orth_residual(c);
      std::printf("  [tt_matrix core %d] orth-resid = %.3e\n", k, e);
      if (e > 1.0e-12)
      {
        std::printf("FAIL: tt_matrix core %d not right-orthogonal\n", k);
        ++failed;
      }
    }
  }

  if (failed == 0)
    std::printf("test_tt_orthogonalize: OK\n");
  return failed == 0 ? 0 : 1;
}
//
// :D
//
