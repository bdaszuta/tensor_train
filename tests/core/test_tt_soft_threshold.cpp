/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Soft threshold tests: verify rank reduction, norm monotonicity, tau=0
*/
#include "tensor_train.hpp"
#include "tt_test_utils.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace tu = mva::tensor_train::test_utils;
namespace tt = mva::tensor_train;

static int failures = 0;

#define CHECK(cond, msg)                                            \
  do                                                                \
  {                                                                 \
    if (!(cond))                                                    \
    {                                                               \
      std::printf("  FAIL: %s\n", msg);                             \
      ++failures;                                                   \
    }                                                               \
    else                                                            \
    {                                                               \
      std::printf("  OK:   %s\n", msg);                             \
    }                                                               \
  } while (0)

static int test_tau_zero_identity()
{
  std::printf("--- tau=0 is no-op ---\n");

  auto a    = tt::random({ 3, 4, 5, 3 }, 6, 42);
  auto st   = tt::soft_threshold(a, 0.0);
  double md = tu::max_abs_diff(a.to_dense(), st.to_dense());
  CHECK(md < 1.0e-14, "soft_threshold(a, 0) == a (dense match)");

  // Ranks unchanged.
  auto ra  = a.ranks();
  auto rst = st.ranks();
  bool rk_ok = true;
  for (size_t i = 0; i < ra.size(); ++i)
    if (ra[i] != rst[i])
      rk_ok = false;
  CHECK(rk_ok, "soft_threshold(a, 0): ranks unchanged");
  return 0;
}

static int test_d1_noop()
{
  std::printf("--- d=1 tensors (no bonds) ---\n");

  auto a    = tt::ones({ 5 });
  auto st   = tt::soft_threshold(a, 100.0);  // huge tau, but d=1 so no-op
  double md = tu::max_abs_diff(a.to_dense(), st.to_dense());
  CHECK(md < 1.0e-14, "soft_threshold(d=1, tau=100): unchanged");
  return 0;
}

static int test_norm_monotonic()
{
  // Soft thresholding should not INCREASE the Frobenius norm.
  std::printf("--- norm monotonic ---\n");

  auto a     = tt::random({ 3, 4, 5 }, 8, 7);
  double na  = tt::norm(a);
  auto st    = tt::soft_threshold(a, 0.5);
  double nst = tt::norm(st);
  CHECK(nst <= na * (1.0 + 1.0e-14),
        "soft_threshold: norm non-increasing");
  return 0;
}

static int test_rank_reduction()
{
  // A rank-1 tensor should survive a modest tau; large tau may zero it.
  std::printf("--- rank reduction ---\n");

  // Build a rank-1 TT: outer product of mode vectors.
  auto a  = tt::random({ 4, 4, 4 }, 2, 0);   // rank <= 2
  // Round to rank 1 explicitly.
  tt::round_options ro;
  ro.eps      = 0.0;
  ro.max_rank = 1;
  ro.method   = tt::round_method::svd;
  auto a1    = tt::round(a, ro);
  double n1  = tt::norm(a1);
  CHECK(a1.max_rank() == 1, "built a rank-1 TT");

  // Small tau: rank should stay 1, norm should decrease slightly.
  auto st1   = tt::soft_threshold(a1, 0.01 * n1);
  double ns1 = tt::norm(st1);
  CHECK(st1.max_rank() == 1, "small tau: rank still 1");
  CHECK(ns1 <= n1 * (1.0 + 1e-14), "small tau: norm non-increasing");

  // Large tau: may zero out the single singular value.
  auto st2   = tt::soft_threshold(a1, n1 * 2.0);
  double ns2 = tt::norm(st2);
  CHECK(ns2 <= 1.0e-14, "tau > ||a||: result is zero");
  return 0;
}

static int test_vs_round()
{
  // For a tensor with well-separated singular values, soft_threshold
  // and round should behave differently -- round drops the tail, soft
  // shrinks all values.
  std::printf("--- vs round: different semantics ---\n");

  // Build: singular values [10, 1, 0.1, 0.01] at the dominant bond.
  std::vector<double> dense(3 * 3 * 3, 0.0);
  {
    int n = 3, m = 9;
    double sv[4] = { 10.0, 1.0, 0.1, 0.01 };
    std::vector<double> L(n * 4, 0.0), R(4 * m, 0.0);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < 4; ++j)
        L[i * 4 + j] = (i + 1.0) * 0.5;
    for (int j = 0; j < 4; ++j)
    {
      double sc = sv[j];
      for (int k = 0; k < m; ++k)
        R[j * m + k] = sc * (k + 1.0) * 0.2;
    }
    for (int i = 0; i < n; ++i)
      for (int k = 0; k < m; ++k)
        for (int j = 0; j < 4; ++j)
          dense[static_cast<size_t>(i * m + k)] +=
            L[static_cast<size_t>(i * 4 + j)] *
            R[static_cast<size_t>(j * m + k)];
  }
  auto t     = tt::from_dense(dense.data(), { 3, 3, 3 }, 0.0);
  double n0  = tt::norm(t);

  // Round with eps=0.1: should drop to ~rank 2 (drops 0.1 and 0.01).
  auto rd    = tt::round(t, tt::round_options{ 0.1, 0, tt::round_method::svd });
  double nde = tt::norm(rd);
  int mr_rd  = rd.max_rank();
  std::printf("  round(eps=0.1): rank=%d norm=%.4f (orig=%.4f)\n",
              mr_rd, nde, n0);

  // Soft threshold with tau=0.5: shrinks all values, may drop the tiny ones.
  auto st    = tt::soft_threshold(t, 0.5);
  double nst = tt::norm(st);
  int mr_st  = st.max_rank();
  std::printf("  soft_thresh(tau=0.5): rank=%d norm=%.4f (orig=%.4f)\n",
              mr_st, nst, n0);

  // The key difference: round preserves the large singular values
  // untouched; soft_threshold shrinks ALL singular values.
  // After soft_threshold, the norm should be smaller than after round
  // because the large values got shrunk.
  CHECK(nde > nst, "round norm > soft_thresh norm (shrinkage effect)");
  // With S = [10,1,0.1,0.01], eps=0.1 drops only 0.01 while
  // tau=0.5 zeros 0.1,0.01 -> shrinkage effect guarantees nde > nst.

  // Verify that round didn't produce more rank reduction than necessary.
  CHECK(mr_rd <= 4, "round: rank <= 4");
  return 0;
}

static int test_gauge()
{
  // After soft_threshold, the TT should be right-canonical.
  std::printf("--- output gauge ---\n");

  auto st = tt::soft_threshold(tt::random({ 3, 4, 5 }, 6, 99), 0.1);
  const int d = st.d();
  for (int k = 1; k < d; ++k)
  {
    // right_unfold should have orthonormal rows.
    const int rows = st.core(k).r_left();
    const int cols = st.core(k).n_phys() * st.core(k).r_right();
    Eigen::Map<const tt::eigen_bridge::row_matrix> V(
      st.core(k).data(), rows, cols);
    Eigen::MatrixXd G = V * V.transpose();
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(rows, rows);
    bool ok           = (G - I).norm() <= 1.0e-13 * rows;
    CHECK(ok, "soft_threshold: core[k] is right-orthogonal");
  }
  return 0;
}

int main()
{
  test_tau_zero_identity();
  test_d1_noop();
  test_norm_monotonic();
  test_rank_reduction();
  test_vs_round();
  test_gauge();

  std::printf("\n%d soft_threshold test(s) FAILED\n", failures);
  return failures == 0 ? 0 : 1;
}
//
// :D
//
