/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Invariant-focused tests beyond the basic correctness suite
*/
#include "tensor_train.hpp"
#include "tt_test_utils.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace tu = mva::tensor_train::test_utils;
namespace tt = mva::tensor_train;

static int failures = 0;

#define CHECK(cond, msg)                                               \
  do                                                                   \
  {                                                                    \
    if (!(cond))                                                       \
    {                                                                  \
      std::printf("  FAIL: %s\n", msg);                                \
      ++failures;                                                      \
    }                                                                  \
    else                                                               \
    {                                                                  \
      std::printf("  OK:   %s\n", msg);                                \
    }                                                                  \
  } while (0)

// ---- helpers ----

static tt::round_options opts_svd(double eps = 1.0e-10, int max_rank = 0)
{
  tt::round_options o;
  o.eps      = eps;
  o.max_rank = max_rank;
  o.method   = tt::round_method::svd;
  return o;
}

// Verify columns of left_unfold(c) are orthonormal.
static bool is_left_orthogonal(const tt::tt_core& c, double tol = 1.0e-14)
{
  const int rows = c.r_left() * c.n_phys();
  const int cols = c.r_right();
  Eigen::Map<const tt::eigen_bridge::row_matrix> U(c.data(), rows, cols);
  Eigen::MatrixXd G = U.transpose() * U;
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(cols, cols);
  return (G - I).norm() <= tol * cols;
}

// Verify rows of right_unfold(c) are orthonormal.
static bool is_right_orthogonal(const tt::tt_core& c, double tol = 1.0e-14)
{
  const int rows = c.r_left();
  const int cols = c.n_phys() * c.r_right();
  Eigen::Map<const tt::eigen_bridge::row_matrix> V(c.data(), rows, cols);
  Eigen::MatrixXd G = V * V.transpose();
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(rows, rows);
  return (G - I).norm() <= tol * rows;
}

// ---- tests ----

static int test_zero_round_trip()
{
  std::printf("--- zero tensor round-trip ---\n");
  auto z = tt::zeros({ 3, 4, 2 });
  CHECK(tt::norm(z) < 1e-300, "zeros: Frobenius norm = 0");

  auto r = tt::round(z, opts_svd(1.0e-10));
  CHECK(tt::norm(r) < 1e-300, "round(zeros): norm still 0");
  CHECK(r.ranks()[1] == 1, "round(zeros): bond ranks = 1");

  auto z2    = tt::zeros({ 3, 4, 2 });
  double ip  = tt::inner(z, z2);
  CHECK(std::fabs(ip) < 1e-300, "inner(zeros, zeros) = 0");
  return 0;
}

static int test_d1_edge_cases()
{
  std::printf("--- d=1 edge cases ---\n");

  auto ones = tt::ones({ 7 });
  auto r1   = tt::round(ones, opts_svd(1.0e-10));
  double d  = tu::max_abs_diff(ones.to_dense(), r1.to_dense());
  CHECK(d < 1.0e-15, "round(d=1) identity");

  std::vector<double> d1 = { 3.0, 1.0, 4.0, 1.0, 5.0 };
  auto t1 = tt::from_dense(d1.data(), { 5 }, 1.0e-10);
  CHECK(t1.d() == 1, "from_dense(d=1): d correct");
  double md = tu::max_abs_diff(t1.to_dense(), d1);
  CHECK(md < 1.0e-14, "from_dense(d=1): reconstruction exact");

  std::vector<double> d1n1 = { 42.0 };
  auto t2 = tt::from_dense(d1n1.data(), { 1 }, 1.0e-10);
  CHECK(t2.d() == 1, "from_dense(d=1,n=1): d correct");
  double e2 = std::abs(t2.to_dense()[0] - 42.0);
  CHECK(e2 < 1.0e-14, "from_dense(d=1,n=1): value correct");
  CHECK(std::abs(tt::round(t2, opts_svd(1.0e-10)).to_dense()[0] - 42.0) <= 1.0e-14,
        "round(d=1,n=1) preserves value");
  return 0;
}

static int test_mode_size_1_tensors()
{
  // Mode-size-1 dimensions don't force rank 1 -- the tensor can have
  // nontrivial rank structure across the non-trivial modes.  We just
  // verify the round-trip reconstruction is accurate and round
  // doesn't blow up.
  std::printf("--- mode-size-1 tensors ---\n");
  std::vector<int> shp = { 1, 5, 1, 3, 1 };
  int total = 15;
  std::vector<double> buf(static_cast<std::size_t>(total));
  for (int i = 0; i < total; ++i)
    buf[static_cast<std::size_t>(i)] = i + 1.0;

  auto t = tt::from_dense(buf.data(), shp, 1.0e-10);
  double md = tu::max_abs_diff(t.to_dense(), buf);
  CHECK(md < 2.0e-14, "from_dense with n=1 modes: reconstruct matches");

  // Round should preserve the reconstruction.
  auto r = tt::round(t, opts_svd(1.0e-10));
  double md2 = tu::max_abs_diff(r.to_dense(), buf);
  CHECK(md2 < 2.0e-14, "round(mode-size-1): reconstruct matches");

  // hadamard with ones should preserve the TT.
  auto t2 = tt::ones(shp);
  auto h  = tt::hadamard(t, t2);
  // After hadamard, ranks may grow (ranks multiply).  Round to compress.
  auto hr = tt::round(h, opts_svd(1.0e-10));
  double mh = tu::max_abs_diff(hr.to_dense(), buf);
  CHECK(mh < 2.0e-14, "hadamard then round with n=1 modes: matches");
  return 0;
}

static int test_gauge_after_round()
{
  std::printf("--- gauge after round ---\n");
  auto rng = tt::random({ 3, 4, 5, 3 }, 6, 42);
  auto r   = tt::round(rng, opts_svd(1.0e-10));
  const int d = r.d();
  for (int k = 0; k < d - 1; ++k)
    CHECK(is_left_orthogonal(r.core(k)), "round: core[k] is left-orthogonal");
  return 0;
}

static int test_gauge_after_orthogonalize()
{
  std::printf("--- gauge after right_orthogonalize ---\n");

  auto a = tt::random({ 3, 4, 5, 3 }, 6, 99);
  tt::right_orthogonalize(a);
  const int d = a.d();
  for (int k = 1; k < d; ++k)
    CHECK(is_right_orthogonal(a.core(k)),
          "right_orthogonalize: core[k] is right-orthogonal");

  // Norm invariance: orthogonalization preserves Frobenius norm.
  auto b = tt::random({ 3, 4, 5, 3 }, 6, 99);
  double n1 = tt::norm(b);
  tt::right_orthogonalize(b);
  double n2 = tt::norm(b);
  CHECK(std::abs(n1 - n2) <= 1.0e-14 * (std::abs(n1) + 1.0),
        "right_orthogonalize: norm preserved");
  return 0;
}

static int test_round_norm_monotonic()
{
  std::printf("--- round norm monotonic ---\n");

  auto a = tt::random({ 2, 3, 4, 2 }, 8, 7);
  double n1 = tt::norm(a);
  auto r    = tt::round(a, opts_svd(1.0e-2));
  double n2 = tt::norm(r);
  CHECK(n2 <= n1 * (1.0 + 1.0e-14),
        "round: norm non-increasing (eps=1e-2)");

  auto ro = opts_svd(0.0, 3);
  auto r2 = tt::round(a, ro);
  double n3 = tt::norm(r2);
  CHECK(n3 <= n1 * (1.0 + 1.0e-14),
        "round(max_rank=3): norm non-increasing");
  return 0;
}

static int test_add_round_consistency()
{
  std::printf("--- add + round consistency ---\n");

  auto a  = tt::random({ 2, 3, 4 }, 4, 0);
  auto b  = tt::random({ 2, 3, 4 }, 4, 1);
  auto s  = tt::add(a, b);
  auto sr = tt::round(s, opts_svd(1.0e-10));

  auto da = a.to_dense();
  auto db = b.to_dense();
  for (std::size_t i = 0; i < da.size(); ++i)
    da[i] += db[i];
  double md = tu::max_abs_diff(sr.to_dense(), da);
  CHECK(md < 1.0e-13, "round(add(a,b)) == a+b dense");
  return 0;
}

static int test_hadamard_idempotent()
{
  std::printf("--- hadamard idempotent ---\n");

  auto a  = tt::random({ 3, 4, 3 }, 5, 42);
  auto o  = tt::ones({ 3, 4, 3 });
  auto ha = tt::hadamard(a, o);
  double md = tu::max_abs_diff(ha.to_dense(), a.to_dense());
  CHECK(md < 1.0e-14, "hadamard(a, ones) == a");
  return 0;
}

static int test_axpby_identity()
{
  std::printf("--- axpby identity ---\n");

  auto a = tt::random({ 2, 3, 2 }, 4, 0);
  auto b = tt::random({ 2, 3, 2 }, 4, 1);
  double alpha = 2.5, beta = -0.7;
  auto ab  = tt::axpby(alpha, a, beta, b);
  auto abr = tt::round(ab, opts_svd(1.0e-10));

  auto da = a.to_dense();
  auto db = b.to_dense();
  std::vector<double> ref(da.size());
  for (std::size_t i = 0; i < ref.size(); ++i)
    ref[i] = alpha * da[i] + beta * db[i];

  double md = tu::max_abs_diff(abr.to_dense(), ref);
  CHECK(md < 1.0e-13, "round(axpby(alpha,a,beta,b)) == ref");
  return 0;
}

static int test_round_idempotence()
{
  std::printf("--- round idempotence ---\n");

  auto a  = tt::random({ 2, 3, 4, 2, 3 }, 6, 123);
  auto r1 = tt::round(a, opts_svd(1.0e-6));
  auto r2 = tt::round(r1, opts_svd(1.0e-6));
  double diff = tu::max_abs_diff(r1.to_dense(), r2.to_dense());
  double nrm  = tu::frob_norm(r1.to_dense());
  CHECK(diff <= 1.0e-14 * (nrm + 1.0),
        "round idempotence: round(round(a)) == round(a)");
  return 0;
}

static int test_svd_error_bound()
{
  std::printf("--- SVD delta error bound ---\n");

  auto a = tt::random({ 2, 3, 4, 2 }, 8, 99);
  double na  = tt::norm(a);
  double eps = 0.01;
  auto ar    = tt::round(a, opts_svd(eps));
  double err = tt::norm(tt::sub(a, ar));
  CHECK(err <= eps * na * 1.01,
        "round error bound: ||a - round(a)|| <= eps * ||a||");
  return 0;
}

static int test_identity_round_trip()
{
  // Construct a TT from dense via SVD, round it, and verify the
  // dense reconstruction matches the original.
  std::printf("--- identity round-trip ---\n");

  std::vector<int> shp = { 4, 3, 5 };
  auto dense = tu::make_dense(shp, 42);
  auto t     = tt::from_dense(dense.data(), shp, 1.0e-10);
  auto r     = tt::round(t, opts_svd(1.0e-10));
  double md  = tu::max_abs_diff(r.to_dense(), dense);
  CHECK(md < 1.0e-13,
        "from_dense -> round -> to_dense matches original dense");
  return 0;
}

int main()
{
  test_zero_round_trip();
  test_d1_edge_cases();
  test_mode_size_1_tensors();
  test_gauge_after_round();
  test_gauge_after_orthogonalize();
  test_round_norm_monotonic();
  test_add_round_consistency();
  test_hadamard_idempotent();
  test_axpby_identity();
  test_round_idempotence();
  test_svd_error_bound();
  test_identity_round_trip();

  std::printf("\n%d invariant test(s) FAILED\n", failures);
  return failures == 0 ? 0 : 1;
}
//
// :D
//
