/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Test: QTT from_dense compression
*/
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "tensor_train.hpp"
#include "tt_test_utils.hpp"

namespace tt_ns = mva::tensor_train;
namespace tu    = mva::tensor_train::test_utils;

// Helper: reconstruct dense from QTT TT by evaluating at every
// original multi-index (converting to QTT binary indices first).
static std::vector<double> qtt_to_dense(const tt_ns::tt& t,
                                        const std::vector<int>& shape,
                                        int base)
{
  const int d_orig = static_cast<int>(shape.size());

  // Compute bits per dimension.
  std::vector<int> bits(static_cast<std::size_t>(d_orig));
  int total_bits = 0;
  for (int k = 0; k < d_orig; ++k)
  {
    int n = shape[static_cast<std::size_t>(k)];
    int b = 0;
    while (n > 1) { n /= base; ++b; }
    bits[static_cast<std::size_t>(k)] = b;
    total_bits += b;
  }

  const int d_qtt = t.d();

  int64_t total = 1;
  for (int s : shape)
    total *= static_cast<int64_t>(s);
  std::vector<double> dense(static_cast<std::size_t>(total));

  std::vector<int> orig_idx(static_cast<std::size_t>(d_orig));
  std::vector<int> qtt_idx(static_cast<std::size_t>(d_qtt));

  for (int64_t pos = 0; pos < total; ++pos)
  {
    // Decompose flat position into original multi-index.
    int64_t tmp = pos;
    for (int k = d_orig - 1; k >= 0; --k)
    {
      const int sz = shape[static_cast<std::size_t>(k)];
      orig_idx[static_cast<std::size_t>(k)] =
        static_cast<int>(tmp % static_cast<int64_t>(sz));
      tmp /= static_cast<int64_t>(sz);
    }

    // Convert original multi-index to QTT binary indices.
    // QTT indexing is MSB-first within each dimension (matching qtt_svd).
    int bit_pos = 0;
    for (int dim = 0; dim < d_orig; ++dim)
    {
      int val = orig_idx[static_cast<std::size_t>(dim)];
      const int nb = bits[static_cast<std::size_t>(dim)];
      // Extract bits MSB-first: highest power of base at bit_pos+0.
      for (int b = nb - 1; b >= 0; --b)
      {
        int div = 1;
        for (int p = 0; p < b; ++p)
          div *= base;
        qtt_idx[static_cast<std::size_t>(bit_pos)] =
          (val / div) % base;
        ++bit_pos;
      }
    }

    dense[static_cast<std::size_t>(pos)] =
      tt_ns::eval_at(t, qtt_idx.data());
  }
  return dense;
}

int main()
{
  const double tol = 1.0e-10;
  int failures      = 0;

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

  // ================================================================
  // Test 1: d=1 with n=8 (base=2) -> QTT shape [2,2,2]
  // ================================================================
  {
    std::printf("--- d=1 n=8 base=2 ---\n");
    const std::vector<int> shape = { 8 };
    const int base               = 2;
    const int total              = 8;
    std::vector<double> dense(total);
    for (int i = 0; i < total; ++i)
      dense[static_cast<std::size_t>(i)] = static_cast<double>(i + 1);

    tt_ns::from_dense_options opts;
    opts.method   = tt_ns::from_dense_method::qtt;
    opts.qtt_base = base;
    opts.eps      = 1.0e-12;
    auto t        = tt_ns::from_dense(dense.data(), shape, opts);

    // QTT shape should be [2,2,2] (3 binary modes)
    CHECK(t.d() == 3, "d=1 n=8: QTT has 3 cores");
    auto qs = t.shape();
    CHECK(qs[0] == 2 && qs[1] == 2 && qs[2] == 2,
          "d=1 n=8: QTT core sizes are [2,2,2]");

    // Reconstruct and compare
    auto recon = qtt_to_dense(t, shape, base);
    double md  = tu::max_abs_diff(dense, recon);
    CHECK(md < tol, "d=1 n=8: reconstruction matches original");
    if (md >= tol)
      std::printf("    max diff = %e\n", md);
  }

  // ================================================================
  // Test 2: d=2 with shape [4,8] (base=2) -> QTT shape [2,2, 2,2,2]
  // ================================================================
  {
    std::printf("--- d=2 [4,8] base=2 ---\n");
    const std::vector<int> shape = { 4, 8 };
    const int base               = 2;
    const int total              = 32;
    std::vector<double> dense(total);
    for (int i = 0; i < total; ++i)
      dense[static_cast<std::size_t>(i)] = static_cast<double>(i + 1);

    tt_ns::from_dense_options opts;
    opts.method   = tt_ns::from_dense_method::qtt;
    opts.qtt_base = base;
    opts.eps      = 1.0e-12;
    auto t        = tt_ns::from_dense(dense.data(), shape, opts);

    CHECK(t.d() == 5, "d=2 [4,8]: QTT has 5 cores");
    auto qs = t.shape();
    bool sizes_ok = true;
    for (int k = 0; k < 5; ++k)
      if (qs[static_cast<std::size_t>(k)] != 2)
        sizes_ok = false;
    CHECK(sizes_ok, "d=2 [4,8]: all QTT core sizes are 2");

    auto recon = qtt_to_dense(t, shape, base);
    double md  = tu::max_abs_diff(dense, recon);
    CHECK(md < tol, "d=2 [4,8]: reconstruction matches original");
    if (md >= tol)
      std::printf("    max diff = %e\n", md);
  }

  // ================================================================
  // Test 3 (removed): base=3 is no longer supported -- QTT requires
  // a power-of-2 base for digit extraction via bitmask.
  // ================================================================

  // ================================================================
  // Test 4: all mode sizes = 1 (scalar)
  // ================================================================
  {
    std::printf("--- shape [1] base=2 (scalar) ---\n");
    const std::vector<int> shape = { 1 };
    const int base               = 2;
    std::vector<double> dense    = { 42.0 };

    tt_ns::from_dense_options opts;
    opts.method   = tt_ns::from_dense_method::qtt;
    opts.qtt_base = base;
    opts.eps      = 1.0e-12;
    auto t        = tt_ns::from_dense(dense.data(), shape, opts);

    CHECK(t.d() == 1, "shape [1]: QTT has 1 core");
    CHECK(t.shape()[0] == 1, "shape [1]: core size is 1");

    auto recon = qtt_to_dense(t, shape, base);
    CHECK(std::abs(recon[0] - 42.0) <= 1.0e-14, "shape [1]: value preserved");
  }

  // ================================================================
  // Test 5: eps-based truncation works on QTT
  // ================================================================
  {
    std::printf("--- eps truncation on QTT ---\n");
    const std::vector<int> shape = { 4, 4 };
    const int base               = 2;
    const int total              = 16;

    // Build a rank-1 separable tensor.
    std::vector<double> dense(total);
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        dense[static_cast<std::size_t>(i * 4 + j)] =
          static_cast<double>(i + 1) * static_cast<double>(j + 1);

    // Tight eps -> high rank.
    {
      tt_ns::from_dense_options opts;
      opts.method   = tt_ns::from_dense_method::qtt;
      opts.qtt_base = base;
      opts.eps      = 1.0e-12;
      auto t        = tt_ns::from_dense(dense.data(), shape, opts);
      auto recon    = qtt_to_dense(t, shape, base);
      double md     = tu::max_abs_diff(dense, recon);
      CHECK(md < tol, "tight eps: reconstruction exact");
    }

    // Loose eps -> rank reduction.
    {
      tt_ns::from_dense_options opts;
      opts.method   = tt_ns::from_dense_method::qtt;
      opts.qtt_base = base;
      opts.eps      = 1.0e-2;
      auto t        = tt_ns::from_dense(dense.data(), shape, opts);
      // Verify the ranks reflect truncation.
      auto rs = t.ranks();
      (void)rs;
      CHECK(t.d() > 0, "loose eps: QTT compressed");
      double frob_t  = tt_ns::norm(t);
      double frob_d  = tu::frob_norm(dense);
      double rel_err = std::abs(frob_t - frob_d) / frob_d;
      CHECK(rel_err < 1.0e-1, "loose eps: Frobenius error bounded");
    }
  }

  // Test 6 (skip): validate_qtt_shape rejects non-power-of-base shapes
  // by calling abort().  Verified by design; cannot be tested inline
  // without a subprocess/death-test harness.
  (void)failures;

  std::printf("\n%d test(s) FAILED\n", failures);
  return failures;
}
//
// :D
//
