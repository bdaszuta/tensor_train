/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Reference: sin-product function on [3,3,3,3]
*/
#include <cmath>
#include <cstdio>
#include <vector>

#include "tensor_train.hpp"
#include "tt_test_utils.hpp"

namespace tt_ns = mva::tensor_train;
namespace tu    = mva::tensor_train::test_utils;

int main()
{
  int failed = 0;

  // Reference: sin-product function on [3,3,3,3].
  std::vector<int> shape = {3, 3, 3, 3};
  int total             = 1;
  for (int s : shape)
    total *= s;

  auto func = [](const int* idx) -> double
  {
    double v = 1.0;
    for (int k = 0; k < 4; ++k)
      v *= std::sin(static_cast<double>(idx[k] + 1));
    return v;
  };

  // Build dense reference via from_dense.
  std::vector<double> dense(total);
  std::vector<int> idx(4);
  int pos = 0;
  for (int i0 = 0; i0 < shape[0]; ++i0)
  {
    idx[0] = i0;
    for (int i1 = 0; i1 < shape[1]; ++i1)
    {
      idx[1] = i1;
      for (int i2 = 0; i2 < shape[2]; ++i2)
      {
        idx[2] = i2;
        for (int i3 = 0; i3 < shape[3]; ++i3)
        {
          idx[3]   = i3;
          dense[pos++] = func(idx.data());
        }
      }
    }
  }
  auto ref = tt_ns::from_dense(dense.data(), shape, 1.0e-12);

  // --- dmrg_cross ---
  {
    tt_ns::dmrg_cross_options opts;
    opts.eps        = 1.0e-8;
    opts.max_rank   = 16;
    opts.max_sweeps = 5;
    opts.init_rank  = 4;
    opts.seed       = 42;
    auto a          = tt_ns::dmrg_cross(func, shape, opts);
    double err      = tu::max_abs_diff(a.to_dense(), ref.to_dense());
    std::printf("  [dmrg_cross standalone] sup-err = %.3e\n", err);
    if (err > 1.0e-6)
    {
      std::printf("  FAIL\n");
      ++failed;
    }
  }

  // --- als_cross ---
  {
    tt_ns::als_cross_options opts;
    opts.eps        = 1.0e-8;
    opts.max_rank   = 16;
    opts.max_sweeps = 10;
    opts.init_rank  = 4;
    opts.seed       = 42;
    auto a          = tt_ns::als_cross(func, shape, opts);
    double err      = tu::max_abs_diff(a.to_dense(), ref.to_dense());
    std::printf("  [als_cross standalone] sup-err = %.3e\n", err);
    if (err > 1.0e-6)
    {
      std::printf("  FAIL\n");
      ++failed;
    }
  }

  // --- amen_cross ---
  {
    tt_ns::amen_cross_options opts;
    opts.eps        = 1.0e-8;
    opts.max_rank   = 16;
    opts.max_sweeps = 5;
    opts.init_rank  = 4;
    opts.seed       = 42;
    auto a          = tt_ns::amen_cross(func, shape, opts);
    double err      = tu::max_abs_diff(a.to_dense(), ref.to_dense());
    std::printf("  [amen_cross standalone] sup-err = %.3e\n", err);
    if (err > 1.0e-6)
    {
      std::printf("  FAIL\n");
      ++failed;
    }
  }

  // --- tt_cross ---
  {
    tt_ns::tt_cross_options opts;
    opts.max_rank  = 16;
    opts.init_rank = 4;
    opts.seed      = 42;
    auto a         = tt_ns::tt_cross(func, shape, opts);
    double err     = tu::max_abs_diff(a.to_dense(), ref.to_dense());
    std::printf("  [tt_cross standalone] sup-err = %.3e\n", err);
    if (err > 1.0e-6)
    {
      std::printf("  FAIL\n");
      ++failed;
    }
  }

  // --- inner() vs dense dot product ---
  {
    auto a = tt_ns::random({3, 3, 3, 3}, 8, 12345);
    auto b = tt_ns::random({3, 3, 3, 3}, 8, 67890);
    double tt_inner = tt_ns::inner(a, b);
    auto ad = a.to_dense();
    auto bd = b.to_dense();
    double dense_dot = 0.0;
    for (std::size_t i = 0; i < ad.size(); ++i)
      dense_dot += ad[i] * bd[i];
    double diff = std::abs(tt_inner - dense_dot);
    std::printf("  [inner vs dense] diff = %.3e\n", diff);
    if (diff > 1.0e-12)
    {
      std::printf("  FAIL\n");
      ++failed;
    }
  }

  if (failed == 0)
    std::printf("\ntest_tt_cross_direct: OK\n");
  else
    std::printf("\ntest_tt_cross_direct: %d FAILED\n", failed);

  return failed;
}
//
// :D
//
