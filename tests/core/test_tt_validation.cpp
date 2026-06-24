/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: eval_at with valid indices passes
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

  // eval_at with valid indices passes
  {
    std::vector<int> shape = {2, 3, 4};
    auto a                 = tt_ns::ones(shape);
    int idx[]              = {0, 1, 2};
    double v               = tt_ns::eval_at(a, idx);
    std::printf("  [eval_at valid] value=%.1f (expect 1.0)\n", v);
    if (std::abs(v - 1.0) > 1.0e-14)
    {
      std::printf("  FAIL\n");
      ++failed;
    }
  }

  // canonical_unit produces correct value at target and near-0 elsewhere
  {
    std::vector<int> shape = {3, 3, 3};
    std::vector<int> target = {1, 2, 0};
    auto a   = tt_ns::canonical_unit(shape, target);
    double v = tt_ns::eval_at(a, target.data());
    std::printf("  [canonical_unit at target] value=%.1f (expect 1.0)\n", v);
    if (std::abs(v - 1.0) > 1.0e-14)
    {
      std::printf("  FAIL\n");
      ++failed;
    }
    int off[] = {0, 0, 0};
    double w  = tt_ns::eval_at(a, off);
    std::printf("  [canonical_unit off-target] value=%.1e (expect ~0)\n", w);
    if (std::abs(w) > 1.0e-14)
    {
      std::printf("  FAIL\n");
      ++failed;
    }
  }

  // zeros has exactly norm 0
  {
    std::vector<int> shape = {4, 5, 6};
    auto z = tt_ns::zeros(shape);
    double n = tt_ns::norm(z);
    std::printf("  [zeros norm] %.1e (expect 0.0)\n", n);
    if (std::fabs(n) > 1e-300)
    {
      std::printf("  FAIL\n");
      ++failed;
    }
  }

  // d=0 empty TT consistency
  {
    tt_ns::tt empty;
    int d = empty.d();
    std::printf("  [d=0 d()] %d (expect 0)\n", d);
    if (d != 0) { std::printf("  FAIL\n"); ++failed; }

    auto ranks = empty.ranks();
    std::printf("  [d=0 ranks()] size=%zu (expect 1)\n", ranks.size());
    if (ranks.size() != 1 || ranks[0] != 1)
    { std::printf("  FAIL\n"); ++failed; }

    auto sh = empty.shape();
    std::printf("  [d=0 shape()] size=%zu (expect 0)\n", sh.size());
    if (!sh.empty()) { std::printf("  FAIL\n"); ++failed; }

    int mr = empty.max_rank();
    std::printf("  [d=0 max_rank()] %d (expect 1)\n", mr);
    if (mr != 1) { std::printf("  FAIL\n"); ++failed; }

    auto dense = empty.to_dense();
    std::printf("  [d=0 to_dense()] size=%zu val=%.1f (expect 1, 1.0)\n",
                dense.size(), dense.empty() ? 0.0 : dense[0]);
    if (dense.size() != 1 || std::abs(dense[0] - 1.0) > 1e-14)
    { std::printf("  FAIL\n"); ++failed; }

    std::size_t np = empty.num_params();
    std::printf("  [d=0 num_params()] %zu (expect 0)\n", np);
    if (np != 0) { std::printf("  FAIL\n"); ++failed; }

    double n = tt_ns::norm(empty);
    std::printf("  [d=0 norm()] %.1f (expect 0.0)\n", n);
    if (std::abs(n) > 1e-14) { std::printf("  FAIL\n"); ++failed; }
  }

  if (failed == 0)
    std::printf("\ntest_tt_validation: OK\n");
  else
    std::printf("\ntest_tt_validation: %d FAILED\n", failed);

  return failed;
}
//
// :D
//
