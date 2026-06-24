/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Test: tt_eval at_index / batch agree with to_dense
*/
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "tensor_train.hpp"

namespace tt_ns = mva::tensor_train;

int main()
{
  int failed = 0;
  std::mt19937_64 rng(11);
  std::normal_distribution<double> N(0.0, 1.0);

  const int n0 = 3, n1 = 4, n2 = 5, n3 = 3;
  const int total = n0 * n1 * n2 * n3;
  std::vector<double> T(total);
  for (auto& x : T)
    x = N(rng);

  auto A      = tt_ns::from_dense(T.data(), { n0, n1, n2, n3 }, 1.0e-12);
  auto Adense = A.to_dense();

  // Check single eval_at over all indices.
  for (int i0 = 0; i0 < n0; ++i0)
    for (int i1 = 0; i1 < n1; ++i1)
      for (int i2 = 0; i2 < n2; ++i2)
        for (int i3 = 0; i3 < n3; ++i3)
        {
          int idx[4] = { i0, i1, i2, i3 };
          double v   = tt_ns::eval_at(A, idx);
          int li     = ((i0 * n1 + i1) * n2 + i2) * n3 + i3;
          if (std::abs(v - Adense[li]) > 1.0e-10)
          {
            std::printf("FAIL: eval_at (%d,%d,%d,%d) %g vs %g\n",
                        i0,
                        i1,
                        i2,
                        i3,
                        v,
                        Adense[li]);
            ++failed;
            goto done_single;
          }
        }
done_single:

  // Batch evaluation.
  {
    const int M = 32;
    std::vector<int> idx_buf(M * 4);
    std::uniform_int_distribution<int> u0(0, n0 - 1);
    std::uniform_int_distribution<int> u1(0, n1 - 1);
    std::uniform_int_distribution<int> u2(0, n2 - 1);
    std::uniform_int_distribution<int> u3(0, n3 - 1);
    std::vector<double> ref(M);
    for (int j = 0; j < M; ++j)
    {
      int i0 = u0(rng), i1 = u1(rng), i2 = u2(rng), i3 = u3(rng);
      idx_buf[j * 4 + 0] = i0;
      idx_buf[j * 4 + 1] = i1;
      idx_buf[j * 4 + 2] = i2;
      idx_buf[j * 4 + 3] = i3;
      int li             = ((i0 * n1 + i1) * n2 + i2) * n3 + i3;
      ref[j]             = Adense[li];
    }
    auto vals = tt_ns::eval_batch(A, idx_buf.data(), M);
    for (int j = 0; j < M; ++j)
    {
      if (std::abs(vals[j] - ref[j]) > 1.0e-10)
      {
        std::printf("FAIL: batch[%d] %g vs %g\n", j, vals[j], ref[j]);
        ++failed;
        break;
      }
    }
  }

  if (failed == 0)
    std::printf("test_tt_eval: OK\n");
  return failed == 0 ? 0 : 1;
}
//
// :D
//
