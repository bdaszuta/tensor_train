/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Test: tt_core basic allocate / index / data semantics
*/
#include <cstdio>
#include <vector>

#include "tensor_train.hpp"

namespace tt_ns = mva::tensor_train;

int main()
{
  int failed = 0;

  // Allocate (2, 3, 4), fill, check shape getters and (a,n,b) access.
  tt_ns::tt_core c(2, 3, 4);
  for (int a = 0; a < 2; ++a)
    for (int n = 0; n < 3; ++n)
      for (int b = 0; b < 4; ++b)
        c(a, n, b) = a * 100.0 + n * 10.0 + b;

  if (c.r_left() != 2 || c.n_phys() != 3 || c.r_right() != 4)
  {
    std::printf("FAIL: shape getters  (got %d, %d, %d)\n",
                c.r_left(),
                c.n_phys(),
                c.r_right());
    ++failed;
  }
  if (c.size() != 24)
  {
    std::printf("FAIL: size = %zu\n", c.size());
    ++failed;
  }

  // dims() must mirror the individual getters.
  {
    auto d3 = c.dims();
    if (d3[0] != 2 || d3[1] != 3 || d3[2] != 4)
    {
      std::printf("FAIL: dims (got %d, %d, %d)\n", d3[0], d3[1], d3[2]);
      ++failed;
    }
  }

  // raw_storage escape hatch must alias data().
  {
    auto& s = tt_ns::detail::raw_storage(c);
    if (s.get_data() != c.data())
    {
      std::printf("FAIL: raw_storage data ptr mismatch\n");
      ++failed;
    }
  }

  // Verify row-major (a, n, b) layout: linear = (a*n_phys + n)*r_right + b
  for (int a = 0; a < 2; ++a)
    for (int n = 0; n < 3; ++n)
      for (int b = 0; b < 4; ++b)
      {
        const int li = (a * 3 + n) * 4 + b;
        if (c.data()[li] != a * 100.0 + n * 10.0 + b)
        {
          std::printf("FAIL: layout at (%d,%d,%d) li=%d got %g\n",
                      a,
                      n,
                      b,
                      li,
                      c.data()[li]);
          ++failed;
        }
      }

  // Build a rank-1 TT of shape (2, 3) representing T[i, j] = i + 0.5 * j
  // by factoring T as outer product is impossible; instead construct
  // hand-set rank-2 that returns this exactly.
  // We just sanity-check the tt container interface here.
  std::vector<tt_ns::tt_core> cs;
  {
    tt_ns::tt_core c0(1, 2, 1);
    c0(0, 0, 0) = 1.0;
    c0(0, 1, 0) = 2.0;
    cs.push_back(std::move(c0));

    tt_ns::tt_core c1(1, 3, 1);
    c1(0, 0, 0) = 10.0;
    c1(0, 1, 0) = 20.0;
    c1(0, 2, 0) = 30.0;
    cs.push_back(std::move(c1));
  }
  tt_ns::tt T(std::move(cs));

  if (T.d() != 2)
  {
    std::printf("FAIL: tt.d != 2\n");
    ++failed;
  }
  auto sh = T.shape();
  if (sh.size() != 2 || sh[0] != 2 || sh[1] != 3)
  {
    std::printf("FAIL: shape\n");
    ++failed;
  }
  auto rs = T.ranks();
  if (rs.size() != 3 || rs[0] != 1 || rs[1] != 1 || rs[2] != 1)
  {
    std::printf("FAIL: ranks\n");
    ++failed;
  }
  if (T.max_rank() != 1)
  {
    std::printf("FAIL: max_rank\n");
    ++failed;
  }
  if (T.num_params() != 5)
  {
    std::printf("FAIL: num_params (got %zu)\n", T.num_params());
    ++failed;
  }

  // to_dense: T(i, j) = c0(0, i, 0) * c1(0, j, 0)
  auto dense = T.to_dense();
  if (dense.size() != 6)
  {
    std::printf("FAIL: dense size %zu\n", dense.size());
    ++failed;
  }
  const double ref[6] = { 10.0, 20.0, 30.0, 20.0, 40.0, 60.0 };
  for (int k = 0; k < 6; ++k)
  {
    if (std::abs(dense[k] - ref[k]) > 1e-12)
    {
      std::printf("FAIL: dense[%d] = %g, expected %g\n", k, dense[k], ref[k]);
      ++failed;
    }
  }

  if (failed == 0)
    std::printf("test_tt_core: OK\n");
  return failed == 0 ? 0 : 1;
}
//
// :D
//
