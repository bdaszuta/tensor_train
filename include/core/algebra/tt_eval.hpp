/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: TT evaluation at multi-indices
*/
#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../detail/core_view.hpp"
#include "../types/tt.hpp"
#include "../types/tt_eigen_bridge.hpp"

namespace mva
{
namespace tensor_train
{

/**
 * @brief Evaluate a TT at a single multi-index.
 * Multiplies the appropriate slices of each core together.
 * @param a   Input TT.
 * @param idx Array of length d with indices 0 <= idx[k] < n_k.
 * @return TT value at multi-index idx.
 */
inline double eval_at(const tt& a, const int* idx)
{
  if (idx == nullptr)
  {
    std::fprintf(stderr, "eval_at: idx pointer is null\n");
    std::abort();
  }
  const int d = a.d();
  if (d == 0)
  {
    std::fprintf(stderr, "eval_at: TT is empty (d=0)\n");
    std::abort();
  }
  for (int k = 0; k < d; ++k)
  {
    if (idx[k] < 0 || idx[k] >= a.core(k).n_phys())
    {
      std::fprintf(stderr,
                   "eval_at: idx[%d]=%d out of range [0, %d)\n",
                   k, idx[k], a.core(k).n_phys());
      std::abort();
    }
  }
  // Slice core 0 at physical index idx[0]: row of left_unfold (n, r1).
  // Running row-vector v of length r_k+1.
  const tt_core& c0 = a.core(0);
  const int r1      = c0.r_right();
  eigen_bridge::row_matrix v(1, r1);
  // c0 has r_left == 1; left_unfold is (n, r1) row-major.
  v = detail::left_unfold(c0).row(idx[0]);
  for (int k = 1; k < d; ++k)
  {
    const tt_core& ck = a.core(k);
    const int r_kp1   = ck.r_right();
    const int j       = idx[k];
    // Slice ck at j: rows {a*n_k + j : a in [0, r_k)} of left_unfold.
    // Equivalent: column block of right_unfold (r_k, n_k*r_kp1) at
    // columns [j*r_kp1, (j+1)*r_kp1).
    auto slab = detail::right_unfold(ck).middleCols(
        static_cast<Eigen::Index>(j) * r_kp1, r_kp1);
    eigen_bridge::row_matrix v_new(1, r_kp1);
    v_new.noalias() = v * slab;
    v               = std::move(v_new);
  }
  return v(0, 0);
}

/**
 * @brief Evaluate a TT at M multi-indices.
 * @param a      Input TT.
 * @param idx    Flat buffer of M*d indices, row-major (M samples x d modes).
 * @param M      Number of multi-indices.
 * @return std::vector<double> of length M with the evaluated values.
 */
inline std::vector<double> eval_batch(const tt& a, const int* idx, int M)
{
  if (idx == nullptr && M > 0)
  {
    std::fprintf(stderr,
                 "eval_batch: idx pointer is null with M=%d > 0\n", M);
    std::abort();
  }
  if (M < 0)
  {
    std::fprintf(stderr, "eval_batch: M must be non-negative (got %d)\n", M);
    std::abort();
  }
  const int d = a.d();
  if (d == 0)
  {
    std::fprintf(stderr, "eval_batch: TT is empty (d=0)\n");
    std::abort();
  }
  std::vector<double> out(static_cast<std::size_t>(M), 0.0);
  for (int j = 0; j < M; ++j)
  {
    out[static_cast<std::size_t>(j)] =
      eval_at(a, idx + static_cast<std::ptrdiff_t>(j) * d);
  }
  return out;
}

}  // namespace tensor_train
}  // namespace mva
//
// :D
//
