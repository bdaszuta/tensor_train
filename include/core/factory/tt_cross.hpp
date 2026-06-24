/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: TT-cross interpolation (two-pass maxvol). Builds a TT from function evaluations without materialising the full dense buffer. Uses thin-SVD for pivot selection on sub-blocks.
*/
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include <limits>

#include "../detail/cross_utils.hpp"
#include "../detail/maxvol.hpp"
#include "../types/tt.hpp"
#include "../types/tt_core.hpp"
#include "../types/tt_eigen_bridge.hpp"

namespace mva
{
namespace tensor_train
{

struct tt_cross_options
{
  int max_rank                                     = 0;
  int init_rank                                    = 2;
  std::uint64_t seed                               = 0;
  int maxvol_resolve_interval                      = 16;
  const std::vector<std::vector<int>>* seed_pivots = nullptr;
};

template <typename F>
/**
 * @brief Build a TT via TT-cross interpolation.
 *
 * Classical Oseledets TT-cross method: sweeps left-to-right then
 * right-to-left, selecting maxvol pivots at each step to
 * approximate the function from a small number of evaluations.
 *
 * @param func  Callable f(const int* idx) returning double.
 * @param shape Mode sizes n_0, ..., n_{d-1}.
 * @param opts  tt_cross_options (max_rank, init_rank, seed,
 *              maxvol_resolve_interval).
 * @return TT approximant of the function.
 */
inline tt tt_cross(F&& func,
                   const std::vector<int>& shape,
                   const tt_cross_options& opts = {})
{
  using namespace detail;
  const int d = static_cast<int>(shape.size());
  if (d == 0)
    return tt();

  const int r_target = std::max(1, (opts.max_rank > 0) ? opts.max_rank : opts.init_rank);
  const int init_r   = std::max(2, opts.init_rank);
  std::mt19937_64 rng(opts.seed);

  std::vector<std::vector<int>> L_idx(static_cast<std::size_t>(d));
  std::vector<std::vector<int>> R_idx(static_cast<std::size_t>(d));
  L_idx[0]                               = {};
  R_idx[static_cast<std::size_t>(d - 1)] = {};

  // Random initial left/right index sets.
  for (int k = 1; k < d; ++k)
  {
    const int len = k;
    auto& Lk      = L_idx[static_cast<std::size_t>(k)];
    Lk.assign(static_cast<std::size_t>(init_r * len), 0);
    for (int m = 0; m < len; ++m)
    {
      const int nm = shape[m];
      std::uniform_int_distribution<int> dist(0, nm - 1);
      for (int r = 0; r < init_r; ++r)
        Lk[static_cast<std::size_t>(r * len + m)] = dist(rng);
    }
  }
  for (int k = d - 2; k >= 0; --k)
  {
    const int len = d - 1 - k;
    auto& Rk      = R_idx[static_cast<std::size_t>(k)];
    Rk.assign(static_cast<std::size_t>(init_r * len), 0);
    for (int m = 0; m < len; ++m)
    {
      const int nm = shape[k + 1 + m];
      std::uniform_int_distribution<int> dist(0, nm - 1);
      for (int r = 0; r < init_r; ++r)
        Rk[static_cast<std::size_t>(r * len + m)] = dist(rng);
    }
  }

  std::vector<double> super_buf;

  // --- Pass 1: forward (k = 0..d-2) -- SVD + maxvol on U. ---
  for (int k = 0; k < d - 1; ++k)
  {
    const int n_k   = shape[static_cast<std::size_t>(k)];
    const int n_kp1 = shape[static_cast<std::size_t>(k + 1)];

    const int rL =
      (k == 0)
        ? 1
        : static_cast<int>(L_idx[static_cast<std::size_t>(k)].size()) / k;
    const int rR =
      (k + 2 == d)
        ? 1
        : static_cast<int>(R_idx[static_cast<std::size_t>(k + 1)].size()) /
            (d - k - 2);

    const int64_t M64 = static_cast<int64_t>(rL) * n_k * n_kp1 * rR;
    if (M64 > std::numeric_limits<int>::max() || M64 < 0)
    {
      std::fprintf(stderr,
                   "tt_cross: supercore size exceeds INT_MAX "
                   "(rL=%d * n_k=%d * n_kp1=%d * rR=%d = %ld)\n",
                   rL, n_k, n_kp1, rR, static_cast<long>(M64));
      std::abort();
    }
    const int M = static_cast<int>(M64);
    if (static_cast<int>(super_buf.size()) < M)
      super_buf.resize(static_cast<std::size_t>(M));

    detail::eval_supercore_cross(
      func, k, n_k, n_kp1, L_idx, R_idx, d, super_buf.data(), rL, rR);

    const int rows = rL * n_k;
    const int cols = n_kp1 * rR;

    eigen_bridge::row_matrix U;
    eigen_bridge::col_vector s;
    eigen_bridge::row_matrix Vt;
    detail::svd_thin(
      super_buf.data(), rows, cols, detail::svd_part::U_only, U, s, Vt);

    const int r_new         = std::min(r_target, std::min(rows, cols));
    eigen_bridge::row_matrix U_block = U.leftCols(r_new);
    auto piv = maxvol(U_block, 1.05, 100, opts.maxvol_resolve_interval);

    const int new_len = k + 1;
    auto& new_L       = L_idx[static_cast<std::size_t>(k + 1)];
    new_L.assign(static_cast<std::size_t>(r_new * new_len), 0);
    if (k > 0)
    {
      const auto& Lk = L_idx[static_cast<std::size_t>(k)];
      for (int j = 0; j < r_new; ++j)
      {
        const int pi = piv[static_cast<std::size_t>(j)];
        const int a  = pi / n_k;
        for (int m = 0; m < k; ++m)
          new_L[static_cast<std::size_t>(j * new_len + m)] =
            Lk[static_cast<std::size_t>(a * k + m)];
      }
    }
    for (int j = 0; j < r_new; ++j)
    {
      const int pi = piv[static_cast<std::size_t>(j)];
      new_L[static_cast<std::size_t>(j * new_len + k)] = pi % n_k;
    }
  }

  // --- Pass 2: backward (k = d-2..0) -- SVD + maxvol, reconstruct cores. ---
  std::vector<tt_core> cores(static_cast<std::size_t>(d));

  for (int k = d - 2; k >= 0; --k)
  {
    const int n_k   = shape[static_cast<std::size_t>(k)];
    const int n_kp1 = shape[static_cast<std::size_t>(k + 1)];

    const int rL =
      (k == 0)
        ? 1
        : static_cast<int>(L_idx[static_cast<std::size_t>(k)].size()) / k;
    const int rR =
      (k + 2 == d)
        ? 1
        : static_cast<int>(R_idx[static_cast<std::size_t>(k + 1)].size()) /
            (d - k - 2);

    const int64_t M64 = static_cast<int64_t>(rL) * n_k * n_kp1 * rR;
    if (M64 > std::numeric_limits<int>::max() || M64 < 0)
    {
      std::fprintf(stderr,
                   "tt_cross: supercore size exceeds INT_MAX "
                   "(rL=%d * n_k=%d * n_kp1=%d * rR=%d = %ld)\n",
                   rL, n_k, n_kp1, rR, static_cast<long>(M64));
      std::abort();
    }
    const int M = static_cast<int>(M64);
    if (static_cast<int>(super_buf.size()) < M)
      super_buf.resize(static_cast<std::size_t>(M));

    detail::eval_supercore_cross(
      func, k, n_k, n_kp1, L_idx, R_idx, d, super_buf.data(), rL, rR);

    const int rows = rL * n_k;
    const int cols = n_kp1 * rR;

    eigen_bridge::row_matrix U;
    eigen_bridge::col_vector s;
    eigen_bridge::row_matrix Vt;
    detail::svd_thin(
      super_buf.data(), rows, cols, detail::svd_part::full, U, s, Vt);

    const int r_new          = std::min(r_target, std::min(rows, cols));
    eigen_bridge::row_matrix U_block  = U.leftCols(r_new);
    eigen_bridge::row_matrix Vt_block = Vt.topRows(r_new).transpose();

    auto piv_rows = maxvol(U_block, 1.05, 100, opts.maxvol_resolve_interval);
    auto piv_cols = maxvol(Vt_block, 1.05, 100, opts.maxvol_resolve_interval);

    // Build right index set from Vt_block pivots.
    const int new_len = d - 1 - k;
    auto& new_R       = R_idx[static_cast<std::size_t>(k)];
    new_R.assign(static_cast<std::size_t>(r_new * new_len), 0);
    for (int j = 0; j < r_new; ++j)
    {
      const int pi = piv_cols[static_cast<std::size_t>(j)];
      new_R[static_cast<std::size_t>(j * new_len)] = pi / rR;
      if (k + 2 < d)
      {
        const auto& Rkp      = R_idx[static_cast<std::size_t>(k + 1)];
        const int suffix_len = d - k - 2;
        for (int m = 0; m < suffix_len; ++m)
          new_R[static_cast<std::size_t>(j * new_len + 1 + m)] =
            Rkp[static_cast<std::size_t>((pi % rR) * suffix_len + m)];
      }
    }

    // Core = U_block * C^{-1} where C = U_block[piv_rows, :].
    // Fall back to U_block if C is ill-conditioned.
    Eigen::MatrixXd core_mat;
    {
      Eigen::MatrixXd C(r_new, r_new);
      for (int j = 0; j < r_new; ++j)
        C.row(j) = U_block.row(piv_rows[static_cast<std::size_t>(j)]);
      core_mat =
        std::abs(C.determinant()) > 1e-30 ? U_block * C.inverse() : U_block;
    }

    cores[static_cast<std::size_t>(k)] = tt_core(rL, n_k, r_new);
    Eigen::Map<eigen_bridge::row_matrix> cm(
      cores[static_cast<std::size_t>(k)].data(), rL * n_k, r_new);
    cm = core_mat;
  }

  // --- Last core (k = d-1): direct evaluation. ---
  {
    const int k      = d - 1;
    const int n_last = shape[static_cast<std::size_t>(k)];
    const int rL =
      (d == 1)
        ? 1
        : static_cast<int>(L_idx[static_cast<std::size_t>(k)].size()) / k;

    std::vector<double> last_buf(static_cast<std::size_t>(rL * n_last));
    for (int a = 0; a < rL; ++a)
    {
      for (int i = 0; i < n_last; ++i)
      {
        std::vector<int> idx2(static_cast<std::size_t>(d));
        if (d > 1)
        {
          const auto& Lk = L_idx[static_cast<std::size_t>(k)];
          for (int m = 0; m < k; ++m)
            idx2[m] = Lk[static_cast<std::size_t>(a * k + m)];
        }
        idx2[k]                                            = i;
        last_buf[static_cast<std::size_t>(a * n_last + i)] = func(idx2.data());
      }
    }

    cores[static_cast<std::size_t>(k)] = tt_core(rL, n_last, 1);
    Eigen::Map<eigen_bridge::row_matrix> lm(
      cores[static_cast<std::size_t>(k)].data(), rL, n_last);
    lm =
      Eigen::Map<const eigen_bridge::row_matrix>(last_buf.data(), rL, n_last);
  }

  return tt(std::move(cores));
}

}  // namespace tensor_train
}  // namespace mva
//
// :D
//
