/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: AMEn-cross: two-site DMRG-cross with random enrichment (evaluates func at random indices to add new pivot candidates)
*/
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <limits>

#include "../detail/cross_utils.hpp"
#include "../detail/maxvol.hpp"
#include "../types/tt.hpp"
#include "../types/tt_eigen_bridge.hpp"

namespace mva
{
namespace tensor_train
{

struct amen_cross_options
{
  double eps                                       = 1.0e-10;
  int max_rank                                     = 0;
  int rmin                                         = 1;
  int max_sweeps                                   = 10;
  int init_rank                                    = 2;
  std::uint64_t seed                               = 0;
  int maxvol_resolve_interval                      = 16;
  int enrich_rank                                  = 4;
  bool use_qr_pivots                               = false;
  const std::vector<std::vector<int>>* seed_pivots = nullptr;
};

template <typename F>
/**
 * @brief Build a TT via AMEn-based cross-interpolation.
 *
 * Adaptive Minimum Enrichment method: generalises DMRG cross with
 * automatic rank enrichment.  Typically more robust than ALS cross
 * for functions with complex structure.
 *
 * @param func  Callable f(const int* idx) returning double.
 * @param shape Mode sizes n_0, ..., n_{d-1}.
 * @param opts  amen_cross_options (eps, max_rank, max_sweeps, ...).
 * @return TT approximant of the function.
 */
inline tt amen_cross(F&& func,
                     const std::vector<int>& shape,
                     const amen_cross_options& opts = {})
{
  const int d = static_cast<int>(shape.size());
  if (d == 0)
    return tt();

  const int init_r   = (opts.init_rank < 2) ? 2 : opts.init_rank;
  std::mt19937_64 rng(opts.seed);

  // Index sets.
  std::vector<std::vector<int>> L_idx(static_cast<std::size_t>(d));
  std::vector<std::vector<int>> R_idx(static_cast<std::size_t>(d));

  L_idx[0]                               = {};
  R_idx[static_cast<std::size_t>(d - 1)] = {};

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
    if (opts.seed_pivots)
    {
      const auto& sp   = (*opts.seed_pivots)[static_cast<std::size_t>(k)];
      const int n_seed = static_cast<int>(sp.size()) / len;
      const int n_use  = std::min(n_seed, init_r);
      for (int r = 0; r < n_use; ++r)
        for (int m = 0; m < len; ++m)
          Lk[static_cast<std::size_t>(r * len + m)] =
            sp[static_cast<std::size_t>(r * len + m)];
    }
  }
  for (int k = d - 2; k >= 0; --k)
  {
    const int len = d - 1 - k;
    auto& Rk      = R_idx[static_cast<std::size_t>(k)];
    Rk.assign(static_cast<std::size_t>(init_r * len), 0);
    for (int m = 0; m < len; ++m)
    {
      const int nm = shape[static_cast<std::size_t>(k + 1 + m)];
      std::uniform_int_distribution<int> dist(0, nm - 1);
      for (int r = 0; r < init_r; ++r)
        Rk[static_cast<std::size_t>(r * len + m)] = dist(rng);
    }
  }

  std::vector<int> prev_ranks(static_cast<std::size_t>(d - 1), 0);
  int stable_sweeps = 0;
  std::vector<double> super_buf;

  // --- Sweep loop ---
  for (int sweep = 0; sweep < opts.max_sweeps; ++sweep)
  {
    // Forward: k = 0..d-2.
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

      const int64_t rows64         = static_cast<int64_t>(rL) * n_k;
      const int64_t ncols_struct64 = static_cast<int64_t>(n_kp1) * rR;
      const int64_t cols64         = ncols_struct64 + opts.enrich_rank;
      const int64_t bsz64          = rows64 * cols64;
      if (bsz64 > std::numeric_limits<int>::max() || bsz64 < 0)
      {
        std::fprintf(stderr,
                     "amen_cross fwd: supercore size exceeds INT_MAX "
                     "(rows=%ld * cols=%ld = %ld)\n",
                     static_cast<long>(rows64),
                     static_cast<long>(cols64),
                     static_cast<long>(bsz64));
        std::abort();
      }
      const int rows         = static_cast<int>(rows64);
      const int ncols_struct = static_cast<int>(ncols_struct64);
      const int cols         = static_cast<int>(cols64);
      super_buf.resize(static_cast<std::size_t>(static_cast<int>(bsz64)),
                       0.0);
      {
        std::vector<double> tmp(static_cast<std::size_t>(rows * ncols_struct));
        detail::eval_supercore_cross(
          func, k, n_k, n_kp1, L_idx, R_idx, d, tmp.data(), rL, rR);
        for (int row = 0; row < rows; ++row)
          std::memcpy(
            super_buf.data() + static_cast<std::size_t>(row) * cols,
            tmp.data() + static_cast<std::size_t>(row) * ncols_struct,
            static_cast<std::size_t>(ncols_struct) * sizeof(double));
      }
      // Fill enrichment columns: evaluate func at random right-side
      // indices pinned to each left context a and physical index i.
      if (opts.enrich_rank > 0)
      {
        std::vector<int> idx(static_cast<std::size_t>(d));
        for (int a = 0; a < rL; ++a)
        {
          for (int m = 0; m < k; ++m)
            idx[static_cast<std::size_t>(m)] =
              L_idx[static_cast<std::size_t>(k)]
                   [static_cast<std::size_t>(a * k + m)];
          for (int i = 0; i < n_k; ++i)
          {
            idx[static_cast<std::size_t>(k)] = i;
            const int row                    = a * n_k + i;
            for (int e = 0; e < opts.enrich_rank; ++e)
            {
              for (int m = 0; m < d - k - 2; ++m)
                idx[static_cast<std::size_t>(k + 2 + m)] =
                  std::uniform_int_distribution<int>(
                    0, shape[static_cast<std::size_t>(k + 2 + m)] - 1)(rng);
              idx[static_cast<std::size_t>(k + 1)] =
                std::uniform_int_distribution<int>(
                  0, shape[static_cast<std::size_t>(k + 1)] - 1)(rng);
              super_buf[static_cast<std::size_t>(row * cols + n_kp1 * rR +
                                                 e)] = func(idx.data());
            }
          }
        }
      }

      eigen_bridge::row_matrix U;
      eigen_bridge::col_vector s;
      eigen_bridge::row_matrix Vt;
      detail::svd_thin(
        super_buf.data(), rows, cols, detail::svd_part::U_only, U, s, Vt);

      double delta = 0.0;
      if (opts.eps > 0.0)
      {
        const double nrm  = s.norm();
        const double sq_d = static_cast<double>(std::max(d - 1, 1));
        delta = (nrm > 0.0) ? opts.eps * nrm / std::sqrt(sq_d) : 0.0;
      }
      const int r_new = detail::cross_truncate_rank(s, delta, opts.max_rank, opts.rmin);

      eigen_bridge::row_matrix U_block = U.leftCols(r_new);
      std::vector<int> piv;
      if (opts.use_qr_pivots)
      {
        Eigen::Map<const Eigen::MatrixXd> M_str(
          super_buf.data(), rows, ncols_struct);
        piv = detail::qr_pivots(M_str, r_new);
      }
      else
      {
        piv =
          detail::maxvol(
            U_block, 1.05, 100, opts.maxvol_resolve_interval);
      }
      const int new_len = k + 1;
      auto& new_L       = L_idx[static_cast<std::size_t>(k + 1)];
      new_L.assign(static_cast<std::size_t>(r_new * new_len), 0);
      for (int j = 0; j < r_new; ++j)
      {
        const int pi = piv[static_cast<std::size_t>(j)];
        const int a  = pi / n_k;
        const int i  = pi % n_k;
        if (k > 0)
          for (int m = 0; m < k; ++m)
            new_L[static_cast<std::size_t>(j * new_len + m)] =
              L_idx[static_cast<std::size_t>(k)]
                   [static_cast<std::size_t>(a * k + m)];
        new_L[static_cast<std::size_t>(j * new_len + k)] = i;
      }
    }

    // Backward: k = d-2..0.
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

      const int64_t rows64 = static_cast<int64_t>(rL) * n_k + opts.enrich_rank;
      const int64_t cols64 = static_cast<int64_t>(n_kp1) * rR;
      const int64_t bsz64  = rows64 * cols64;
      if (bsz64 > std::numeric_limits<int>::max() || bsz64 < 0)
      {
        std::fprintf(stderr,
                     "amen_cross bwd: supercore size exceeds INT_MAX "
                     "(rows=%ld * cols=%ld = %ld)\n",
                     static_cast<long>(rows64),
                     static_cast<long>(cols64),
                     static_cast<long>(bsz64));
        std::abort();
      }
      const int rows = static_cast<int>(rows64);
      const int cols = static_cast<int>(cols64);
      super_buf.resize(static_cast<std::size_t>(static_cast<int>(bsz64)), 0.0);
      detail::eval_supercore_cross(
        func, k, n_k, n_kp1, L_idx, R_idx, d, super_buf.data(), rL, rR);
      // Fill enrichment rows: evaluate func at random left-side
      // indices with one random k-mode physical index per row.
      if (opts.enrich_rank > 0)
      {
        std::vector<int> idx(static_cast<std::size_t>(d));
        for (int e = 0; e < opts.enrich_rank; ++e)
        {
          for (int m = 0; m < k; ++m)
            idx[static_cast<std::size_t>(m)] =
              std::uniform_int_distribution<int>(
                0, shape[static_cast<std::size_t>(m)] - 1)(rng);
          const int i = std::uniform_int_distribution<int>(0, n_k - 1)(rng);
          idx[static_cast<std::size_t>(k)] = i;
          for (int j = 0; j < n_kp1; ++j)
          {
            idx[static_cast<std::size_t>(k + 1)] = j;
            for (int b = 0; b < rR; ++b)
            {
              if (k + 2 < d)
                for (int m = 0; m < d - k - 2; ++m)
                  idx[static_cast<std::size_t>(k + 2 + m)] =
                    R_idx[static_cast<std::size_t>(k + 1)]
                         [static_cast<std::size_t>(b * (d - k - 2) + m)];
              const int row = rL * n_k + e;
              super_buf[static_cast<std::size_t>(row * cols + j * rR + b)] =
                func(idx.data());
            }
          }
        }
      }

      eigen_bridge::row_matrix U;
      eigen_bridge::col_vector s;
      eigen_bridge::row_matrix Vt;
      detail::svd_thin(
        super_buf.data(), rows, cols, detail::svd_part::V_only, U, s, Vt);

      double delta = 0.0;
      if (opts.eps > 0.0)
      {
        const double nrm  = s.norm();
        const double sq_d = static_cast<double>(std::max(d - 1, 1));
        delta = (nrm > 0.0) ? opts.eps * nrm / std::sqrt(sq_d) : 0.0;
      }
      const int r_new = detail::cross_truncate_rank(s, delta, opts.max_rank, opts.rmin);

      eigen_bridge::row_matrix Vt_block = Vt.topRows(r_new).transpose();
      std::vector<int> piv;
      if (opts.use_qr_pivots)
      {
        // Map the structured part of super_buf as (rL*n_k, n_kp1*rR)
        // and transpose to (n_kp1*rR, rL*n_k) for column-pivot selection.
        const int struct_rows = rows - opts.enrich_rank;
        Eigen::Map<const Eigen::MatrixXd> M_buf(
          super_buf.data(), struct_rows, cols);
        piv = detail::qr_pivots(M_buf.transpose(), r_new);
      }
      else
      {
        piv =
          detail::maxvol(
            Vt_block, 1.05, 100, opts.maxvol_resolve_interval);
      }
      const int new_len = d - 1 - k;
      auto& new_R       = R_idx[static_cast<std::size_t>(k)];
      new_R.assign(static_cast<std::size_t>(r_new * new_len), 0);
      for (int j = 0; j < r_new; ++j)
      {
        const int pi = piv[static_cast<std::size_t>(j)];
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
    }

    // Check convergence (ranks stable).
    bool changed = false;
    for (int k = 0; k < d - 1; ++k)
    {
      const int rk =
        static_cast<int>(L_idx[static_cast<std::size_t>(k + 1)].size()) /
        (k + 1);
      if (rk != prev_ranks[static_cast<std::size_t>(k)])
      {
        prev_ranks[static_cast<std::size_t>(k)] = rk;
        changed                                 = true;
      }
    }
    for (int k = 1; k < d; ++k)
    {
      const int r_len = d - k - 1;
      if (r_len <= 0)
        continue;
      const int rk =
        static_cast<int>(R_idx[static_cast<std::size_t>(k)].size()) / r_len;
      if (rk != prev_ranks[static_cast<std::size_t>(k - 1)])
      {
        prev_ranks[static_cast<std::size_t>(k - 1)] = rk;
        changed                                     = true;
      }
    }
    if (!changed)
    {
      ++stable_sweeps;
      if (stable_sweeps >= 2)
        break;
    }
    else
      stable_sweeps = 0;
  }  // sweep loop

  // Reconstruction: same as DMRG-cross -- build cores from L_idx +
  // R_idx via U_block * C^{-1}.
  std::vector<tt_core> cores(static_cast<std::size_t>(d));

  int prev_r = 1;
  for (int k = 0; k < d - 1; ++k)
  {
    const int n_k   = shape[static_cast<std::size_t>(k)];
    const int n_kp1 = shape[static_cast<std::size_t>(k + 1)];
    const int rL    = prev_r;
    const int rR =
      (k + 2 == d)
        ? 1
        : static_cast<int>(R_idx[static_cast<std::size_t>(k + 1)].size()) /
            (d - k - 2);

    const int64_t rows64 = static_cast<int64_t>(rL) * n_k;
    const int64_t cols64 = static_cast<int64_t>(n_kp1) * rR;
    const int64_t bsz64  = rows64 * cols64;
    if (bsz64 > std::numeric_limits<int>::max() || bsz64 < 0)
    {
      std::fprintf(stderr,
                   "amen_cross rec: supercore size exceeds INT_MAX "
                   "(rows=%ld * cols=%ld = %ld)\n",
                   static_cast<long>(rows64),
                   static_cast<long>(cols64),
                   static_cast<long>(bsz64));
      std::abort();
    }
    const int rows = static_cast<int>(rows64);
    const int cols = static_cast<int>(cols64);
    super_buf.resize(static_cast<std::size_t>(static_cast<int>(bsz64)));
    detail::eval_supercore_cross(
      func, k, n_k, n_kp1, L_idx, R_idx, d, super_buf.data(), rL, rR);

    // Standard SVD: no enrichment in reconstruction.
    eigen_bridge::row_matrix U;
    eigen_bridge::col_vector s;
    eigen_bridge::row_matrix Vt;
    detail::svd_thin(
      super_buf.data(), rows, cols, detail::svd_part::U_only, U, s, Vt);

    double delta = 0.0;
    if (opts.eps > 0.0)
    {
      const double nrm  = s.norm();
      const double sq_d = static_cast<double>(std::max(d - 1, 1));
      delta             = (nrm > 0.0) ? opts.eps * nrm / std::sqrt(sq_d) : 0.0;
    }
    const int r_new = detail::cross_truncate_rank(s, delta, opts.max_rank, opts.rmin);

    eigen_bridge::row_matrix U_block = U.leftCols(r_new);
    std::vector<int> piv;
    if (r_new < rows)
    {
      if (opts.use_qr_pivots)
      {
        Eigen::Map<const Eigen::MatrixXd> M_map(
          super_buf.data(), rows, cols);
        piv = detail::qr_pivots(M_map, r_new);
      }
      else
      {
        piv = detail::maxvol(
          U_block, 1.05, 100, opts.maxvol_resolve_interval);
      }

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

    eigen_bridge::row_matrix core_mat;
    if (!piv.empty())
    {
      eigen_bridge::row_matrix C(r_new, r_new);
      for (int j = 0; j < r_new; ++j)
        C.row(j) = U_block.row(piv[static_cast<std::size_t>(j)]);
      if (std::abs(C.determinant()) > 1e-30)
        core_mat = U_block * C.inverse();
      else
        core_mat = U_block;
    }
    else
    {
      core_mat = U_block;
    }

    cores[static_cast<std::size_t>(k)] = tt_core(rL, n_k, r_new);
    Eigen::Map<eigen_bridge::row_matrix> cm(
      cores[static_cast<std::size_t>(k)].data(), rL * n_k, r_new);
    cm     = core_mat;
    prev_r = r_new;
  }

  // Last core: direct function evaluation.
  {
    const int k_last  = d - 1;
    const int n_last  = shape[static_cast<std::size_t>(k_last)];
    const int rL_last = (d == 1) ? 1 : prev_r;

    std::vector<double> last_buf(static_cast<std::size_t>(rL_last * n_last));
    for (int a = 0; a < rL_last; ++a)
    {
      for (int i = 0; i < n_last; ++i)
      {
        std::vector<int> idx(static_cast<std::size_t>(d));
        if (d > 1)
          for (int m = 0; m < d - 1; ++m)
            idx[static_cast<std::size_t>(m)] =
              L_idx[static_cast<std::size_t>(k_last)]
                   [static_cast<std::size_t>(a * (d - 1) + m)];
        idx[static_cast<std::size_t>(k_last)]              = i;
        last_buf[static_cast<std::size_t>(a * n_last + i)] = func(idx.data());
      }
    }
    cores[static_cast<std::size_t>(k_last)] = tt_core(rL_last, n_last, 1);
    Eigen::Map<eigen_bridge::row_matrix> lm(
      cores[static_cast<std::size_t>(k_last)].data(), rL_last * n_last, 1);
    Eigen::Map<const eigen_bridge::row_matrix> lb(
      last_buf.data(), rL_last * n_last, 1);
    lm = lb;
  }

  return tt(std::move(cores));
}

}  // namespace tensor_train
}  // namespace mva
//
// :D
//
