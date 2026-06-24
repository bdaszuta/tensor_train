/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: ALS-cross: single-site cross approximation for TT construction
*/
#pragma once

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include "../detail/cross_utils.hpp"
#include "../detail/maxvol.hpp"
#include "../types/tt.hpp"
#include "../types/tt_eigen_bridge.hpp"

namespace mva
{
namespace tensor_train
{

struct als_cross_options
{
  double eps                  = 1.0e-10;
  int max_rank                = 0;
  int rmin                    = 1;
  int max_sweeps              = 20;
  int init_rank               = 0;  // 0 = default to max_rank/2
  std::uint64_t seed          = 0;
  int maxvol_resolve_interval = 16;
  int enrich_rank             = 0;
  bool use_qr_pivots          = false;
  const std::vector<std::vector<int>>* seed_pivots = nullptr;
};

namespace detail
{

template <typename F>
inline void eval_als_core(F&& func,
                          int k,
                          int d,
                          const std::vector<std::vector<int>>& L_idx,
                          const std::vector<std::vector<int>>& R_idx,
                          const std::vector<int>& shape,
                          eigen_bridge::row_matrix& M,
                          int rL,
                          int rR)
{
  const int n_k = shape[static_cast<std::size_t>(k)];
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
      for (int b = 0; b < rR; ++b)
      {
        for (int m = 0; k + 1 + m < d; ++m)
          idx[static_cast<std::size_t>(k + 1 + m)] =
            R_idx[static_cast<std::size_t>(k + 1)]
                 [static_cast<std::size_t>(b * (d - k - 1) + m)];
        M(a * n_k + i, b) = func(idx.data());
      }
    }
  }
}

}  // namespace detail

template <typename F>
/**
 * @brief Build a TT via ALS-based cross-interpolation.
 *
 * Cross-interpolation: evaluates func at adaptively-chosen pivots,
 * forms a matrix M, applies truncated SVD to select new pivots via
 * maxvol on the left singular vectors, then reconstructs cores by
 * interpolation (core = U_block * C^{-1}).  Repeats in alternating
 * sweeps, updating one core at a time.
 *
 * @param func  Callable f(const int* idx) returning double.
 * @param shape Mode sizes n_0, ..., n_{d-1}.
 * @param opts  als_cross_options (eps, max_rank, max_sweeps, seed, ...).
 * @return TT approximant of the function.
 * @note  Maxvol pivots are periodically re-solved to improve stability.
 */
inline tt als_cross(F&& func,
                    const std::vector<int>& shape,
                    const als_cross_options& opts = {})
{
  const int d = static_cast<int>(shape.size());
  if (d == 0)
    return tt();

  const int init_r =
    (opts.init_rank > 0) ? opts.init_rank : std::max(2, opts.max_rank / 2);
  std::mt19937_64 rng(opts.seed);

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
    const int len = d - k;
    auto& Rk      = R_idx[static_cast<std::size_t>(k)];
    Rk.assign(static_cast<std::size_t>(init_r * len), 0);
    for (int m = 0; m < len; ++m)
    {
      const int nm = shape[static_cast<std::size_t>(k + m)];
      std::uniform_int_distribution<int> dist(0, nm - 1);
      for (int r = 0; r < init_r; ++r)
        Rk[static_cast<std::size_t>(r * len + m)] = dist(rng);
    }
  }
  {
    auto& Rd = R_idx[static_cast<std::size_t>(d - 1)];
    Rd.assign(static_cast<std::size_t>(init_r), 0);
    std::uniform_int_distribution<int> dist(
      0, shape[static_cast<std::size_t>(d - 1)] - 1);
    for (int r = 0; r < init_r; ++r)
      Rd[static_cast<std::size_t>(r)] = dist(rng);
  }

  std::vector<int> prev_ranks(static_cast<std::size_t>(d - 1), 0);
  int stable_sweeps = 0;
  std::vector<tt_core> cores(static_cast<std::size_t>(d));

  for (int sweep = 0; sweep < opts.max_sweeps; ++sweep)
  {
    // --- Forward: k = 0..d-1 ---
    for (int k = 0; k < d; ++k)
    {
      const int n_k = shape[static_cast<std::size_t>(k)];
      const int rL =
        (k == 0)
          ? 1
          : static_cast<int>(L_idx[static_cast<std::size_t>(k)].size()) / k;
      const int rR =
        (k + 1 == d)
          ? 1
          : static_cast<int>(R_idx[static_cast<std::size_t>(k + 1)].size()) /
              (d - k - 1);

      eigen_bridge::row_matrix M(static_cast<Eigen::Index>(rL) * n_k, rR);
      detail::eval_als_core(func, k, d, L_idx, R_idx, shape, M, rL, rR);

      eigen_bridge::row_matrix U;
      eigen_bridge::col_vector s;
      eigen_bridge::row_matrix Vt;
      int r_new = 1;
      if (k < d - 1)
      {
        detail::svd_thin(M.data(),
                         static_cast<int>(M.rows()),
                         static_cast<int>(M.cols()),
                         detail::svd_part::U_only,
                         U,
                         s,
                         Vt);

        double delta = 0.0;
        if (opts.eps > 0.0)
        {
          const double nrm  = s.norm();
          const double sq_d = static_cast<double>(std::max(d - 1, 1));
          delta = (nrm > 0.0) ? opts.eps * nrm / std::sqrt(sq_d) : 0.0;
        }
        r_new = detail::cross_truncate_rank(s, delta, opts.max_rank, opts.rmin);
      }

      eigen_bridge::row_matrix core_mat;
      if (k == d - 1)
      {
        core_mat = M;
      }
      else
      {
        eigen_bridge::row_matrix U_block = U.leftCols(r_new);
        std::vector<int> piv;
        if (r_new < M.rows())
        {
          if (opts.use_qr_pivots)
          {
            Eigen::Map<const Eigen::MatrixXd> M_map(
              M.data(), M.rows(), M.cols());
            piv = detail::qr_pivots(M_map, r_new);
          }
          else
          {
            piv =
              detail::maxvol(
                U_block, 1.05, 100, opts.maxvol_resolve_interval);
          }
          if (k + 1 < d)
          {
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
        }

        if (!piv.empty())
        {
          eigen_bridge::row_matrix C(r_new, r_new);
          for (int j = 0; j < r_new; ++j)
            C.row(j) = U_block.row(piv[static_cast<std::size_t>(j)]);
          core_mat = std::abs(C.determinant()) > 1e-30 ? U_block * C.inverse()
                                                       : U_block;
        }
        else
        {
          core_mat = U_block;
        }
      }

      cores[static_cast<std::size_t>(k)] = tt_core(rL, n_k, r_new);
      Eigen::Map<eigen_bridge::row_matrix> cm(
        cores[static_cast<std::size_t>(k)].data(), rL * n_k, r_new);
      cm = core_mat;
    }

    // --- Backward: k = d-1..0 ---
    for (int k = d - 1; k >= 0; --k)
    {
      const int n_k = shape[static_cast<std::size_t>(k)];
      const int rL =
        (k == 0)
          ? 1
          : static_cast<int>(L_idx[static_cast<std::size_t>(k)].size()) / k;
      const int rR =
        (k + 1 == d)
          ? 1
          : static_cast<int>(R_idx[static_cast<std::size_t>(k + 1)].size()) /
              (d - k - 1);

      eigen_bridge::row_matrix M(static_cast<Eigen::Index>(rL) * n_k, rR);
      detail::eval_als_core(func, k, d, L_idx, R_idx, shape, M, rL, rR);

      eigen_bridge::row_matrix U;
      eigen_bridge::col_vector s;
      eigen_bridge::row_matrix Vt;
      detail::svd_thin(M.data(),
                       static_cast<int>(M.rows()),
                       static_cast<int>(M.cols()),
                       detail::svd_part::U_only,
                       U,
                       s,
                       Vt);

      double delta = 0.0;
      if (opts.eps > 0.0)
      {
        const double nrm  = s.norm();
        const double sq_d = static_cast<double>(std::max(d - 1, 1));
        delta = (nrm > 0.0) ? opts.eps * nrm / std::sqrt(sq_d) : 0.0;
      }
      const int r_new = detail::cross_truncate_rank(s, delta, opts.max_rank, opts.rmin);

      // Note: R_idx is NOT updated in the backward sweep.
      // The Vt_block from single-site SVD has shape (rR, r_new)
      // whose row indices are right-rank indices only (no physical
      // dimension).  Unlike dmrg_cross where Vt_block rows carry
      // (physical, rank) pairs, here the physical index must come
      // from U_block pivots and be paired with right-rank pivots
      // through the SVD -- which admits no natural 1:1 pairing.
      // Omitting the R_idx update is safe: L_idx refinement in the
      // forward sweep plus core rebuild here is sufficient for
      // convergence in the single-site case.

      // Rebuild core.
      {
        eigen_bridge::row_matrix U_block = U.leftCols(r_new);
        eigen_bridge::row_matrix core_mat;
        if (k == d - 1)
        {
          core_mat = M;
        }
        else if (r_new < M.rows())
        {
          std::vector<int> piv_u;
          if (opts.use_qr_pivots)
          {
            Eigen::Map<const Eigen::MatrixXd> M_map(
              M.data(), M.rows(), M.cols());
            piv_u = detail::qr_pivots(M_map, r_new);
          }
          else
          {
            piv_u =
              detail::maxvol(
                U_block, 1.05, 100, opts.maxvol_resolve_interval);
          }
          eigen_bridge::row_matrix C(r_new, r_new);
          for (int j = 0; j < r_new; ++j)
            C.row(j) = U_block.row(piv_u[static_cast<std::size_t>(j)]);
          core_mat = std::abs(C.determinant()) > 1e-30 ? U_block * C.inverse()
                                                       : U_block;
        }
        else
        {
          core_mat = U_block;
        }
        cores[static_cast<std::size_t>(k)] = tt_core(rL, n_k, r_new);
        Eigen::Map<eigen_bridge::row_matrix> cm(
          cores[static_cast<std::size_t>(k)].data(), rL * n_k, r_new);
        cm = core_mat;
      }
    }

    // --- Convergence check ---
    // Only L_idx is reliable: R_idx is NOT updated during sweeps
    // (see comment at backward-sweep R_idx note).  The single-site
    // Vt_block lacks physical-dimension info, so R_idx cannot be
    // refined.  L_idx refinement alone is sufficient for convergence.
    bool changed = false;
    for (int k = 0; k < d - 1; ++k)
    {
      const int l_rk =
        static_cast<int>(L_idx[static_cast<std::size_t>(k + 1)].size()) /
        (k + 1);
      if (l_rk != prev_ranks[static_cast<std::size_t>(k)])
      {
        prev_ranks[static_cast<std::size_t>(k)] = l_rk;
        changed                                 = true;
      }
    }
    if (!changed)
    {
      ++stable_sweeps;
      if (stable_sweeps >= 2)
        break;
    }
    else
    {
      stable_sweeps = 0;
    }
  }

  return tt(std::move(cores));
}

}  // namespace tensor_train
}  // namespace mva
//
// :D
//
