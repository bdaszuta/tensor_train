/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Two-site DMRG-style TT rounding. Adaptive rank via SVD truncation
*/
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "../algebra/tt_inner.hpp"
#include "../detail/contract.hpp"
#include "../detail/dmrg_local_solve.hpp"
#include "../detail/warm_start.hpp"
#include "../gauge/tt_orthogonalize.hpp"
#include "../types/tt.hpp"
#include "../types/tt_core.hpp"
#include "../types/tt_eigen_bridge.hpp"
#include "tt_round.hpp"

namespace mva
{
namespace tensor_train
{

namespace detail
{

struct dmrg_options
{
  int max_rank         = 0;        // 0 = derive from SVD reference when eps > 0
  double eps           = 1.0e-12;  // per-bond truncation tolerance
  int max_iters        = 20;
  double tol           = 1.0e-12;  // convergence: max relative change of bond singular values
  const tt* warm_start = nullptr;
  std::uint64_t seed   = 0;  // unused (init via round() in cold path)
};

struct dmrg_result
{
  int iters_run       = 0;
  double final_change = 0.0;
  bool converged      = false;
  int max_bond_rank   = 0;
};

inline tt dmrg_engine(const tt& Y,
                      tt X,
                      const dmrg_options& opts,
                      dmrg_result* info,
                      bool input_is_right_canonical = false)
{
  using detail::contract_inner_step;
  using detail::contract_inner_step_right;
  using detail::merge_y_block;
  using detail::split_two_site_lr;
  using detail::split_two_site_rl;
  using eigen_bridge::row_matrix;

  const int d = Y.d();

  if (d <= 1)
  {
    if (info)
    {
      info->iters_run     = 0;
      info->final_change  = 0.0;
      info->converged     = true;
      info->max_bond_rank = 0;
    }
    return (d == 0) ? tt{} : tt{std::vector<tt_core>{Y.core(0)}};
  }

  // Place X in right-canonical gauge so the initial R envs are valid.
  if (!input_is_right_canonical)
  {
    right_orthogonalize(X);
  }

  // Per-bond delta uses ||Y||_F (not ||X||_F): the target tensor is Y.
  const double y_inner = inner(Y, Y);
  const double y_norm  = std::sqrt(std::max(y_inner, 0.0));
  const double delta =
    opts.eps * y_norm / std::sqrt(static_cast<double>(std::max(d - 1, 1)));

  // R_envs[k] = env over sites k..d-1, shape (rYl_k, rXl_k).
  // Bond k (sites k, k+1) consumes R_envs[k+2].
  std::vector<row_matrix> R_envs(d + 1);
  R_envs[d].resize(1, 1);
  R_envs[d](0, 0) = 1.0;
  for (int k = d - 1; k >= 1; --k)
  {
    R_envs[k] = contract_inner_step_right(Y.core(k), X.core(k), R_envs[k + 1]);
  }

  std::vector<row_matrix> L_envs(d + 1);
  L_envs[0].resize(1, 1);
  L_envs[0](0, 0) = 1.0;

  // Per-bond singular-value head from the previous iteration's L->R
  // sweep.  Gauge-invariant fingerprint of X used for sweep-internal
  // convergence: max relative ||s_new[k] - s_prev[k]||_2 across bonds.
  using eigen_bridge::col_vector;
  std::vector<col_vector> s_prev(d - 1);
  bool have_prev_s = false;
  double change    = 0.0;
  bool converged   = false;
  int iters_run    = 0;

  std::vector<col_vector> s_now(d - 1);

  for (int it = 0; it < opts.max_iters; ++it)
  {
    // -------- L -> R sweep over bonds 0..d-2 --------
    for (int k = 0; k < d - 1; ++k)
    {
      const int rXl_k     = static_cast<int>(L_envs[k].rows());
      const int n_phys_k  = Y.core(k).n_phys();
      const int n_phys_kp = Y.core(k + 1).n_phys();
      const int rXr_p     = static_cast<int>(R_envs[k + 2].cols());

      row_matrix Theta =
        merge_y_block(L_envs[k], Y.core(k), Y.core(k + 1), R_envs[k + 2]);
      split_two_site_lr(Theta,
                        rXl_k,
                        n_phys_k,
                        n_phys_kp,
                        rXr_p,
                        delta,
                        opts.max_rank,
                        X.cores()[k],
                        X.cores()[k + 1],
                        &s_now[k]);

      // X_k now left-orthogonal; refresh L env at bond k+1.
      L_envs[k + 1] = contract_inner_step(L_envs[k], X.core(k), Y.core(k));
    }

    // -------- R -> L sweep over bonds d-2..0 --------
    for (int k = d - 2; k >= 0; --k)
    {
      const int rXl_k     = static_cast<int>(L_envs[k].rows());
      const int n_phys_k  = Y.core(k).n_phys();
      const int n_phys_kp = Y.core(k + 1).n_phys();
      const int rXr_p     = static_cast<int>(R_envs[k + 2].cols());

      row_matrix Theta =
        merge_y_block(L_envs[k], Y.core(k), Y.core(k + 1), R_envs[k + 2]);
      split_two_site_rl(Theta,
                        rXl_k,
                        n_phys_k,
                        n_phys_kp,
                        rXr_p,
                        delta,
                        opts.max_rank,
                        X.cores()[k],
                        X.cores()[k + 1]);

      // X_{k+1} now right-orthogonal; refresh R env at bond k+1.
      R_envs[k + 1] =
        contract_inner_step_right(Y.core(k + 1), X.core(k + 1), R_envs[k + 2]);
    }

    ++iters_run;

    // Sweep-internal convergence: max relative change of bond
    // singular-value heads.  Gauge-invariant; available "for free"
    // from the L->R sweep's truncated SVDs.
    if (have_prev_s)
    {
      double max_rel = 0.0;
      for (int k = 0; k < d - 1; ++k)
      {
        const double n_now = s_now[k].norm();
        if (s_prev[k].size() == s_now[k].size())
        {
          const double diff = (s_now[k] - s_prev[k]).norm();
          const double rel  = diff / (n_now + 1.0e-300);
          if (rel > max_rel)
            max_rel = rel;
        }
        else
        {
          // shape change at this bond -> not converged
          max_rel = 1.0;
        }
      }
      change = max_rel;
    }
    else
    {
      change = 1.0;  // no baseline yet
    }
    s_prev      = s_now;
    have_prev_s = true;

    if (change <= opts.tol)
    {
      converged = true;
      break;
    }
  }

  if (info)
  {
    info->iters_run    = iters_run;
    info->final_change = change;
    info->converged    = converged;
    int rmax           = 0;
    for (int r : X.ranks())
      if (r > rmax)
        rmax = r;
    info->max_bond_rank = rmax;
  }
  return X;
}

// DMRG bond-compression of an explicit tt Y.  Uses a
// round_right_canonical() initial guess in the cold path;
// iterative codes should pass a warm_start.
inline tt dmrg_round(const tt& Y,
                     const dmrg_options& opts,
                     dmrg_result* info = nullptr)
{
  if (std::isnan(opts.eps) || opts.eps <= 0.0)
  {
    std::fprintf(stderr,
                 "dmrg_round: eps must be > 0 for DMRG compression "
                 "(got %.15e); eps=0 means 'no truncation' only for "
                 "the SVD path\n",
                 opts.eps);
    std::abort();
  }

  const int d = Y.d();

  if (d <= 1)
  {
    tt out = round(Y, opts.eps, opts.max_rank);
    if (info)
    {
      info->iters_run    = 0;
      info->final_change = 0.0;
      info->converged    = true;
      int rmax           = 0;
      for (int r : out.ranks())
        if (r > rmax)
          rmax = r;
      info->max_bond_rank = rmax;
    }
    return out;
  }

  bool use_warm = detail::warm_start_tt_compatible(opts.warm_start, Y);

  tt X_init;
  bool init_is_right_canonical = false;
  if (use_warm)
  {
    X_init = detail::copy_tt(*opts.warm_start);
    if (opts.max_rank > 0)
    {
      for (int r : X_init.ranks())
      {
        if (r > opts.max_rank)
        {
          std::fprintf(stderr,
                       "dmrg_round: warm_start max rank %d"
                       " exceeds max_rank %d\n",
                       r,
                       opts.max_rank);
          std::abort();
        }
      }
    }
  }
  else if (opts.max_rank > 0 || opts.eps > 0.0)
  {
    X_init = round_right_canonical(Y, opts.eps, opts.max_rank);
    init_is_right_canonical = true;
  }
  else
  {
    // NaN eps can reach here (NaN > 0.0 is false per IEEE 754)
    // despite the std::isnan guard above (defense-in-depth).
    std::fprintf(stderr,
                 "dmrg_round: eps=%.15e is not a valid positive "
                 "number\n",
                 opts.eps);
    std::abort();
  }

  return detail::dmrg_engine(
    Y, std::move(X_init), opts, info, init_is_right_canonical);
}

}  // namespace detail

}  // namespace tensor_train
}  // namespace mva
//
// :D
//
