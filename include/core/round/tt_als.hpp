/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: ALS-based TT rounding. Alternative to round() for the regime where a good warm-start guess is available (e.g. from a previous timestep).
*/
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#include "../detail/als_local_solve.hpp"
#include "../detail/contract.hpp"
#include "../detail/orthogonalize.hpp"
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

struct als_options
{
  int max_rank         = 0;    // target bond cap (0 = derive from SVD reference when eps > 0)
  double eps           = 0.0;  // used by als_round() for NaN guard + SVD-ref rank estimate
  int max_iters        = 20;
  double tol           = 1.0e-12;  // convergence: max relative change of bond singular values
  const tt* warm_start = nullptr;
  std::uint64_t seed   = 0;  // unused
};

struct als_result
{
  int iters_run       = 0;
  double final_change = 0.0;
  bool converged      = false;
};

// Core ALS sweep loop.  Caller supplies an initial guess X_init with the
// desired bond cap; engine assumes X_init has matching mode shape.
inline tt als_engine(const tt& Y,
                     tt X,
                     const als_options& opts,
                     als_result* info,
                     bool input_is_right_canonical = false)
{
  using detail::als_local_update;
  using detail::contract_inner_step;
  using detail::contract_inner_step_right;
  using detail::left_qr_step;
  using detail::right_qr_step;
  using eigen_bridge::row_matrix;

  const int d = Y.d();

  if (d <= 1)
  {
    if (info)
    {
      info->iters_run     = 0;
      info->final_change  = 0.0;
      info->converged     = true;
    }
    return (d == 0) ? tt{} : tt{std::vector<tt_core>{Y.core(0)}};
  }

  // ---------------- canonical form + right envs ----------------
  if (!input_is_right_canonical)
  {
    right_orthogonalize(X);
  }
  // R_envs[k] = right env at bond k, for k = 1 .. d, using a
  // length-(d+1) vector where R_envs[0] is unused.  R_envs[d]
  // is the trivial 1x1 [[1]] (right boundary).  Equivalently:
  // R_{k} = R_envs[k] for k = 1..d, with R_{d} = [[1]].
  std::vector<row_matrix> R_envs(d + 1);
  R_envs[d].resize(1, 1);
  R_envs[d](0, 0) = 1.0;
  for (int k = d - 1; k >= 1; --k)
  {
    R_envs[k] = contract_inner_step_right(Y.core(k), X.core(k), R_envs[k + 1]);
  }

  // ---------------- sweep loop ----------------
  // We run up to max_iters full sweeps (one L->R + one R->L per iter).
  // Convergence: max relative change of per-core singular values across
  // all bonds.  After R->L sweep, the SVD of left_unfold(core(k)) for
  // k=0..d-2 extracts per-core SVs; these converge alongside the tensor
  // X and their max relative change signals when all cores have stopped
  // moving.  Tracking all bonds (unlike the previous bond-0-only check)
  // catches interior-bond lag that can cause premature termination.
  using eigen_bridge::col_vector;
  std::vector<col_vector> s_prev(d - 1);
  bool have_prev_s = false;
  double change    = 0.0;
  bool converged   = false;
  int iters_run    = 0;

  std::vector<row_matrix> L_envs(d + 1);
  L_envs[0].resize(1, 1);
  L_envs[0](0, 0) = 1.0;

  for (int it = 0; it < opts.max_iters; ++it)
  {
    // -------- L -> R sweep --------
    for (int k = 0; k < d; ++k)
    {
      // Local optimum X_k = L_k . Y_k . R_{k+1}.
      X.cores()[k] = als_local_update(L_envs[k], Y.core(k), R_envs[k + 1]);

      if (k < d - 1)
      {
        // Move orth center to k+1: left-QR on X_k, absorb into X_{k+1}.
        left_qr_step(X.cores()[k], X.cores()[k + 1]);
        // Update L_{k+1} now that X_k is left-orthogonal.
        L_envs[k + 1] = contract_inner_step(L_envs[k], X.core(k), Y.core(k));
      }
    }

    // -------- R -> L sweep --------
    for (int k = d - 1; k >= 0; --k)
    {
      if (k < d - 1)
      {
        // For k = d-1 the orth center is already at d-1 (just solved),
        // so re-solving is a no-op modulo round-off; skip.  For k < d-1
        // the L env is stale w.r.t. orthogonality of X_k+1..d-1 (which
        // is now right-orthogonal after our R->L moves), but the L env
        // is built from earlier left-orthogonal X cores so it remains
        // valid.
        X.cores()[k] = als_local_update(L_envs[k], Y.core(k), R_envs[k + 1]);
      }

      if (k > 0)
      {
        // Move orth center to k-1: right-QR on X_k, absorb into X_{k-1}.
        right_qr_step(X.cores()[k], X.cores()[k - 1]);
        // Update R_k now that X_k is right-orthogonal.
        R_envs[k] =
          contract_inner_step_right(Y.core(k), X.core(k), R_envs[k + 1]);
      }
    }

    ++iters_run;

    // All-bond convergence: SVD of left_unfold for every core 0..d-2.
    // These per-core singular values converge as the ALS iteration
    // converges, and their max relative change across bonds reliably
    // detects when all cores have stopped moving.  Mirrors the DMRG
    // engine's convergence logic (tt_dmrg.hpp:191-209) verbatim; the
    // only difference is that DMRG collects s_now[k] from two-site
    // SVDs during the L->R sweep, while ALS computes them after the
    // R->L sweep from per-core left-unfolds.
    std::vector<col_vector> s_now(d - 1);
    for (int k = 0; k < d - 1; ++k)
    {
      const tt_core& ck = X.core(k);
      eigen_bridge::row_matrix U_tmp;
      eigen_bridge::row_matrix Vt_tmp;
      detail::svd_thin(ck.data(),
                       static_cast<Eigen::Index>(ck.r_left())
                         * ck.n_phys(),
                       static_cast<Eigen::Index>(ck.r_right()),
                       U_tmp,
                       s_now[k],
                       Vt_tmp);
    }

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
    s_prev      = std::move(s_now);
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
  }
  return X;
}

inline tt als_round(const tt& Y,
                    const als_options& opts,
                    als_result* info = nullptr)
{
  if (std::isnan(opts.eps))
  {
    std::fprintf(stderr,
                 "als_round: eps is NaN; must be a valid number\n");
    std::abort();
  }

  const int d = Y.d();

  // Trivial case: degenerate d -- no bonds to compress.
  if (d <= 1)
  {
    tt out = round(Y, opts.eps, opts.max_rank);
    if (info)
    {
      info->iters_run    = 0;
      info->final_change = 0.0;
      info->converged    = true;
    }
    return out;
  }

  // Build effective options.  When max_rank is not specified, derive
  // the target rank from an SVD reference TT at tolerance eps.
  tt svd_ref;  // lives for duration of als_round; pointed to by warm_start
  als_options effective_opts = opts;
  if (effective_opts.max_rank <= 0)
  {
    if (effective_opts.eps <= 0.0)
    {
      std::fprintf(stderr,
                   "als_round: max_rank <= 0 and eps <= 0; "
                   "ALS requires either max_rank > 0 or eps > 0 "
                   "to determine compression target\n");
      std::abort();
    }
    svd_ref = round(Y, effective_opts.eps, 0);
    effective_opts.max_rank = svd_ref.max_rank();
    // Use SVD reference as warm-start if caller did not provide one.
    // Avoids paying the SVD cost twice (once for rank, once for init).
    if (effective_opts.warm_start == nullptr)
      effective_opts.warm_start = &svd_ref;
  }

  // ---------------- initial guess ----------------
  bool use_warm = detail::warm_start_tt_compatible(effective_opts.warm_start,
                                                    Y);
  if (use_warm)
  {
    for (int r : effective_opts.warm_start->ranks())
    {
      if (r > effective_opts.max_rank)
      {
        std::fprintf(stderr,
                     "als_round: warm_start max rank %d exceeds "
                     "max_rank %d\n",
                     r, effective_opts.max_rank);
        std::abort();
      }
    }
  }
  tt X_init     = use_warm
                    ? detail::copy_tt(*effective_opts.warm_start)
                    : round_right_canonical(Y, 0.0, effective_opts.max_rank);
  const bool init_is_right_canonical = !use_warm;

  return detail::als_engine(
    Y, std::move(X_init), effective_opts, info, init_is_right_canonical);
}

}  // namespace detail

}  // namespace tensor_train
}  // namespace mva
//
// :D
//
