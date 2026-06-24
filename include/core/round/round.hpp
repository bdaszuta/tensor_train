/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Unified round() entry point for tt and tt_matrix
*/
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#include "../types/tt.hpp"
#include "../types/tt_matrix.hpp"
#include "tt_als.hpp"
#include "tt_dmrg.hpp"
#include "tt_matrix_als_round.hpp"
#include "tt_matrix_dmrg_round.hpp"
#include "tt_matrix_round.hpp"
#include "tt_round.hpp"

namespace mva
{
namespace tensor_train
{

/** Algorithm selector for round / matvec_round / matmat_round. */
enum class round_method
{
  svd,            /**< Oseledets Alg. 2: QR + truncated SVD (round).
                        For matmat_round, dispatches to a fused streaming
                        variant.  Gauge:none does R->L QR + L->R SVD;
                        right_canonical reverses both sweeps */
  svd_streaming,  /**< matvec_round only: fused apply+round, no full product */
  svd_naive,      /**< matmat_round only: full apply then round */
  als,            /**< Alternating least squares (warm-start friendly) */
  dmrg            /**< Two-site DMRG with adaptive-rank SVD per bond */
};

/** Output gauge selector for round().
 *  When gauge == right_canonical, the SVD path produces output with
 *  cores[1..d-1] right-orthogonal.  When gauge == none (default),
 *  the output is norm-balanced with no gauge guarantee. */
enum class round_gauge
{
  none,              /**< Norm-balanced output, no gauge guarantee */
  right_canonical    /**< cores[1..d-1] right-orthogonal */
};

struct round_options
{
  /**
   * @brief SVD truncation tolerance (used by every method).
   * Truncates singular values such that the relative Frobenius
   * error vs the original is <= eps.  Set to 0.0 to disable
   * truncation (SVD path only; DMRG requires eps > 0).
   */
  double eps   = 0.0;
  /**
   * @brief Hard cap on per-bond rank (0 = unlimited).
   */
  int max_rank = 0;

  /**
   * @brief Compression method.  Default = svd.
   * svd_streaming is matvec_round only; svd_naive is matmat_round only.
   */
  round_method method = round_method::svd;

  /**
   * @brief Output gauge (svd method only).
   * right_canonical gives cores[1..d-1] right-orthogonal output.
   * Default: norm-balanced output, no gauge guarantee.
   */
  round_gauge gauge = round_gauge::none;

  /**
   * @brief Iterative knobs (ALS and DMRG only).
   */
  int max_iters      = 20;
  /**
   * @brief Convergence tolerance (ALS and DMRG only).
   * Iterative methods stop when the max relative change of bond
   * singular values between iterations falls below tol.
   */
  double tol         = 1.0e-12;
  /**
   * @brief Random seed for warm-start synthesis (ALS/DMRG only).
   * Currently unused in all paths; reserved for future use.
   */
  std::uint64_t seed = 0;

  /**
   * @brief Optional warm start for ALS and DMRG.
   * The matching pointer is consulted for the matching entry-point type.
   * Borrowed; must outlive the call.
   */
  const tt* warm_start_tt               = nullptr;
  const tt_matrix* warm_start_tt_matrix = nullptr;
};

/**
 * @brief Diagnostics out-param.
 * Populated when non-null.  SVD path writes
 * {iters_run=1, final_resid=0, converged=true} for uniformity.
 * ALS/DMRG paths populate final_resid with final_change (the
 * max relative change of bond singular values between iterations).
 */
struct round_result
{
  int iters_run      = 0;
  double final_resid = 0.0;
  bool converged     = false;
};

namespace detail
{

inline void round_assert_method_for_value(round_method m, const char* what)
{
  if (m == round_method::svd_streaming || m == round_method::svd_naive)
  {
    std::fprintf(stderr,
                 "%s: round_method::svd_streaming / svd_naive are only "
                 "valid on matvec_round / matmat_round respectively.\n",
                 what);
    std::abort();
  }
}

inline detail::als_options round_options_to_als(const round_options& o)
{
  detail::als_options a;
  a.max_rank   = o.max_rank;
  // Reject NaN eps (IEEE 754: NaN <= 0.0 is false).
  if (std::isnan(o.eps))
  {
    std::fprintf(stderr,
                 "round: eps is NaN for method=als\n");
    std::abort();
  }
  a.eps        = o.eps;
  a.max_iters  = o.max_iters;
  a.tol        = o.tol;
  a.warm_start = o.warm_start_tt;
  a.seed       = o.seed;
  return a;
}

inline detail::dmrg_options round_options_to_dmrg(const round_options& o)
{
  detail::dmrg_options d;
  d.max_rank   = o.max_rank;
  // round_options::eps defaults to 0.0 ("no truncation" for SVD), but
  // DMRG requires a positive eps for the per-bond truncation threshold.
  // Reject eps <= 0 and NaN.  IEEE 754: NaN <= 0.0 is false,
  // so a separate std::isnan check is required.
  if (std::isnan(o.eps) || o.eps <= 0.0)
  {
    std::fprintf(stderr,
                 "round: eps must be > 0 for method=dmrg "
                 "(eps=0 means 'no truncation' only for the SVD path)\n");
    std::abort();
  }
  d.eps        = o.eps;
  d.max_iters  = o.max_iters;
  d.tol        = o.tol;
  d.warm_start = o.warm_start_tt;
  d.seed       = o.seed;
  return d;
}

inline void als_result_to_round(const detail::als_result& a, round_result* r)
{
  if (!r)
    return;
  r->iters_run   = a.iters_run;
  r->final_resid = a.final_change;
  r->converged   = a.converged;
}

inline void dmrg_result_to_round(const detail::dmrg_result& d, round_result* r)
{
  if (!r)
    return;
  r->iters_run   = d.iters_run;
  r->final_resid = d.final_change;
  r->converged   = d.converged;
}

inline void svd_result_default(round_result* r)
{
  if (!r)
    return;
  r->iters_run   = 1;
  r->final_resid = 0.0;
  r->converged   = true;
}

}  // namespace detail

/**
 * @brief Compress a TT via the method specified in opts.
 *
 * Supports SVD truncation (Oseledets Alg. 2), alternating least
 * squares (ALS), and two-site DMRG.  The SVD path performs R->L QR
 * followed by L->R truncated SVD under gauge=none; the
 * right_canonical gauge reverses both sweep directions.  The ALS
 * and DMRG paths iteratively optimize each core to satisfy the
 * TT-SVD quasi-optimality condition.
 *
 * The const-ref overload copies internally (more overhead, leaves
 * input intact).  Prefer the rvalue-ref overload when the input is
 * an expiring temporary.
 *
 * @param a    Input TT (const-ref or rvalue).
 * @param opts Round options: method, eps, max_rank, gauge, iterative
 *             parameters, and optional warm-start pointer.
 * @param info Optional diagnostics out-param (iters_run, final_resid,
 *             converged).  SVD path reports iters_run=1, converged=true.
 * @return Compressed copy of a.
 * @note  Rejects svd_streaming and svd_naive (matvec/matmat only).
 *        The SVD path ignores max_iters, tol, and seed.
 */
inline tt round(const tt& a,
                const round_options& opts,
                round_result* info = nullptr)
{
  detail::round_assert_method_for_value(opts.method, "round(tt)");
  switch (opts.method)
  {
    case round_method::als:
    {
      detail::als_options aopts = detail::round_options_to_als(opts);
      detail::als_result ainfo;
      tt out = detail::als_round(a, aopts, &ainfo);
      detail::als_result_to_round(ainfo, info);
      return out;
    }
    case round_method::dmrg:
    {
      detail::dmrg_options dopts = detail::round_options_to_dmrg(opts);
      detail::dmrg_result dinfo;
      tt out = detail::dmrg_round(a, dopts, &dinfo);
      detail::dmrg_result_to_round(dinfo, info);
      return out;
    }
    case round_method::svd:
    {
      tt out = (opts.gauge == round_gauge::right_canonical)
                 ? detail::round_right_canonical(a, opts.eps, opts.max_rank)
                 : detail::round(a, opts.eps, opts.max_rank);
      detail::svd_result_default(info);
      return out;
    }
    default:
      std::fprintf(stderr, "round: unknown method %d\n",
                   static_cast<int>(opts.method));
      std::abort();
  }
}

/**
 * @brief Rvalue overload: may move from the input TT's cores.
 * Semantics are identical to the const-ref overload; the rvalue
 * path in the SVD method exploits move semantics to reduce copies.
 */
inline tt round(tt&& a,
                const round_options& opts,
                round_result* info = nullptr)
{
  detail::round_assert_method_for_value(opts.method, "round(tt&&)");
  switch (opts.method)
  {
    case round_method::als:
    {
      // ALS engine consumes by const-ref; rvalue path reduces to const-ref.
      detail::als_options aopts = detail::round_options_to_als(opts);
      detail::als_result ainfo;
      tt out = detail::als_round(a, aopts, &ainfo);
      detail::als_result_to_round(ainfo, info);
      return out;
    }
    case round_method::dmrg:
    {
      detail::dmrg_options dopts = detail::round_options_to_dmrg(opts);
      detail::dmrg_result dinfo;
      tt out = detail::dmrg_round(a, dopts, &dinfo);
      detail::dmrg_result_to_round(dinfo, info);
      return out;
    }
    case round_method::svd:
    {
      tt out = (opts.gauge == round_gauge::right_canonical)
                 ? detail::round_right_canonical(std::move(a),
                                                  opts.eps,
                                                  opts.max_rank)
                 : detail::round(std::move(a), opts.eps, opts.max_rank);
      detail::svd_result_default(info);
      return out;
    }
    default:
      std::fprintf(stderr, "round: unknown method %d\n",
                   static_cast<int>(opts.method));
      std::abort();
  }
}

/**
 * @brief Compress a TT-matrix via the method specified in opts.
 *
 * TT-matrix compression works by packing each 4-axis core
 * (r, m, n, r') into a 3-axis core (r, m*n, r') and applying
 * the same round algorithms as the TT vector case, then unpacking.
 * All method, eps, max_rank, and iterative options apply identically.
 *
 * @param a    Input TT-matrix.
 * @param opts Round options.
 * @param info Optional diagnostics out-param.
 * @return Compressed copy of a.
 * @note  The warm_start_tt_matrix field in opts is used instead of
 *        warm_start_tt for the ALS and DMRG paths.
 */
inline tt_matrix round(const tt_matrix& a,
                       const round_options& opts,
                       round_result* info = nullptr)
{
  detail::round_assert_method_for_value(opts.method, "round(tt_matrix)");
  switch (opts.method)
  {
    case round_method::als:
    {
      detail::als_options aopts = detail::round_options_to_als(opts);
      // For tt_matrix, warm-start is supplied separately.
      aopts.warm_start = nullptr;
      detail::als_result ainfo;
      tt_matrix out =
        detail::als_round_matrix(a, aopts, opts.warm_start_tt_matrix, &ainfo);
      detail::als_result_to_round(ainfo, info);
      return out;
    }
    case round_method::dmrg:
    {
      detail::dmrg_options dopts = detail::round_options_to_dmrg(opts);
      dopts.warm_start           = nullptr;
      detail::dmrg_result dinfo;
      tt_matrix out =
        detail::dmrg_round_matrix(a, dopts, opts.warm_start_tt_matrix, &dinfo);
      detail::dmrg_result_to_round(dinfo, info);
      return out;
    }
    case round_method::svd:
    {
      tt_matrix out = (opts.gauge == round_gauge::right_canonical)
                        ? detail::round_matrix_right_canonical(
                            a, opts.eps, opts.max_rank)
                        : detail::round_matrix(a, opts.eps, opts.max_rank);
      detail::svd_result_default(info);
      return out;
    }
    default:
      std::fprintf(stderr, "round: unknown method %d\n",
                   static_cast<int>(opts.method));
      std::abort();
  }
}

/**
 * @brief Rvalue overload for TT-matrix round.
 * Semantics are identical to the const-ref overload; the rvalue
 * path in the SVD method exploits move semantics to reduce copies.
 */
inline tt_matrix round(tt_matrix&& a,
                       const round_options& opts,
                       round_result* info = nullptr)
{
  detail::round_assert_method_for_value(opts.method, "round(tt_matrix&&)");
  switch (opts.method)
  {
    case round_method::als:
    {
      // Matrix ALS path takes const-ref internally (pack-as-tt requires
      // a memcpy per core for narray rank-4->rank-3 conversion).
      // The rvalue binds to const&; no move semantics at the pack level,
      // but we avoid the redundant const-ref dispatch hop.
      detail::als_options aopts = detail::round_options_to_als(opts);
      aopts.warm_start = nullptr;
      detail::als_result ainfo;
      tt_matrix out = detail::als_round_matrix(
        a, aopts, opts.warm_start_tt_matrix, &ainfo);
      detail::als_result_to_round(ainfo, info);
      return out;
    }
    case round_method::dmrg:
    {
      detail::dmrg_options dopts = detail::round_options_to_dmrg(opts);
      dopts.warm_start = nullptr;
      detail::dmrg_result dinfo;
      tt_matrix out = detail::dmrg_round_matrix(
        a, dopts, opts.warm_start_tt_matrix, &dinfo);
      detail::dmrg_result_to_round(dinfo, info);
      return out;
    }
    case round_method::svd:
    {
      tt_matrix out = (opts.gauge == round_gauge::right_canonical)
                        ? detail::round_matrix_right_canonical(
                            std::move(a), opts.eps, opts.max_rank)
                        : detail::round_matrix(std::move(a),
                                               opts.eps,
                                               opts.max_rank);
      detail::svd_result_default(info);
      return out;
    }
    default:
      std::fprintf(stderr, "round: unknown method %d\n",
                   static_cast<int>(opts.method));
      std::abort();
  }
}

}  // namespace tensor_train
}  // namespace mva
//
// :D
//
