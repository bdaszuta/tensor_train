/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Unified matvec_round() entry point: fused tt_matrix * tt apply
*/
#pragma once

#include <cstdio>
#include <cstdlib>

#include "../types/tt.hpp"
#include "../types/tt_matrix.hpp"
#include "round.hpp"
#include "tt_matrix_apply_round.hpp"
#include "tt_matvec_als_round.hpp"
#include "tt_matvec_dmrg_round.hpp"

namespace mva
{
namespace tensor_train
{

/**
 * @brief Fused matvec + round: compute A*x in TT form then compress.
 *
 * Applies the TT-matrix A to the TT vector x and rounds the result.
 * The svd_streaming method fuses the apply and round in a single pass
 * (no intermediate full-rank product).  Other methods apply first then
 * round.
 *
 * @param a    TT-matrix operator.
 * @param x    TT vector.
 * @param opts Round options controlling method, truncation, iteration.
 * @param info Optional diagnostics out-param.
 * @return Compressed A*x as a tt.
 * @note  Rejects svd_naive (matmat only).
 *        svd_streaming is the most memory-efficient for large systems.
 */
inline tt matvec_round(const tt_matrix& a,
                       const tt& x,
                       const round_options& opts,
                       round_result* info = nullptr)
{
  if (opts.method == round_method::svd_naive)
  {
    std::fprintf(stderr,
                 "matvec_round: round_method::svd_naive is matmat-only\n");
    std::abort();
  }
  switch (opts.method)
  {
    case round_method::svd_streaming:
    {
      tt out = detail::matvec_round_streaming(a, x, opts.eps, opts.max_rank);
      detail::svd_result_default(info);
      return out;
    }
    case round_method::als:
    {
      detail::als_options aopts = detail::round_options_to_als(opts);
      detail::als_result ainfo;
      tt out = detail::matvec_als_round(a, x, aopts, &ainfo);
      detail::als_result_to_round(ainfo, info);
      return out;
    }
    case round_method::dmrg:
    {
      detail::dmrg_options dopts = detail::round_options_to_dmrg(opts);
      detail::dmrg_result dinfo;
      tt out = detail::matvec_dmrg_round(a, x, dopts, &dinfo);
      detail::dmrg_result_to_round(dinfo, info);
      return out;
    }
    case round_method::svd:
    {
      tt out = detail::matvec_round(a, x, opts.eps, opts.max_rank);
      detail::svd_result_default(info);
      return out;
    }
    default:
    {
      std::fprintf(stderr,
                   "matvec_round: unknown round_method %d\n",
                   static_cast<int>(opts.method));
      std::abort();
    }
  }
}

}  // namespace tensor_train
}  // namespace mva
//
// :D
//
