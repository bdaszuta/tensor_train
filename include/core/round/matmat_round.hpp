/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Unified matmat_round() entry point: fused tt_matrix * tt_matrix
*/
#pragma once

#include <cstdio>
#include <cstdlib>

#include "../types/tt_matrix.hpp"
#include "round.hpp"
#include "tt_matmat_als_round.hpp"
#include "tt_matmat_dmrg_round.hpp"
#include "tt_matrix_apply_round.hpp"

namespace mva
{
namespace tensor_train
{

/**
 * @brief Fused matmat + round: compute A*B in TT-matrix form then compress.
 *
 * Multiplies two TT-matrices A and B and rounds the result.
 * The svd method uses a fused streaming approach; svd_naive computes
 * the full product first then rounds (higher memory).
 *
 * @param a    Left TT-matrix operand.
 * @param b    Right TT-matrix operand.
 * @param opts Round options controlling method, truncation, iteration.
 * @param info Optional diagnostics out-param.
 * @return Compressed A*B as a tt_matrix.
 * @note  Rejects svd_streaming (matvec only).
 *        svd_naive computes the full product then rounds (higher
 *        memory); svd uses a fused streaming approach.
 */
inline tt_matrix matmat_round(const tt_matrix& a,
                              const tt_matrix& b,
                              const round_options& opts,
                              round_result* info = nullptr)
{
  if (opts.method == round_method::svd_streaming)
  {
    std::fprintf(stderr,
                 "matmat_round: round_method::svd_streaming is matvec-only\n");
    std::abort();
  }
  switch (opts.method)
  {
    case round_method::svd_naive:
    {
      tt_matrix out =
        detail::matmat_round_naive(a, b, opts.eps, opts.max_rank);
      detail::svd_result_default(info);
      return out;
    }
    case round_method::als:
    {
      detail::als_options aopts = detail::round_options_to_als(opts);
      detail::als_result ainfo;
      tt_matrix out = detail::matmat_als_round(
        a, b, aopts, opts.warm_start_tt_matrix, &ainfo);
      detail::als_result_to_round(ainfo, info);
      return out;
    }
    case round_method::dmrg:
    {
      detail::dmrg_options dopts = detail::round_options_to_dmrg(opts);
      detail::dmrg_result dinfo;
      tt_matrix out = detail::matmat_dmrg_round(
        a, b, dopts, opts.warm_start_tt_matrix, &dinfo);
      detail::dmrg_result_to_round(dinfo, info);
      return out;
    }
    case round_method::svd:
    {
      tt_matrix out = detail::matmat_round(a, b, opts.eps, opts.max_rank);
      detail::svd_result_default(info);
      return out;
    }
    default:
    {
      std::fprintf(stderr,
                   "matmat_round: unknown round_method %d\n",
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
