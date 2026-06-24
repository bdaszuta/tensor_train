/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: TT-SVD compression of a dense d-mode tensor at a given relative Frobenius tolerance (Oseledets' Algorithm 1)
*/
#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "../detail/core_view.hpp"
#include "../detail/truncate.hpp"
#include "../types/tt.hpp"
#include "../types/tt_core.hpp"
#include "../types/tt_eigen_bridge.hpp"

namespace mva
{
namespace tensor_train
{
namespace detail
{

/**
 * @brief Compress a dense tensor to TT form via Oseledets' Algorithm 1.
 *
 * Sequentially reshapes and SVD-truncates the dense buffer, building
 * cores from left to right.  The relative Frobenius error satisfies
 * \f$ \|T - T_{\mathrm{TT}}\|_F \le \epsilon \cdot \|T\|_F \f$.
 *
 * @param dense    Flat row-major buffer of length prod(shape).
 * @param shape    Mode sizes n_0, ..., n_{d-1}.
 * @param eps      Relative Frobenius truncation tolerance.
 * @param max_rank Hard cap on per-bond rank (0 = unlimited).
 * @return TT approximant of the dense tensor.
 * @note  The dense buffer must be row-major in the physical indices.
 */
inline tt svd(const double* dense,
              const std::vector<int>& shape,
              double eps   = 1.0e-10,
              int max_rank = 0)
{
  namespace eb = eigen_bridge;
  const int d  = static_cast<int>(shape.size());
  if (d == 0)
  {
    // Empty shape: return an empty TT (no cores).
    return tt();
  }

  long long total_ll = 1;
  for (int k = 0; k < d; ++k)
  {
    const int n = shape[k];
    if (n <= 0)
    {
      std::fprintf(stderr,
                   "tt_svd: shape[%d]=%d is not positive; aborting.\n",
                   k, n);
      std::abort();
    }
    total_ll *= n;
  }
  if (total_ll > static_cast<long long>(std::numeric_limits<int>::max()))
  {
    std::fprintf(stderr,
                 "tt_svd: dense tensor has %lld entries -- exceeds INT_MAX; "
                 "not supported.\n",
                 total_ll);
    std::abort();
  }
  const int total = static_cast<int>(total_ll);

  // Frobenius norm via Eigen for vectorisation.
  Eigen::Map<const Eigen::VectorXd> dvec(dense, total);
  const double norm_T = dvec.norm();

  if (!std::isfinite(norm_T))
  {
    // NaN or Inf in input: abort with diagnostic rather than
    // silently producing garbage.
    std::fprintf(stderr,
                 "tt_svd: input contains non-finite values "
                 "(Frobenius norm=%.15e); cannot compress\n",
                 norm_T);
    std::abort();
  }

  std::vector<tt_core> cores;
  cores.reserve(d);

  if (norm_T == 0.0)
  {
    // All-zero tensor: rank-1 zero TT.
    for (int k = 0; k < d; ++k)
    {
      tt_core c(1, shape[k], 1);
      c.zero_clear();
      cores.push_back(std::move(c));
    }
    return tt(std::move(cores));
  }

  const double delta = eps * norm_T / std::sqrt(std::max(d - 1, 1));

  // Working buffer C, initially holds the full dense tensor.
  std::vector<double> C(dense, dense + total);
  int r_prev = 1;
  int rest   = total;  // remaining product across yet-unswept modes.

  for (int k = 0; k < d - 1; ++k)
  {
    const int n_k    = shape[k];
    if (rest % n_k != 0)
    {
      std::fprintf(stderr,
                   "tt_svd: shape integrity violation at axis %d "
                   "(rest=%d not divisible by n_k=%d); aborting.\n",
                   k, rest, n_k);
      std::abort();
    }
    const int M_rows = r_prev * n_k;
    const int M_cols = rest / n_k;
    eb::row_matrix U;
    eb::col_vector s;
    eb::row_matrix Vt;
    svd_thin(C.data(), M_rows, M_cols, U, s, Vt);
    const int r_new = truncate_eps_rank(s, delta, max_rank);

    // Store core: U.leftCols(r_new) reshaped (r_prev, n_k, r_new).
    tt_core core(r_prev, n_k, r_new);
    detail::left_unfold(core) = U.leftCols(r_new);
    cores.push_back(std::move(core));

    // Push s_head * Vt_head into next C: shape (r_new, M_cols) row-major.
    eb::row_matrix sVt = diag_scale_rows(Vt, s, r_new);
    C.assign(static_cast<std::size_t>(r_new) * M_cols, 0.0);
    Eigen::Map<eb::row_matrix>(C.data(), r_new, M_cols) = sVt;
    r_prev                                              = r_new;
    rest                                                = M_cols;
  }

  // Last core: shape (r_prev, shape[d-1], 1).
  {
    const int n_last = shape[d - 1];
    tt_core core(r_prev, n_last, 1);
    std::memcpy(core.data(),
                C.data(),
                sizeof(double) * static_cast<std::size_t>(r_prev) * n_last);
    cores.push_back(std::move(core));
  }

  return tt(std::move(cores));
}

}  // namespace detail
}  // namespace tensor_train
}  // namespace mva
//
// :D
//
