/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: TT-matrix apply (matvec, matmat) -- rank-multiplying products
*/
#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "../detail/shape_check.hpp"
#include "../types/tt.hpp"
#include "../types/tt_core.hpp"
#include "../types/tt_eigen_bridge.hpp"
#include "../types/tt_matrix.hpp"

namespace mva
{
namespace tensor_train
{
namespace detail
{

// y_core(a*s+ax, i, b*sp+bx)
//   = sum_j A_core(a, i, j, b) * x_core(ax, j, bx).
//
// Plan: two permutes + one GEMM.  Pack A into M_p((a,i,b), j) and
// x into X_p(j, (ax,bx)).  GEMM gives T((a,i,b), (ax,bx)).  The
// bx-axis is the innermost in both T (last col axis) and out (last
// linear axis), so it is a memcpy of sp doubles per slab.  The
// resulting core has shape (r*s, m, rp*sp) row-major.
inline tt_core matvec_core(const tt_matrix_core& mc, const tt_core& xc)
{
  const int r  = mc.r_left();
  const int m  = mc.m_phys();
  const int n  = mc.n_phys();
  const int rp = mc.r_right();
  const int s  = xc.r_left();
  const int nx = xc.n_phys();
  const int sp = xc.r_right();
  if (n != nx)
  {
    std::fprintf(stderr, "matvec: mode mismatch %d vs %d\n", n, nx);
    std::abort();
  }

  using row_mat = eigen_bridge::row_matrix;

  const Eigen::Index M_rows = static_cast<Eigen::Index>(r) * m * rp;
  const Eigen::Index X_cols = static_cast<Eigen::Index>(s) * sp;
  const Eigen::Index T_rows = M_rows;
  const Eigen::Index T_cols = X_cols;

  // Permute mc(a,i,j,b) -> M_p((a,i,b), j) so j is the contracted axis.
  row_mat M_p(M_rows, n);
  for (int a = 0; a < r; ++a)
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)
        for (int b = 0; b < rp; ++b)
          M_p((static_cast<Eigen::Index>(a) * m + i) * rp + b, j) =
            mc(a, i, j, b);

  // Permute xc(ax,j,bx) -> X_p(j, (ax,bx)) so j is the contracted axis.
  row_mat X_p(n, X_cols);
  for (int ax = 0; ax < s; ++ax)
    for (int j = 0; j < n; ++j)
      for (int bx = 0; bx < sp; ++bx)
        X_p(j, static_cast<Eigen::Index>(ax) * sp + bx) = xc(ax, j, bx);

  // One GEMM: T((a,i,b), (ax,bx)) = sum_j M_p((a,i,b),j) X_p(j,(ax,bx)).
  row_mat T(T_rows, T_cols);
  T.noalias() = M_p * X_p;

  // Repackage T into output core shape (r*s, m, rp*sp) with row-major
  // linear = ((a*s+ax)*m + i)*rp*sp + (b*sp+bx).
  {
    const int64_t rL64 = static_cast<int64_t>(r) * s;
    const int64_t rR64 = static_cast<int64_t>(rp) * sp;
    if (rL64 > std::numeric_limits<int>::max() || rR64 > std::numeric_limits<int>::max())
    {
      std::fprintf(stderr,
                   "matvec_core: rank product exceeds INT_MAX "
                   "(rL=%ld, rR=%ld); aborting.\n",
                   static_cast<long>(rL64), static_cast<long>(rR64));
      std::abort();
    }
    tt_core out(static_cast<int>(rL64), m, static_cast<int>(rR64));
    double* out_data         = out.data();
    const double* T_data     = T.data();
    const std::size_t Tcols  = static_cast<std::size_t>(s) * static_cast<std::size_t>(sp);
    const std::size_t slab   = static_cast<std::size_t>(sp) * sizeof(double);
    const std::size_t out_rs = static_cast<std::size_t>(rp) * static_cast<std::size_t>(sp);
    for (int a = 0; a < r; ++a)
      for (int i = 0; i < m; ++i)
        for (int b = 0; b < rp; ++b)
        {
          const double* trow =
            T_data +
            ((static_cast<std::size_t>(a) * m +
              static_cast<std::size_t>(i)) * rp +
             b) * Tcols;
          for (int ax = 0; ax < s; ++ax)
          {
            // dst = &out(a*s+ax, i, b*sp + 0)
            double* dst =
              out_data +
              ((static_cast<std::size_t>(a) * s + ax) * m + i) * out_rs +
              static_cast<std::size_t>(b) * sp;
            const double* src =
              trow + static_cast<std::size_t>(ax) * sp;
            std::memcpy(dst, src, slab);
          }
        }

    return out;
  }
}

// y_core(a*s+ax, i, j, b*sp+bx) = sum_p mc_A(a,i,p,b) * mc_B(ax,p,j,bx).
inline tt_matrix_core matmat_core(const tt_matrix_core& A,
                                  const tt_matrix_core& B)
{
  const int r  = A.r_left();
  const int m  = A.m_phys();
  const int p  = A.n_phys();
  const int rp = A.r_right();
  const int s  = B.r_left();
  const int pB = B.m_phys();
  const int n  = B.n_phys();
  const int sp = B.r_right();
  if (p != pB)
  {
    std::fprintf(stderr, "matmat: mode mismatch %d vs %d\n", p, pB);
    std::abort();
  }

  using row_mat = eigen_bridge::row_matrix;

  const Eigen::Index A_rows = static_cast<Eigen::Index>(r) * m * rp;
  const Eigen::Index B_cols = static_cast<Eigen::Index>(s) * n * sp;
  const Eigen::Index T_rows = A_rows;
  const Eigen::Index T_cols = B_cols;

  // Permute A(a,i,p,b) -> A_p((a,i,b), p).
  row_mat A_p(A_rows, p);
  for (int a = 0; a < r; ++a)
    for (int i = 0; i < m; ++i)
      for (int q = 0; q < p; ++q)
        for (int b = 0; b < rp; ++b)
          A_p((static_cast<Eigen::Index>(a) * m + i) * rp + b, q) =
            A(a, i, q, b);

  // Permute B(ax, p, j, bx) -> B_p(p, (ax, j, bx)).
  row_mat B_p(p, B_cols);
  for (int ax = 0; ax < s; ++ax)
    for (int q = 0; q < p; ++q)
      for (int j = 0; j < n; ++j)
        for (int bx = 0; bx < sp; ++bx)
          B_p(q,
              (static_cast<Eigen::Index>(ax) * n + j) * sp + bx) =
            B(ax, q, j, bx);

  // T((a,i,b), (ax,j,bx)) = sum_p A_p((a,i,b),p) * B_p(p,(ax,j,bx)).
  row_mat T(T_rows, T_cols);
  T.noalias() = A_p * B_p;

  // Repackage to out(a*s+ax, i, j, b*sp+bx).
  {
    const int64_t rL64 = static_cast<int64_t>(r) * s;
    const int64_t rR64 = static_cast<int64_t>(rp) * sp;
    if (rL64 > std::numeric_limits<int>::max() || rR64 > std::numeric_limits<int>::max())
    {
      std::fprintf(stderr,
                   "matmat_core: rank product exceeds INT_MAX "
                   "(rL=%ld, rR=%ld); aborting.\n",
                   static_cast<long>(rL64), static_cast<long>(rR64));
      std::abort();
    }
    tt_matrix_core out(static_cast<int>(rL64), m, n,
                        static_cast<int>(rR64));
    double* out_data         = out.data();
    const double* T_data     = T.data();
    const std::size_t Tcols  = static_cast<std::size_t>(s) * static_cast<std::size_t>(n)
                               * static_cast<std::size_t>(sp);
    const std::size_t slab   = static_cast<std::size_t>(sp) * sizeof(double);
    const std::size_t out_rs = static_cast<std::size_t>(rp) * static_cast<std::size_t>(sp);
    const std::size_t out_m  = static_cast<std::size_t>(m);
    const std::size_t out_n  = static_cast<std::size_t>(n);
    for (int a = 0; a < r; ++a)
      for (int i = 0; i < m; ++i)
        for (int b = 0; b < rp; ++b)
        {
          const double* trow =
            T_data +
            ((static_cast<std::size_t>(a) * out_m +
              static_cast<std::size_t>(i)) * rp +
             b) * Tcols;
          for (int ax = 0; ax < s; ++ax)
            for (int j = 0; j < n; ++j)
            {
              double* dst =
                out_data +
                ((static_cast<std::size_t>(a) * s + ax) * out_m + i) *
                  out_n * out_rs +
                static_cast<std::size_t>(j) * out_rs +
                static_cast<std::size_t>(b) * sp;
              const double* src =
                trow +
                (static_cast<std::size_t>(ax) * out_n + j) * sp;
              std::memcpy(dst, src, slab);
            }
        }

    return out;
  }
}

}  // namespace detail

/**
 * @brief Apply a TT-matrix to a TT vector (raw, no rounding).
 * Each output core contracts an A-core with an x-core along the
 * mode axis; rank axes are tensor-multiplied.  Result ranks are r_A[k] * r_x[k].
 * @param A TT-matrix operator.
 * @param x TT vector.
 * @return Rank-multiplying A*x.  Must be rounded before further use.
 * @note  Do NOT call in inner loops; use matvec_round instead.
 */
inline tt matvec(const tt_matrix& A, const tt& x)
{
  detail::check_matvec_compatible(A, x, "matvec");
  const int d = A.d();
  if (d == 0)
    return tt();
  std::vector<tt_core> cores;
  cores.reserve(static_cast<std::size_t>(d));
  for (int k = 0; k < d; ++k)
    cores.push_back(detail::matvec_core(A.core(k), x.core(k)));
  return tt(std::move(cores));
}
/**
 * @brief Multiply two TT-matrices (raw, no rounding).
 * Each output core contracts matching A and B cores along the inner
 * mode axis; rank axes are tensor-multiplied.
 * Result ranks are r_A[k] * r_B[k].
 * @param A Left TT-matrix operand.
 * @param B Right TT-matrix operand.
 * @return Rank-multiplying A*B.  Must be rounded before further use.
 * @note  Do NOT call in inner loops; use matmat_round instead.
 */

inline tt_matrix matmat(const tt_matrix& A, const tt_matrix& B)
{
  detail::check_matmat_compatible(A, B, "matmat");
  const int d = A.d();
  if (d == 0)
    return tt_matrix();
  std::vector<tt_matrix_core> cores;
  cores.reserve(static_cast<std::size_t>(d));
  for (int k = 0; k < d; ++k)
    cores.push_back(detail::matmat_core(A.core(k), B.core(k)));
  return tt_matrix(std::move(cores));
}

}  // namespace tensor_train
}  // namespace mva
//
// :D
//
