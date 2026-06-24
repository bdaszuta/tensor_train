/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Fused apply-and-absorb kernels for the streaming round path
*/
#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "../types/tt_core.hpp"
#include "../types/tt_eigen_bridge.hpp"
#include "../types/tt_matrix.hpp"

namespace mva
{
namespace tensor_train
{
namespace detail
{

// Fused matvec: F[a, ax] -> A_k[a,i,j,b] -> x_k[ax,j,bx] -> out[a*ax', i, b*bx']
// via two permutes and two GEMMs.  The F-environment tensor encodes
// previously accumulated left contractions.
inline tt_core fused_matvec_core(const eigen_bridge::row_matrix& F,
                                 const tt_matrix_core& A_k,
                                 const tt_core& x_k)
{
  using row_mat = eigen_bridge::row_matrix;

  const int alpha = static_cast<int>(F.rows());
  const int a     = A_k.r_left();
  const int m     = A_k.m_phys();
  const int n     = A_k.n_phys();
  const int b     = A_k.r_right();
  const int sx    = x_k.r_left();
  const int nx    = x_k.n_phys();
  const int bx    = x_k.r_right();

  if (n != nx)
  {
    std::fprintf(stderr, "fused_matvec: mode mismatch %d vs %d\n", n, nx);
    std::abort();
  }
  if (static_cast<int64_t>(F.cols()) != static_cast<int64_t>(a) * sx)
  {
    std::fprintf(stderr,
                 "fused_matvec: F cols %lld != a*sx %lld\n",
                 static_cast<long long>(F.cols()),
                 static_cast<long long>(static_cast<int64_t>(a) * sx));
    std::abort();
  }

  // Overflow guard: hoisted before any GEMM or permute.
  {
    const int64_t nphys64 = static_cast<int64_t>(b) * bx;
    if (nphys64 > std::numeric_limits<int>::max() || nphys64 < 0)
    {
      std::fprintf(stderr,
                   "fused_matvec_core: b*bx product exceeds INT_MAX "
                   "(b=%d * bx=%d = %ld)\n",
                   b, bx, static_cast<long>(nphys64));
      std::abort();
    }
  }

  const Eigen::Index alpha_sx = static_cast<Eigen::Index>(alpha) * sx;
  const Eigen::Index m_n_b = static_cast<Eigen::Index>(m) * n * b;
  const Eigen::Index alpha_m_b = static_cast<Eigen::Index>(alpha) * m * b;
  const Eigen::Index sx_n = static_cast<Eigen::Index>(sx) * n;

  // Permute 1: Fp(alpha*sx, a) from F(alpha, a*sx)
  // with F(alpha, a*sx + ax) -> Fp(alpha*sx + ax, a).
  row_mat Fp(alpha_sx, a);
  for (int al = 0; al < alpha; ++al)
    for (int aa = 0; aa < a; ++aa)
      for (int ax = 0; ax < sx; ++ax)
        Fp(static_cast<Eigen::Index>(al) * sx + ax, aa) =
          F(al, static_cast<Eigen::Index>(aa) * sx + ax);

  // GEMM 1: M1(alpha*sx, m*n*b) = Fp(alpha*sx, a) * A_view(a, m*n*b).
  Eigen::Map<const row_mat> A_view(A_k.data(), a, m_n_b);
  row_mat M1(alpha_sx, m_n_b);
  M1.noalias() = Fp * A_view;
  // M1 row-major axes: (alpha, ax, i, j, b).

  // Permute 2: M1 (alpha, ax, i, j, b) -> M1p (alpha, i, b, ax, j)
  // viewed as (alpha*m*b, sx*n).
  row_mat M1p(alpha_m_b, sx_n);
  const std::size_t M1_rs =
    static_cast<std::size_t>(m) * static_cast<std::size_t>(n) * b;
  const double* M1_data = M1.data();
  for (int al = 0; al < alpha; ++al)
    for (int ax = 0; ax < sx; ++ax)
    {
      const double* src_row =
        M1_data +
        (static_cast<std::size_t>(al) * static_cast<std::size_t>(sx) + ax) *
          M1_rs;
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
          for (int bb = 0; bb < b; ++bb)
            M1p((static_cast<Eigen::Index>(al) * m + i) * b + bb,
                static_cast<Eigen::Index>(ax) * n + j) =
              src_row[(static_cast<std::size_t>(i) *
                         static_cast<std::size_t>(n) +
                       j) *
                        b +
                      bb];
    }

  // GEMM 2: out_view(alpha*m*b, bx) = M1p(alpha*m*b, sx*n)
  //                                 * x_view(sx*n, bx).
  // x_k stored (ax, j, bx) row-major - direct view as (sx*n, bx).
  Eigen::Map<const row_mat> x_view(x_k.data(), sx_n, bx);
  tt_core out(alpha, m, b * bx);
  Eigen::Map<row_mat> out_view(out.data(), alpha_m_b, bx);
  out_view.noalias() = M1p * x_view;
  // out_view row-major (alpha, i, b, bx) == out tt_core layout
  // (alpha, i, b*bx) since the inner two trailing axes flatten.

  return out;
}

// Fused matmat: F[a, ax] -> A_k[a,m,p,b] -> B_k[ax,p,n,bx] -> out[a*ax', m, n, b*bx']
// via three permutes and two GEMMs.  The F-environment tensor encodes
// previously accumulated left contractions.
inline tt_matrix_core fused_matmat_core(const eigen_bridge::row_matrix& F,
                                        const tt_matrix_core& A_k,
                                        const tt_matrix_core& B_k)
{
  using row_mat = eigen_bridge::row_matrix;

  const int alpha = static_cast<int>(F.rows());
  const int a     = A_k.r_left();
  const int m     = A_k.m_phys();
  const int p     = A_k.n_phys();
  const int b     = A_k.r_right();
  const int sx    = B_k.r_left();
  const int pB    = B_k.m_phys();
  const int n     = B_k.n_phys();
  const int bx    = B_k.r_right();

  if (p != pB)
  {
    std::fprintf(stderr, "fused_matmat: mode mismatch %d vs %d\n", p, pB);
    std::abort();
  }
  if (static_cast<int64_t>(F.cols()) != static_cast<int64_t>(a) * sx)
  {
    std::fprintf(stderr,
                 "fused_matmat: F cols %lld != a*sx %lld\n",
                 static_cast<long long>(F.cols()),
                 static_cast<long long>(static_cast<int64_t>(a) * sx));
    std::abort();
  }

  // Overflow guard: hoisted before any GEMM or permute.
  {
    const int64_t nphys64 = static_cast<int64_t>(b) * bx;
    if (nphys64 > std::numeric_limits<int>::max() || nphys64 < 0)
    {
      std::fprintf(stderr,
                   "fused_matmat_core: b*bx product exceeds INT_MAX "
                   "(b=%d * bx=%d = %ld)\n",
                   b, bx, static_cast<long>(nphys64));
      std::abort();
    }
  }

  const Eigen::Index alpha_sx = static_cast<Eigen::Index>(alpha) * sx;
  const Eigen::Index m_p_b = static_cast<Eigen::Index>(m) * p * b;
  const Eigen::Index alpha_m_b = static_cast<Eigen::Index>(alpha) * m * b;
  const Eigen::Index sx_p = static_cast<Eigen::Index>(sx) * p;
  const Eigen::Index n_bx = static_cast<Eigen::Index>(n) * bx;
  const Eigen::Index rn_bx = static_cast<Eigen::Index>(alpha) * m * b;
  const Eigen::Index cn_bx = static_cast<Eigen::Index>(n) * bx;

  // Permute 1: Fp(alpha*sx, a) [same as matvec].
  row_mat Fp(alpha_sx, a);
  for (int al = 0; al < alpha; ++al)
    for (int aa = 0; aa < a; ++aa)
      for (int ax = 0; ax < sx; ++ax)
        Fp(static_cast<Eigen::Index>(al) * sx + ax, aa) =
          F(al, static_cast<Eigen::Index>(aa) * sx + ax);

  // GEMM 1: M1(alpha*sx, m*p*b) = Fp(alpha*sx, a) * A_view(a, m*p*b).
  Eigen::Map<const row_mat> A_view(A_k.data(), a, m_p_b);
  row_mat M1(alpha_sx, m_p_b);
  M1.noalias() = Fp * A_view;
  // M1 row-major axes: (alpha, ax, i, q, b)  where q is A's phys (=p).

  // Permute 2: M1 (alpha, ax, i, q, b) -> M1p (alpha, i, b, ax, q)
  // viewed as (alpha*m*b, sx*p).
  row_mat M1p(alpha_m_b, sx_p);
  const std::size_t M1_rs =
    static_cast<std::size_t>(m) * static_cast<std::size_t>(p) * b;
  const double* M1_data = M1.data();
  for (int al = 0; al < alpha; ++al)
    for (int ax = 0; ax < sx; ++ax)
    {
      const double* src_row =
        M1_data +
        (static_cast<std::size_t>(al) * static_cast<std::size_t>(sx) + ax) *
          M1_rs;
      for (int i = 0; i < m; ++i)
        for (int q = 0; q < p; ++q)
          for (int bb = 0; bb < b; ++bb)
            M1p((static_cast<Eigen::Index>(al) * m + i) * b + bb,
                static_cast<Eigen::Index>(ax) * p + q) =
              src_row[(static_cast<std::size_t>(i) *
                         static_cast<std::size_t>(p) +
                       q) *
                        b +
                      bb];
    }

  // GEMM 2: R(alpha*m*b, n*bx) = M1p(alpha*m*b, sx*p)
  //                            * B_view(sx*p, n*bx).
  // B_k stored (ax, q, j, bx) row-major - direct view as (sx*p, n*bx).
  Eigen::Map<const row_mat> B_view(B_k.data(), sx_p, n_bx);
  row_mat R(rn_bx, cn_bx);
  R.noalias() = M1p * B_view;
  // R row-major axes: (alpha, i, b, j, bx).

  // Permute 3: pack R (alpha, i, b, j, bx) into out (alpha, i, j, b*bx)
  // tt_matrix_core layout (alpha, i, j, b, bx) row-major; innermost
  // (b, bx) slab pairs, so this is a strided rearrangement, not a
  // single memcpy.  Slab of bx doubles is contiguous in BOTH R (last
  // col axis) and out (last linear axis under fixed (alpha, i, j, b)).
  tt_matrix_core out(alpha, m, n, b * bx);
  double* out_data        = out.data();
  const double* R_data    = R.data();
  const std::size_t Rcols = static_cast<std::size_t>(n) * static_cast<std::size_t>(bx);
  const std::size_t slab  = static_cast<std::size_t>(bx) * sizeof(double);
  const std::size_t orsbx = static_cast<std::size_t>(b) * static_cast<std::size_t>(bx);
  const std::size_t out_n = static_cast<std::size_t>(n);
  const std::size_t out_m = static_cast<std::size_t>(m);
  for (int al = 0; al < alpha; ++al)
    for (int i = 0; i < m; ++i)
      for (int bb = 0; bb < b; ++bb)
      {
        const double* rrow =
          R_data +
          ((static_cast<std::size_t>(al) * out_m +
            static_cast<std::size_t>(i)) *
             static_cast<std::size_t>(b) +
           bb) * Rcols;
        for (int j = 0; j < n; ++j)
        {
          // dst = &out(al, i, j, bb*bx + 0):
          //   linear = (((al*m + i)*n + j) * orsbx) + bb*bx
          double* dst = out_data +
                        ((static_cast<std::size_t>(al) * out_m + i) * out_n +
                         static_cast<std::size_t>(j)) *
                          orsbx +
                        static_cast<std::size_t>(bb) * bx;
          const double* src = rrow + static_cast<std::size_t>(j) * bx;
          std::memcpy(dst, src, slab);
        }
      }

  return out;
}

}  // namespace detail
}  // namespace tensor_train
}  // namespace mva
//
// :D
//
