/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Lightweight Eigen type aliases and utility functions for tensor-train operations (QR, thin-SVD, rank truncation)
*/
#pragma once

#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <algorithm>

namespace mva
{
namespace tensor_train
{
namespace eigen_bridge
{

using row_matrix =
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using col_vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;

}  // namespace eigen_bridge

namespace detail
{

enum class svd_part
{
  full,
  U_only,
  V_only
};

// Thin QR of M (rows x cols).  Produces:
//   Q : rows x k row-major, with k = min(rows, cols), columns orthonormal.
//   R : k    x cols row-major, upper-triangular in its leading k x k block.
inline void qr_thin(const double* M_data,
                    Eigen::Index rows,
                    Eigen::Index cols,
                    eigen_bridge::row_matrix& Q,
                    eigen_bridge::row_matrix& R)
{
  if (rows <= 0 || cols <= 0)
  {
    Q.resize(0, 0);
    R.resize(0, 0);
    return;
  }
  Eigen::Map<const eigen_bridge::row_matrix> M(M_data, rows, cols);
  Eigen::HouseholderQR<eigen_bridge::row_matrix> qr(M);
  const Eigen::Index k = std::min(rows, cols);
  Q.setIdentity(rows, k);
  Q.applyOnTheLeft(qr.householderQ());
  R = qr.matrixQR().topRows(k).template triangularView<Eigen::Upper>();
}

// Thin SVD of M (rows x cols).  Produces:
//   U  : rows x k row-major.
//   s  : length-k vector of singular values (descending).
//   Vt : k    x cols row-major.
// where k = min(rows, cols).
inline void svd_thin(const double* M_data,
                     Eigen::Index rows,
                     Eigen::Index cols,
                     eigen_bridge::row_matrix& U,
                     eigen_bridge::col_vector& s,
                     eigen_bridge::row_matrix& Vt)
{
  if (rows <= 0 || cols <= 0)
  {
    U.resize(0, 0);
    s.resize(0);
    Vt.resize(0, 0);
    return;
  }
  Eigen::Map<const eigen_bridge::row_matrix> M(M_data, rows, cols);

  // Primary path: BDCSVD (fast).
  {
    Eigen::BDCSVD<eigen_bridge::row_matrix,
                  Eigen::ComputeThinU | Eigen::ComputeThinV>
      svd(M);
    if (svd.info() == Eigen::Success)
    {
      U  = svd.matrixU();
      s  = svd.singularValues();
      Vt = svd.matrixV().transpose();
      return;
    }
  }

  // Fallback: JacobiSVD (robust, guaranteed convergence).
  Eigen::JacobiSVD<eigen_bridge::row_matrix,
                   Eigen::ComputeThinU | Eigen::ComputeThinV>
    svd2(M);
  U  = svd2.matrixU();
  s  = svd2.singularValues();
  Vt = svd2.matrixV().transpose();
}

// Thin SVD with part-select.  svd_part::U_only skips Vt computation;
// svd_part::V_only skips U.  svd_part::full == the 5-arg overload.
inline void svd_thin(const double* M_data,
                     Eigen::Index rows,
                     Eigen::Index cols,
                     svd_part part,
                     eigen_bridge::row_matrix& U,
                     eigen_bridge::col_vector& s,
                     eigen_bridge::row_matrix& Vt)
{
  if (rows <= 0 || cols <= 0)
  {
    U.resize(0, 0);
    s.resize(0);
    Vt.resize(0, 0);
    return;
  }
  Eigen::Map<const eigen_bridge::row_matrix> M(M_data, rows, cols);
  if (part == svd_part::U_only)
  {
    Vt.resize(0, 0);  // not computed
    // Primary: BDCSVD.
    {
      Eigen::BDCSVD<eigen_bridge::row_matrix, Eigen::ComputeThinU> svd(M);
      if (svd.info() == Eigen::Success)
      {
        U = svd.matrixU();
        s = svd.singularValues();
        return;
      }
    }
    // Fallback: JacobiSVD.
    Eigen::JacobiSVD<eigen_bridge::row_matrix, Eigen::ComputeThinU> svd2(M);
    U = svd2.matrixU();
    s = svd2.singularValues();
    return;
  }
  if (part == svd_part::V_only)
  {
    U.resize(0, 0);  // not computed
    // Primary: BDCSVD.
    {
      Eigen::BDCSVD<eigen_bridge::row_matrix, Eigen::ComputeThinV> svd(M);
      if (svd.info() == Eigen::Success)
      {
        s  = svd.singularValues();
        Vt = svd.matrixV().transpose();
        return;
      }
    }
    // Fallback: JacobiSVD.
    Eigen::JacobiSVD<eigen_bridge::row_matrix, Eigen::ComputeThinV> svd2(M);
    s  = svd2.singularValues();
    Vt = svd2.matrixV().transpose();
    return;
  }
  // full: delegate to the 5-arg overload (or invalid enum value;
  // treat as full SVD for forward compatibility).
  svd_thin(M_data, rows, cols, U, s, Vt);
}

// Determine the truncation rank for a singular-value vector ``s`` such
// that the dropped Frobenius tail is at most ``delta``:
//   sum_{i >= r_new} s_i^2 <= delta^2.
// Always returns at least 1.  If ``max_rank > 0``, the result is capped
// at ``max_rank`` (and still at least 1).
inline int truncate_eps_rank(const eigen_bridge::col_vector& s,
                             double delta,
                             int max_rank = 0)
{
  // s.size() is Eigen::Index (ptrdiff_t); TT singular-value vectors
  // are small (bond dimensions), so int narrowing is safe.
  const int n = static_cast<int>(s.size());
  if (n == 0)
    return 1;
  const double thresh = delta * delta;
  double tail2        = 0.0;
  int r_new           = n;
  for (int i = n - 1; i >= 0; --i)
  {
    const double next = tail2 + s[i] * s[i];
    if (next > thresh)
      break;
    tail2 = next;
    r_new = i;
  }
  if (r_new < 1)
    r_new = 1;
  if (max_rank > 0 && r_new > max_rank)
    r_new = max_rank;
  return r_new;
}

}  // namespace detail
}  // namespace tensor_train
}  // namespace mva
//
// :D
//
