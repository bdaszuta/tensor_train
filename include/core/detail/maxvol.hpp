/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Greedy maxvol row selection and post-processing helpers
*/
#pragma once

#include <Eigen/Core>
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <vector>

namespace mva
{
namespace tensor_train
{
namespace detail
{

// A: rows x cols, rows >= cols.
// Returns a vector of ``cols`` row indices (sorted ascending) that
// approximately maximise the submatrix volume.
inline std::vector<int> maxvol(const Eigen::Ref<const Eigen::MatrixXd>& A,
                               double tol           = 1.05,
                               int max_iters        = 100,
                               int resolve_interval = 16)
{
  using Mat = Eigen::MatrixXd;
  using Vec = Eigen::VectorXd;

  const int rows = static_cast<int>(A.rows());
  const int cols = static_cast<int>(A.cols());
  if (cols == 0)
    return {};
  if (rows <= cols)
  {
    std::vector<int> idx(static_cast<std::size_t>(rows));
    for (int i = 0; i < rows; ++i)
      idx[static_cast<std::size_t>(i)] = i;
    return idx;
  }

  Mat B = A;

  // --- LU bootstrap: find ``cols`` rows giving a well-conditioned
  //     submatrix via pivoted LU on B ---
  Eigen::FullPivLU<Mat> lu(B);
  std::vector<int> piv(static_cast<std::size_t>(cols));
  for (int j = 0; j < cols; ++j)
    piv[static_cast<std::size_t>(j)] = static_cast<int>(
      lu.permutationP().indices()[static_cast<Eigen::Index>(j)]);

  // Pivot rows must be sorted for the Sherman-Morrison bookkeeping.
  std::sort(piv.begin(), piv.end());

  // Current submatrix and its LU (for volume check + solve).
  Mat C(cols, cols);
  for (int j = 0; j < cols; ++j)
    C.row(j) = B.row(piv[static_cast<std::size_t>(j)]);
  Eigen::FullPivLU<Mat> C_lu(C);

  // Row-index set for fast "is in pivot set?" lookup.
  std::vector<bool> in_set(static_cast<std::size_t>(rows), false);
  for (int j = 0; j < cols; ++j)
    in_set[static_cast<std::size_t>(piv[static_cast<std::size_t>(j)])] = true;

  // Per-iteration inverse, updated via SM between resolve intervals.
  Mat C_inv = C_lu.inverse();

  // --- Sherman-Morrison swap loop ---
  for (int iter = 0; iter < max_iters; ++iter)
  {
    // Re-solve every 'resolve_interval' iterations to avoid SM drift.
    if (iter > 0 && iter % resolve_interval == 0)
    {
      for (int j = 0; j < cols; ++j)
        C.row(j) = B.row(piv[static_cast<std::size_t>(j)]);
      C_lu.compute(C);
      C_inv = C_lu.inverse();
    }

    // Rows of B not in the pivot set, transported into the space where
    // maxvol pivots are identity.
    int best_i      = -1;
    int best_j      = -1;
    double best_val = 1.0;

    for (int j_idx = 0; j_idx < cols; ++j_idx)
    {
      for (int i = 0; i < rows; ++i)
      {
        if (in_set[static_cast<std::size_t>(i)])
          continue;
        // (B[i,:] * C^{-1})[j_idx] = B.row(i) dot C_inv.col(j_idx).
        double val  = B.row(i).dot(C_inv.col(j_idx));
        double aval = std::abs(val);
        if (aval > best_val)
        {
          best_val = aval;
          best_i   = i;
          best_j   = j_idx;
        }
      }
    }

    if (best_val <= tol)
      break;

    // Swap: pivot row piv[best_j] out, best_i in.
    const int old_row = piv[static_cast<std::size_t>(best_j)];
    piv[static_cast<std::size_t>(best_j)]     = best_i;
    in_set[static_cast<std::size_t>(old_row)] = false;
    in_set[static_cast<std::size_t>(best_i)]  = true;

    // Sherman-Morrison update: C^{-1} with row best_j replaced by
    // B.row(best_i).
    // C_new = C + e_j*u^T  =>  C_new^{-1} = C^{-1} - (C^{-1}e_j)(u^T
    // C^{-1})/(1 + u^T C^{-1}e_j) where u = B.row(best_i) - C.row(best_j).
    const Vec u                      = B.row(best_i) - C.row(best_j);
    const Vec C_ej                   = C_inv.col(best_j);
    const Eigen::RowVectorXd uT_Cinv = u.transpose() * C_inv;
    const double denom               = 1.0 + u.dot(C_inv.col(best_j));
    if (std::abs(denom) < 1e-15)
    {
      C.row(best_j) = B.row(best_i);
      C_lu.compute(C);
      C_inv = C_lu.inverse();
    }
    else
    {
      C_inv -= (C_ej * uT_Cinv) / denom;
      C.row(best_j) = B.row(best_i);
    }
  }

  return piv;
}

// Column-pivoted QR on M^T to select r_new dominant rows of M.
// Unlike maxvol (which picks pivot rows from an abstract basis such as
// singular vectors), this directly selects rows that capture M's actual
// variation.  This avoids false-fixed-point issues in cross-approximation
// when singular values have large gaps.
//
// M: (rows x cols) matrix.  Returns r_new row indices (not sorted).
inline std::vector<int> qr_pivots(
    const Eigen::Ref<const Eigen::MatrixXd>& M,
    int r_new)
{
  const int rows = static_cast<int>(M.rows());
  const int cols = static_cast<int>(M.cols());
  if (rows <= 0 || cols <= 0 || r_new <= 0)
    return {};

  // QR with column pivoting on M^T: pivot columns of M^T = pivot rows of M.
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(M.transpose());
  const auto& piv = qr.colsPermutation().indices();

  const int n_piv = std::min(r_new, rows);
  std::vector<int> result(static_cast<std::size_t>(n_piv));
  for (int j = 0; j < n_piv; ++j)
    result[static_cast<std::size_t>(j)] = static_cast<int>(piv[static_cast<Eigen::Index>(j)]);

  return result;
}

// Post-process pivot set to ensure every physical index i in [0, n_phys)
// appears at least once.  Without this, maxvol can select pivots that span
// the SVD subspace but miss entire physical modes -- the TT then cannot
// represent the function at those modes.
//
// U_block: rows = rL * n_phys, cols = r_new.  Row (a,i) = a*n_phys + i.
// piv: current maxvol-selected row indices (r_new entries).
// Returns updated piv (may be mutated in place).
inline void enforce_phys_coverage(
    std::vector<int>& piv,
    const Eigen::Ref<const Eigen::MatrixXd>& U_block,
    int n_phys)
{
  const int r_new = static_cast<int>(piv.size());
  if (r_new < n_phys)
    return;  // cannot cover all physical modes with fewer pivots than modes

  const int rows = static_cast<int>(U_block.rows());
  const int rL   = rows / n_phys;

  // Which physical indices are covered?
  std::vector<bool> covered(static_cast<std::size_t>(n_phys), false);
  for (int j = 0; j < r_new; ++j)
    covered[static_cast<std::size_t>(piv[static_cast<std::size_t>(j)] % n_phys)] = true;

  // Build the current submatrix and its inverse for volume contribution.
  Eigen::MatrixXd C(r_new, r_new);
  for (int j = 0; j < r_new; ++j)
    C.row(j) = U_block.row(piv[static_cast<std::size_t>(j)]);
  Eigen::MatrixXd C_inv;
  // Guard against singular C.  |det| > 1e-30 catches near-singular
  // matrices; the safe fallback is to return without enforcing coverage.
  if (std::abs(C.determinant()) > 1e-30)
    C_inv = C.inverse();
  else
    return;  // numerically singular; give up on coverage enforcement

  for (int i = 0; i < n_phys; ++i)
  {
    if (covered[static_cast<std::size_t>(i)])
      continue;

    // Find the row with physical index i that has the largest norm.
    int    best_row  = -1;
    double best_norm = -1.0;
    for (int a = 0; a < rL; ++a)
    {
      const int row_idx = a * n_phys + i;
      // Skip rows already in the pivot set.
      bool already = false;
      for (int j = 0; j < r_new; ++j)
        if (piv[static_cast<std::size_t>(j)] == row_idx)
        { already = true; break; }
      if (already) continue;

      double nr = U_block.row(row_idx).norm();
      if (nr > best_norm) { best_norm = nr; best_row = row_idx; }
    }
    if (best_row < 0)
      continue;  // no uncovered row available (should not happen)

    // Find the current pivot that contributes LEAST to the submatrix
    // volume.  Contribution ~ 1 / ||C_inv.row(j)||.
    int    worst_j  = -1;
    double worst_contrib = 1e99;
    for (int j = 0; j < r_new; ++j)
    {
      // Don't remove a pivot if it's the ONLY one covering its phys index.
      int pi = piv[static_cast<std::size_t>(j)] % n_phys;
      int count_pi = 0;
      for (int jj = 0; jj < r_new; ++jj)
        if (piv[static_cast<std::size_t>(jj)] % n_phys == pi)
          ++count_pi;
      if (count_pi <= 1)
        continue;

      double contrib = 1.0 / C_inv.row(j).norm();
      if (contrib < worst_contrib) { worst_contrib = contrib; worst_j = j; }
    }
    if (worst_j < 0)
    {
      // All pivots are sole-covering their physical index.
      // Replacing any pivot would break coverage -- nothing to improve.
      continue;
    }

    // Replace pivot worst_j with best_row.
    piv[static_cast<std::size_t>(worst_j)] = best_row;
    covered[static_cast<std::size_t>(i)]   = true;

    // Recompute C and C_inv for the next iteration.
    for (int j = 0; j < r_new; ++j)
      C.row(j) = U_block.row(piv[static_cast<std::size_t>(j)]);
    C_inv = C.inverse();
  }
}

}  // namespace detail
}  // namespace tensor_train
}  // namespace mva
//
// :D
//
