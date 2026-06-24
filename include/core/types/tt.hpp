/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tensor-Train container: ordered list of tt_core objects
*/
#pragma once

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <vector>

#include "tt_core.hpp"
#include "tt_eigen_bridge.hpp"

namespace mva
{
namespace tensor_train
{

class tt
{
  public:
  /** @brief Default-constructed empty TT (d = 0). */
  tt() = default;

  /**
   * @brief Construct a TT from a vector of cores.
   * Validates boundary ranks (r_0 = r_d = 1) and bond consistency.
   * @param cores Ordered list of tt_core objects.
   */
  explicit tt(std::vector<tt_core> cores) : cores_(std::move(cores))
  {
    const int d = static_cast<int>(cores_.size());
    if (d > 0)
    {
      if (cores_[0].r_left() != 1)
      {
        std::fprintf(stderr,
                     "tt: r_0 must be 1 (got %d)\n",
                     cores_[0].r_left());
        std::abort();
      }
      if (cores_[static_cast<std::size_t>(d - 1)].r_right() != 1)
      {
        std::fprintf(stderr,
                     "tt: r_d must be 1 (got %d)\n",
                     cores_[static_cast<std::size_t>(d - 1)].r_right());
        std::abort();
      }
      for (int k = 0; k < d - 1; ++k)
      {
        if (cores_[static_cast<std::size_t>(k)].r_right()
            != cores_[static_cast<std::size_t>(k + 1)].r_left())
        {
          std::fprintf(stderr,
                       "tt: rank mismatch at bond %d: "
                       "cores_[%d].r_right=%d != cores_[%d].r_left=%d\n",
                       k + 1,
                       k, cores_[static_cast<std::size_t>(k)].r_right(),
                       k + 1,
                       cores_[static_cast<std::size_t>(k + 1)].r_left());
          std::abort();
        }
      }
    }
  }

  /**
   * @brief Number of modes (cores).  Returns 0 for default-constructed TT.
   */
  int d() const
  {
    return static_cast<int>(cores_.size());
  }

  /**
   * @brief Mode-size shape: vector of n_k for k = 0..d-1.
   */
  std::vector<int> shape() const
  {
    std::vector<int> s;
    s.reserve(cores_.size());
    for (const auto& c : cores_)
    {
      s.push_back(c.n_phys());
    }
    return s;
  }

  /**
   * @brief Bond ranks r_0, r_1, ..., r_d (length d+1).
   * r_0 = r_d = 1 by convention.
   */
  std::vector<int> ranks() const
  {
    std::vector<int> rs;
    rs.reserve(cores_.size() + 1);
    if (cores_.empty())
    {
      rs.push_back(1);
      return rs;
    }
    rs.push_back(cores_.front().r_left());
    for (const auto& c : cores_)
    {
      rs.push_back(c.r_right());
    }
    return rs;
  }

  /**
   * @brief Maximum bond rank across all bonds.
   */
  int max_rank() const
  {
    int mr = 0;
    for (int r : ranks())
    {
      if (r > mr)
        mr = r;
    }
    return mr;
  }

  /**
   * @brief Total number of stored double-precision parameters.
   */
  std::size_t num_params() const
  {
    std::size_t total = 0;
    for (const auto& c : cores_)
    {
      total += static_cast<std::size_t>(c.size());
    }
    return total;
  }

  /**
   * @brief Reconstruct the full dense tensor as a flat row-major vector.
   */
  std::vector<double> to_dense() const
  {
    using row_matrix = eigen_bridge::row_matrix;
    using map_c      = Eigen::Map<const row_matrix>;
    const int dd     = d();
    if (dd == 0)
      return {1.0};  // d=0 is the scalar 1 (empty tensor product)
    if (cores_.front().r_left() != 1 || cores_.back().r_right() != 1)
    {
      std::fprintf(stderr,
                   "to_dense: boundary ranks must be 1 "
                   "(r_left=%d, r_right=%d); aborting.\n",
                   cores_.front().r_left(),
                   cores_.back().r_right());
      std::abort();
    }
    const tt_core& c0 = cores_.front();
    Eigen::Index rows = c0.n_phys();
    Eigen::Index cols = c0.r_right();
    // c0 has shape (1, n_0, r_1); raw buffer is already (n_0, r_1) row-major.
    row_matrix T(rows, cols);
    std::memcpy(T.data(),
                c0.data(),
                sizeof(double) * static_cast<std::size_t>(rows) * cols);
    for (int k = 1; k < dd; ++k)
    {
      const tt_core& ck = cores_[k];
      const int r_k     = ck.r_left();
      const int n_phys  = ck.n_phys();
      const int r_kp1   = ck.r_right();
      // T (rows x r_k) * Ck-right-unfold (r_k x n_phys*r_kp1)
      //                        -> (rows x n_phys*r_kp1)
      // then reinterpret as (rows*n_phys x r_kp1).
      const Eigen::Index rk_unfold = static_cast<Eigen::Index>(n_phys) * r_kp1;
      row_matrix Tn(rows, rk_unfold);
      map_c ck_right(ck.data(), r_k, rk_unfold);
      Tn.noalias()       = T * ck_right;
      const Eigen::Index new_rows = rows * static_cast<Eigen::Index>(n_phys);
      const Eigen::Index new_cols = r_kp1;
      // Copy Tn into a freshly-shaped T (rows*n_phys, r_kp1); same total
      // size and same row-major order, so memcpy is sufficient.
      T.resize(new_rows, new_cols);
      std::memcpy(
        T.data(),
        Tn.data(),
        sizeof(double) * static_cast<std::size_t>(new_rows) * new_cols);
      rows = new_rows;
      cols = new_cols;
    }
    // Final cols == r_d == 1, so flat T already has length prod(n_k).
    std::vector<double> out(static_cast<std::size_t>(rows) * cols);
    std::memcpy(out.data(), T.data(), sizeof(double) * out.size());
    return out;
  }

  /**
   * @brief Access core k (read/write).
   * @param k Core index, 0 <= k < d.
   */
  tt_core& core(int k)
  {
    if (k < 0 || k >= static_cast<int>(cores_.size()))
    {
      std::fprintf(stderr,
                   "tt::core: index %d out of range [0, %zu)\n",
                   k,
                   cores_.size());
      std::abort();
    }
    return cores_[static_cast<std::size_t>(k)];
  }
  const tt_core& core(int k) const
  {
    if (k < 0 || k >= static_cast<int>(cores_.size()))
    {
      std::fprintf(stderr,
                   "tt::core: index %d out of range [0, %zu)\n",
                   k,
                   cores_.size());
      std::abort();
    }
    return cores_[static_cast<std::size_t>(k)];
  }

  /**
   * @brief Mutable access to the core vector.
   * @warning Callers must maintain TT invariants after mutation.
   */
  std::vector<tt_core>& cores()
  {
    return cores_;
  }
  const std::vector<tt_core>& cores() const
  {
    return cores_;
  }

  private:
  std::vector<tt_core> cores_;
};

}  // namespace tensor_train
}  // namespace mva
//
// :D
//
