/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: TT-matrix core and TT-matrix (MPO) container
*/
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "../detail/storage_backend.hpp"
#include "tt_eigen_bridge.hpp"

namespace mva
{
namespace tensor_train
{

/**
 * @brief Single TT-matrix (MPO) core wrapping a 4-axis narray.
 *
 * Shape: (r_left, m_phys, n_phys, r_right) row-major.
 * Linear storage: ((a * m_phys + i) * n_phys + j) * r_right + b.
 * Bit-identical to a 3-axis tt_core of mode size m_phys * n_phys,
 * allowing pack/unpack via memcpy.
 */
class tt_matrix_core
{
  public:
  tt_matrix_core() = default;

  /**
   * @brief Construct a TT-matrix core with given dimensions.
   * Allocates row-major storage for shape (r_left, m_phys, n_phys, r_right).
   * @param r_left  Left bond dimension (must be positive).
   * @param m_phys  Row-mode (input) size (must be positive).
   * @param n_phys  Column-mode (output) size (must be positive).
   * @param r_right Right bond dimension (must be positive).
   */
  tt_matrix_core(int r_left, int m_phys, int n_phys, int r_right)
  {
    if (r_left <= 0 || m_phys <= 0 || n_phys <= 0 || r_right <= 0)
    {
      std::fprintf(stderr,
                   "tt_matrix_core: dimensions must be positive "
                   "(got %d, %d, %d, %d)\n",
                   r_left, m_phys, n_phys, r_right);
      std::abort();
    }
    {
      const int64_t rl64 = static_cast<int64_t>(r_left);
      const int64_t mp64 = static_cast<int64_t>(m_phys);
      const int64_t np64 = static_cast<int64_t>(n_phys);
      const int64_t rr64 = static_cast<int64_t>(r_right);
      const int64_t int_max64 = static_cast<int64_t>(
        std::numeric_limits<int>::max());
      const int64_t mp_np = mp64 * np64;
      if (mp_np > int_max64 / rr64
          || rl64 > int_max64 / (mp_np * rr64))
      {
        const int64_t total64 = rl64 * mp_np * rr64;
        std::fprintf(stderr,
                     "tt_matrix_core: dimension product "
                     "%ld * %d * %d * %d = %ld exceeds INT_MAX\n",
                     static_cast<long>(r_left), m_phys, n_phys,
                     r_right, static_cast<long>(total64));
        std::abort();
      }
    }
    arr_.allocate(r_left, m_phys, n_phys, r_right);
  }

  // Value-semantic: narray defines copy/move; we just default through.
  tt_matrix_core(const tt_matrix_core&)            = default;
  tt_matrix_core(tt_matrix_core&&)                 = default;
  tt_matrix_core& operator=(const tt_matrix_core&) = default;
  tt_matrix_core& operator=(tt_matrix_core&&)      = default;

  /**
   * @brief Allocate storage for shape (r_left, m_phys, n_phys, r_right).
   * Deallocates any previous allocation first.
   */
  void allocate(int r_left, int m_phys, int n_phys, int r_right)
  {
    if (r_left <= 0 || m_phys <= 0 || n_phys <= 0 || r_right <= 0)
    {
      std::fprintf(stderr,
                   "tt_matrix_core::allocate: dimensions must be positive "
                   "(got %d, %d, %d, %d)\n",
                   r_left, m_phys, n_phys, r_right);
      std::abort();
    }
    {
      const int64_t rl64 = static_cast<int64_t>(r_left);
      const int64_t mp64 = static_cast<int64_t>(m_phys);
      const int64_t np64 = static_cast<int64_t>(n_phys);
      const int64_t rr64 = static_cast<int64_t>(r_right);
      const int64_t int_max64 = static_cast<int64_t>(
        std::numeric_limits<int>::max());
      const int64_t mp_np = mp64 * np64;
      if (mp_np > int_max64 / rr64
          || rl64 > int_max64 / (mp_np * rr64))
      {
        const int64_t total64 = rl64 * mp_np * rr64;
        std::fprintf(stderr,
                     "tt_matrix_core::allocate: dimension product "
                     "%ld * %d * %d * %d = %ld exceeds INT_MAX\n",
                     static_cast<long>(r_left), m_phys, n_phys,
                     r_right, static_cast<long>(total64));
        std::abort();
      }
    }
    if (arr_.get_status() == detail::storage_status::allocated)
    {
      arr_.deallocate();
    }
    arr_.allocate(r_left, m_phys, n_phys, r_right);
  }

  /**
   * @brief Deallocate storage.  Core becomes unallocated.
   */
  void deallocate()
  {
    if (arr_.get_status() == detail::storage_status::allocated)
    {
      arr_.deallocate();
    }
  }

  /**
   * @brief Zero all elements.  Core must be allocated.
   */
  void zero_clear()
  {
    if (arr_.get_status() != detail::storage_status::allocated)
    {
      std::fprintf(stderr,
                   "tt_matrix_core::zero_clear: core is not allocated\n");
      std::abort();
    }
    arr_.zero_clear();
  }

  /**
   * @brief Mutable element access (a in [0,r_left), i in [0,m_phys),
   * j in [0,n_phys), b in [0,r_right)).
   */
  double& operator()(int a, int i, int j, int b)
  {
    return arr_(a, i, j, b);
  }

  /**
   * @brief Read-only element access (a in [0,r_left), i in [0,m_phys),
   * j in [0,n_phys), b in [0,r_right)).
   */
  double operator()(int a, int i, int j, int b) const
  {
    return arr_(a, i, j, b);
  }

  /** @brief Left bond dimension. */
  int r_left() const
  {
    return arr_.get_dim(0);
  }
  /** @brief Row-mode size. */
  int m_phys() const
  {
    return arr_.get_dim(1);
  }
  /** @brief Column-mode size. */
  int n_phys() const
  {
    return arr_.get_dim(2);
  }
  /** @brief Right bond dimension. */
  int r_right() const
  {
    return arr_.get_dim(3);
  }
  /** @brief Total number of stored double-precision elements. */
  std::size_t size() const
  {
    return static_cast<std::size_t>(arr_.get_size());
  }

  /**
   * @brief (r_left, m_phys, n_phys, r_right) packed as a std::array.
   */
  std::array<int, 4> dims() const
  {
    return {
      arr_.get_dim(0), arr_.get_dim(1), arr_.get_dim(2), arr_.get_dim(3)
    };
  }

  /** @brief Mutable access to the flat row-major buffer. */
  double* data()
  {
    return arr_.get_data();
  }
  /** @brief Read-only access to the flat row-major buffer. */
  const double* data() const
  {
    return arr_.get_data();
  }

  /**
   * @brief Underlying narray accessor.  Prefer
   * detail::raw_storage(c) at call sites for explicit intent.
   */
  detail::core_matrix_storage& storage()
  {
    return arr_;
  }
  const detail::core_matrix_storage& storage() const
  {
    return arr_;
  }

  private:
  detail::core_matrix_storage arr_;
};

namespace detail
{

inline detail::core_matrix_storage& raw_storage(tt_matrix_core& c)
{
  return c.storage();
}

inline const detail::core_matrix_storage& raw_storage(const tt_matrix_core& c)
{
  return c.storage();
}

}  // namespace detail

/**
 * @brief Tensor-Train matrix (MPO) -- vector of tt_matrix_core.
 *
 * Each core has shape (r_k, m_k, n_k, r_{k+1}) row-major.
 * Boundary ranks: r_0 = r_d = 1.
 * Reconstruction: row-major in both row and column indices.
 */
class tt_matrix
{
  public:
  /** @brief Default-constructed empty TT-matrix (d = 0). */
  tt_matrix() = default;

  /**
   * @brief Construct a TT-matrix from a vector of tt_matrix_core objects.
   * Validates boundary ranks (r_0 = r_d = 1) and bond consistency.
   * @param cores Ordered list of tt_matrix_core objects.
   */
  explicit tt_matrix(std::vector<tt_matrix_core> cores)
      : cores_(std::move(cores))
  {
    const int d = static_cast<int>(cores_.size());
    if (d > 0)
    {
      if (cores_[0].r_left() != 1)
      {
        std::fprintf(stderr,
                     "tt_matrix: r_0 must be 1 (got %d)\n",
                     cores_[0].r_left());
        std::abort();
      }
      if (cores_[static_cast<std::size_t>(d - 1)].r_right() != 1)
      {
        std::fprintf(stderr,
                     "tt_matrix: r_d must be 1 (got %d)\n",
                     cores_[static_cast<std::size_t>(d - 1)].r_right());
        std::abort();
      }
      for (int k = 0; k < d - 1; ++k)
      {
        if (cores_[static_cast<std::size_t>(k)].r_right()
            != cores_[static_cast<std::size_t>(k + 1)].r_left())
        {
          std::fprintf(stderr,
                       "tt_matrix: rank mismatch at bond %d: "
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
   * @brief Number of modes.  0 for default-constructed TT-matrix.
   */
  int d() const
  {
    return static_cast<int>(cores_.size());
  }

  /**
   * @brief Row-mode sizes m_0, ..., m_{d-1}.
   */
  std::vector<int> row_shape() const
  {
    std::vector<int> s;
    s.reserve(cores_.size());
    for (const auto& c : cores_)
    {
      s.push_back(c.m_phys());
    }
    return s;
  }

  /**
   * @brief Column-mode sizes n_0, ..., n_{d-1}.
   */
  std::vector<int> col_shape() const
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
   * @brief Bond ranks r_0, ..., r_d (length d+1).  r_0 = r_d = 1.
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
   * @brief Maximum bond rank.
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
   * @brief Total row extent = prod(row_shape).
   */
  std::size_t total_rows() const
  {
    std::size_t M = 1;
    for (const auto& c : cores_)
      M *= static_cast<std::size_t>(c.m_phys());
    return M;
  }
  /**
   * @brief Total column extent = prod(col_shape).
   */
  std::size_t total_cols() const
  {
    std::size_t N = 1;
    for (const auto& c : cores_)
      N *= static_cast<std::size_t>(c.n_phys());
    return N;
  }

  /**
   * @brief Reconstruct the full dense matrix as a flat row-major vector.
   */
  std::vector<double> to_dense() const
  {
    using row_matrix = eigen_bridge::row_matrix;
    using map_c      = Eigen::Map<const row_matrix>;
    const int dd     = d();
    if (dd == 0)
      return {1.0};  // d=0 is the 1x1 identity (empty tensor product)
    if (cores_.front().r_left() != 1 || cores_.back().r_right() != 1)
    {
      std::fprintf(stderr,
                   "to_dense: boundary ranks must be 1 "
                   "(r_left=%d, r_right=%d); aborting.\n",
                   cores_.front().r_left(),
                   cores_.back().r_right());
      std::abort();
    }

    // Step 1: accumulate the product (k_0 . . . k_{d-1}) flattening as
    // we go.  c0 has shape (1, m_0, n_0, r_1) -> view as (m_0*n_0, r_1).
    const tt_matrix_core& c0 = cores_.front();
    Eigen::Index rows = static_cast<Eigen::Index>(c0.m_phys()) * c0.n_phys();
    Eigen::Index cols = c0.r_right();
    row_matrix T(rows, cols);
    std::memcpy(T.data(),
                c0.data(),
                sizeof(double) * static_cast<std::size_t>(rows) * cols);
    for (int k = 1; k < dd; ++k)
    {
      const tt_matrix_core& ck = cores_[k];
      const int r_k            = ck.r_left();
      const int m_phys         = ck.m_phys();
      const int n_phys         = ck.n_phys();
      const int r_kp1          = ck.r_right();
      const Eigen::Index mn_phys = static_cast<Eigen::Index>(m_phys) * n_phys;
      // Right-unfold of ck viewed as 3-axis: (r_k, mn_phys * r_kp1).
      const Eigen::Index rk_unfold = mn_phys * r_kp1;
      row_matrix Tn(rows, rk_unfold);
      map_c ck_right(ck.data(), r_k, rk_unfold);
      Tn.noalias()       = T * ck_right;
      const Eigen::Index new_rows = rows * mn_phys;
      const Eigen::Index new_cols = r_kp1;
      T.resize(new_rows, new_cols);
      std::memcpy(
        T.data(),
        Tn.data(),
        sizeof(double) * static_cast<std::size_t>(new_rows) * new_cols);
      rows = new_rows;
      cols = new_cols;
    }
    // After the loop, T has length prod_k (m_k*n_k) and cols == 1.
    // Layout in T (flat): (k_0, k_1, ..., k_{d-1}) row-major, where
    //   k_a = i_a * n_a + j_a.

    // Step 2: permute into (I, J) row-major.
    const std::size_t M_total = total_rows();
    const std::size_t N_total = total_cols();
    std::vector<double> out(M_total * N_total);

    // Row strides in I-space (row-major over m_0..m_{d-1}):
    //   I_stride[k] = prod_{l > k} m_l.
    // Same for J-space and for the input k-space (mn_k product).
    std::vector<std::size_t> mstride(dd), nstride(dd), kstride(dd);
    {
      std::size_t mp = 1, np = 1, kp = 1;
      for (int k = dd - 1; k >= 0; --k)
      {
        mstride[k] = mp;
        nstride[k] = np;
        kstride[k] = kp;
        mp *= static_cast<std::size_t>(cores_[k].m_phys());
        np *= static_cast<std::size_t>(cores_[k].n_phys());
        kp *=
          static_cast<std::size_t>(cores_[k].m_phys()) *
          static_cast<std::size_t>(cores_[k].n_phys());
      }
    }

    // Walk the multi-index (i_0, j_0, i_1, j_1, ..., i_{d-1}, j_{d-1})
    // by counters; compute both flat positions in lock-step.  For
    // small d this is plenty fast and the buffer access is sequential
    // in the output.
    std::vector<int> ii(dd, 0), jj(dd, 0);
    while (true)
    {
      std::size_t flat_in = 0;
      std::size_t flat_I  = 0;
      std::size_t flat_J  = 0;
      for (int k = 0; k < dd; ++k)
      {
        const std::size_t k_idx =
          static_cast<std::size_t>(ii[k]) * cores_[k].n_phys() + jj[k];
        flat_in += k_idx * kstride[k];
        flat_I += static_cast<std::size_t>(ii[k]) * mstride[k];
        flat_J += static_cast<std::size_t>(jj[k]) * nstride[k];
      }
      out[flat_I * N_total + flat_J] = T(static_cast<Eigen::Index>(flat_in), 0);

      // Increment (i_0..,j_0..) lexicographically, fastest-varying =
      // last axis's j, then last axis's i, then second-to-last j, etc.
      int axis = dd - 1;
      while (axis >= 0)
      {
        ++jj[axis];
        if (jj[axis] < cores_[axis].n_phys())
          break;
        jj[axis] = 0;
        ++ii[axis];
        if (ii[axis] < cores_[axis].m_phys())
          break;
        ii[axis] = 0;
        --axis;
      }
      if (axis < 0)
        break;
    }
    return out;
  }

  /**
   * @brief Access core k (read/write).
   * @param k Core index, 0 <= k < d.
   */
  tt_matrix_core& core(int k)
  {
    if (k < 0 || k >= static_cast<int>(cores_.size()))
    {
      std::fprintf(stderr,
                   "tt_matrix::core: index %d out of range [0, %zu)\n",
                   k,
                   cores_.size());
      std::abort();
    }
    return cores_[static_cast<std::size_t>(k)];
  }
  const tt_matrix_core& core(int k) const
  {
    if (k < 0 || k >= static_cast<int>(cores_.size()))
    {
      std::fprintf(stderr,
                   "tt_matrix::core: index %d out of range [0, %zu)\n",
                   k,
                   cores_.size());
      std::abort();
    }
    return cores_[static_cast<std::size_t>(k)];
  }

  /**
   * @brief Mutable access to the core vector.
   * @warning Callers must maintain TT-matrix invariants after mutation.
   */
  std::vector<tt_matrix_core>& cores()
  {
    return cores_;
  }
  const std::vector<tt_matrix_core>& cores() const
  {
    return cores_;
  }

  private:
  std::vector<tt_matrix_core> cores_;
};

}  // namespace tensor_train
}  // namespace mva
//
// :D
//
