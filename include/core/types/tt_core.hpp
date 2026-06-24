/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Single TT core wrapping an narray<double, 0, N0, 0, 3, host_heap, layout_r>
*/
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include "../detail/storage_backend.hpp"

namespace mva
{
namespace tensor_train
{

class tt_core
{
  public:
  /** @brief Default-constructed empty core (r_left = n_phys = r_right = 0). */
  tt_core() = default;

  /**
   * @brief Construct a TT core with given dimensions.
   * Allocates row-major storage for shape (r_left, n_phys, r_right).
   * @param r_left  Left bond dimension (must be positive).
   * @param n_phys  Physical mode size (must be positive).
   * @param r_right Right bond dimension (must be positive).
   */
  tt_core(int r_left, int n_phys, int r_right)
  {
    if (r_left <= 0 || n_phys <= 0 || r_right <= 0)
    {
      std::fprintf(stderr,
                   "tt_core: dimensions must be positive "
                   "(got %d, %d, %d)\n",
                   r_left, n_phys, r_right);
      std::abort();
    }
    {
      const int64_t rl64 = static_cast<int64_t>(r_left);
      const int64_t np64 = static_cast<int64_t>(n_phys);
      const int64_t rr64 = static_cast<int64_t>(r_right);
      const int64_t int_max64 = static_cast<int64_t>(
        std::numeric_limits<int>::max());
      if (rl64 > int_max64 / (np64 * rr64))
      {
        const int64_t total64 = rl64 * np64 * rr64;
        std::fprintf(stderr,
                     "tt_core: dimension product "
                     "%ld * %d * %d = %ld exceeds INT_MAX\n",
                     static_cast<long>(r_left), n_phys, r_right,
                     static_cast<long>(total64));
        std::abort();
      }
    }
    arr_.allocate(r_left, n_phys, r_right);
  }

  // Value-semantic: narray defines copy/move; we just default through.
  tt_core(const tt_core&)            = default;
  tt_core(tt_core&&)                 = default;
  tt_core& operator=(const tt_core&) = default;
  tt_core& operator=(tt_core&&)      = default;

  /**
   * @brief Allocate storage for shape (r_left, n_phys, r_right).
   * Deallocates any previous allocation first.
   */
  void allocate(int r_left, int n_phys, int r_right)
  {
    if (r_left <= 0 || n_phys <= 0 || r_right <= 0)
    {
      std::fprintf(stderr,
                   "tt_core::allocate: dimensions must be positive "
                   "(got %d, %d, %d)\n",
                   r_left, n_phys, r_right);
      std::abort();
    }
    {
      const int64_t rl64 = static_cast<int64_t>(r_left);
      const int64_t np64 = static_cast<int64_t>(n_phys);
      const int64_t rr64 = static_cast<int64_t>(r_right);
      const int64_t int_max64 = static_cast<int64_t>(
        std::numeric_limits<int>::max());
      if (rl64 > int_max64 / (np64 * rr64))
      {
        const int64_t total64 = rl64 * np64 * rr64;
        std::fprintf(stderr,
                     "tt_core::allocate: dimension product "
                     "%ld * %d * %d = %ld exceeds INT_MAX\n",
                     static_cast<long>(r_left), n_phys, r_right,
                     static_cast<long>(total64));
        std::abort();
      }
    }
    if (arr_.get_status() == detail::storage_status::allocated)
    {
      arr_.deallocate();
    }
    arr_.allocate(r_left, n_phys, r_right);
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
                   "tt_core::zero_clear: core is not allocated\n");
      std::abort();
    }
    arr_.zero_clear();
  }

  /**
   * @brief Element access (a in [0,r_left), n in [0,n_phys),
   * b in [0,r_right)).
   */
  double& operator()(int a, int n, int b)
  {
    return arr_(a, n, b);
  }

  double operator()(int a, int n, int b) const
  {
    return arr_(a, n, b);
  }

  /**
   * @brief Left bond dimension.
   */
  int r_left() const
  {
    return arr_.get_dim(0);
  }
  /**
   * @brief Physical mode size.
   */
  int n_phys() const
  {
    return arr_.get_dim(1);
  }
  /**
   * @brief Right bond dimension.
   */
  int r_right() const
  {
    return arr_.get_dim(2);
  }
  /** @brief Total number of stored double-precision elements. */
  std::size_t size() const
  {
    return static_cast<std::size_t>(arr_.get_size());
  }

  /**
   * @brief (r_left, n_phys, r_right) packed as a std::array.
   */
  std::array<int, 3> dims() const
  {
    return { arr_.get_dim(0), arr_.get_dim(1), arr_.get_dim(2) };
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
  detail::core_storage& storage()
  {
    return arr_;
  }
  const detail::core_storage& storage() const
  {
    return arr_;
  }

  private:
  detail::core_storage arr_;
};

namespace detail
{

// Escape hatch for advanced uses (e.g. interfacing with narray-aware
// kernels).  Library code should reach for this rather than the
// member storage() so the intent is explicit at the call site.
inline detail::core_storage& raw_storage(tt_core& c)
{
  return c.storage();
}

inline const detail::core_storage& raw_storage(const tt_core& c)
{
  return c.storage();
}

}  // namespace detail

}  // namespace tensor_train
}  // namespace mva
//
// :D
//
