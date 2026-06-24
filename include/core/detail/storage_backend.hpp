/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Storage backend abstraction for tt_core / tt_matrix_core
*/
#pragma once

#ifdef TENSOR_TRAIN_USE_NARRAY

// -----------------------------------------------------------------
// NARRAY BACKEND
// -----------------------------------------------------------------

#include "narray.hpp"

namespace mva
{
namespace tensor_train
{
namespace detail
{

namespace mcn_b = mva::containers::narray;

// Re-use the narray status enum via an alias so the core headers can
// write storage_status::allocated uniformly.
using storage_status = mcn_b::status;

// 3-axis core storage.
using core_storage = mcn_b::narray<double,
                                    0,
                                    mcn_b::idx_symmetries::N0,
                                    0,
                                    3,
                                    mcn_b::memory_space::host_heap,
                                    mcn_b::memory_layout::layout_r,
                                    0>;

// 4-axis core storage (TT-matrix).
using core_matrix_storage = mcn_b::narray<double,
                                           0,
                                           mcn_b::idx_symmetries::N0,
                                           0,
                                           4,
                                           mcn_b::memory_space::host_heap,
                                           mcn_b::memory_layout::layout_r,
                                           0>;

}  // namespace detail
}  // namespace tensor_train
}  // namespace mva

#else  // !TENSOR_TRAIN_USE_NARRAY

// -----------------------------------------------------------------
// STANDALONE BACKEND
// -----------------------------------------------------------------

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>

namespace mva
{
namespace tensor_train
{
namespace detail
{

enum class storage_status
{
  unallocated = 0,
  allocated   = 1
};

// -----------------------------------------------------------------
// core_storage: 3-axis (r_left, n_phys, r_right), row-major.
//
// Memory order: linear = (a * n_phys + n) * r_right + b.
// This is bit-identical to the narray<..., 3, layout_r> layout.
// -----------------------------------------------------------------

class core_storage
{
  public:
  core_storage() = default;

  core_storage(const core_storage& other)
    : status_(other.status_)
  {
    if (status_ == storage_status::allocated)
    {
      dims_[0] = other.dims_[0];
      dims_[1] = other.dims_[1];
      dims_[2] = other.dims_[2];
      const std::size_t total = static_cast<std::size_t>(dims_[0])
                                * static_cast<std::size_t>(dims_[1])
                                * static_cast<std::size_t>(dims_[2]);
      data_ = new double[total];
      std::memcpy(data_, other.data_, total * sizeof(double));
    }
  }

  core_storage(core_storage&& other) noexcept
    : data_(other.data_), status_(other.status_)
  {
    dims_[0]       = other.dims_[0];
    dims_[1]       = other.dims_[1];
    dims_[2]       = other.dims_[2];
    other.data_    = nullptr;
    other.status_  = storage_status::unallocated;
    other.dims_[0] = 0;
    other.dims_[1] = 0;
    other.dims_[2] = 0;
  }

  core_storage& operator=(const core_storage& other)
  {
    if (this == &other)
      return *this;
    deallocate();
    status_ = other.status_;
    if (status_ == storage_status::allocated)
    {
      dims_[0] = other.dims_[0];
      dims_[1] = other.dims_[1];
      dims_[2] = other.dims_[2];
      const std::size_t total = static_cast<std::size_t>(dims_[0])
                                * static_cast<std::size_t>(dims_[1])
                                * static_cast<std::size_t>(dims_[2]);
      data_ = new double[total];
      std::memcpy(data_, other.data_, total * sizeof(double));
    }
    return *this;
  }

  core_storage& operator=(core_storage&& other) noexcept
  {
    if (this == &other)
      return *this;
    deallocate();
    data_            = other.data_;
    status_          = other.status_;
    dims_[0]         = other.dims_[0];
    dims_[1]         = other.dims_[1];
    dims_[2]         = other.dims_[2];
    other.data_      = nullptr;
    other.status_    = storage_status::unallocated;
    other.dims_[0]   = 0;
    other.dims_[1]   = 0;
    other.dims_[2]   = 0;
    return *this;
  }

  ~core_storage()
  {
    delete[] data_;
    data_   = nullptr;
    status_ = storage_status::unallocated;
  }

  void allocate(int d0, int d1, int d2)
  {
    {
      const int64_t total64 = static_cast<int64_t>(d0) * d1 * d2;
      if (total64 > static_cast<int64_t>(std::numeric_limits<int>::max()))
      {
        std::fprintf(stderr,
                     "core_storage::allocate: dimension product "
                     "%ld * %d * %d = %ld exceeds INT_MAX\n",
                     static_cast<long>(d0), d1, d2,
                     static_cast<long>(total64));
        std::abort();
      }
    }
    deallocate();
    dims_[0] = d0;
    dims_[1] = d1;
    dims_[2] = d2;
    const std::size_t total = static_cast<std::size_t>(d0)
                              * static_cast<std::size_t>(d1)
                              * static_cast<std::size_t>(d2);
    data_   = new double[total]();
    status_ = storage_status::allocated;
  }

  void deallocate()
  {
    delete[] data_;
    data_    = nullptr;
    status_  = storage_status::unallocated;
    dims_[0] = 0;
    dims_[1] = 0;
    dims_[2] = 0;
  }

  storage_status get_status() const
  {
    return status_;
  }

  void zero_clear()
  {
    const std::size_t total = static_cast<std::size_t>(dims_[0])
                              * static_cast<std::size_t>(dims_[1])
                              * static_cast<std::size_t>(dims_[2]);
    std::memset(data_, 0, total * sizeof(double));
  }

  int get_dim(int axis) const
  {
    return dims_[axis];
  }

  int get_size() const
  {
    return dims_[0] * dims_[1] * dims_[2];
  }

  double* get_data()
  {
    return data_;
  }

  const double* get_data() const
  {
    return data_;
  }

  // Element access: (a, n, b).
  double& operator()(int a, int n, int b)
  {
    const std::size_t idx =
      (static_cast<std::size_t>(a) * dims_[1]
       + static_cast<std::size_t>(n)) * dims_[2]
      + static_cast<std::size_t>(b);
    return data_[idx];
  }

  double operator()(int a, int n, int b) const
  {
    const std::size_t idx =
      (static_cast<std::size_t>(a) * dims_[1]
       + static_cast<std::size_t>(n)) * dims_[2]
      + static_cast<std::size_t>(b);
    return data_[idx];
  }

  private:
  double* data_         = nullptr;
  int     dims_[3]      = {0, 0, 0};
  storage_status status_ = storage_status::unallocated;
};


// -----------------------------------------------------------------
// core_matrix_storage: 4-axis (r_left, m_phys, n_phys, r_right),
// row-major.
//
// Memory order: linear = ((a * m + i) * n + j) * r_right + b.
// -----------------------------------------------------------------

class core_matrix_storage
{
  public:
  core_matrix_storage() = default;

  core_matrix_storage(const core_matrix_storage& other)
    : status_(other.status_)
  {
    if (status_ == storage_status::allocated)
    {
      dims_[0] = other.dims_[0];
      dims_[1] = other.dims_[1];
      dims_[2] = other.dims_[2];
      dims_[3] = other.dims_[3];
      const std::size_t total = static_cast<std::size_t>(dims_[0])
                                * static_cast<std::size_t>(dims_[1])
                                * static_cast<std::size_t>(dims_[2])
                                * static_cast<std::size_t>(dims_[3]);
      data_ = new double[total];
      std::memcpy(data_, other.data_, total * sizeof(double));
    }
  }

  core_matrix_storage(core_matrix_storage&& other) noexcept
    : data_(other.data_), status_(other.status_)
  {
    dims_[0]       = other.dims_[0];
    dims_[1]       = other.dims_[1];
    dims_[2]       = other.dims_[2];
    dims_[3]       = other.dims_[3];
    other.data_    = nullptr;
    other.status_  = storage_status::unallocated;
    other.dims_[0] = 0;
    other.dims_[1] = 0;
    other.dims_[2] = 0;
    other.dims_[3] = 0;
  }

  core_matrix_storage& operator=(const core_matrix_storage& other)
  {
    if (this == &other)
      return *this;
    deallocate();
    status_ = other.status_;
    if (status_ == storage_status::allocated)
    {
      dims_[0] = other.dims_[0];
      dims_[1] = other.dims_[1];
      dims_[2] = other.dims_[2];
      dims_[3] = other.dims_[3];
      const std::size_t total = static_cast<std::size_t>(dims_[0])
                                * static_cast<std::size_t>(dims_[1])
                                * static_cast<std::size_t>(dims_[2])
                                * static_cast<std::size_t>(dims_[3]);
      data_ = new double[total];
      std::memcpy(data_, other.data_, total * sizeof(double));
    }
    return *this;
  }

  core_matrix_storage& operator=(core_matrix_storage&& other) noexcept
  {
    if (this == &other)
      return *this;
    deallocate();
    data_            = other.data_;
    status_          = other.status_;
    dims_[0]         = other.dims_[0];
    dims_[1]         = other.dims_[1];
    dims_[2]         = other.dims_[2];
    dims_[3]         = other.dims_[3];
    other.data_      = nullptr;
    other.status_    = storage_status::unallocated;
    other.dims_[0]   = 0;
    other.dims_[1]   = 0;
    other.dims_[2]   = 0;
    other.dims_[3]   = 0;
    return *this;
  }

  ~core_matrix_storage()
  {
    delete[] data_;
    data_   = nullptr;
    status_ = storage_status::unallocated;
  }

  void allocate(int d0, int d1, int d2, int d3)
  {
    {
      const int64_t total64 = static_cast<int64_t>(d0) * d1 * d2 * d3;
      if (total64 > static_cast<int64_t>(std::numeric_limits<int>::max()))
      {
        std::fprintf(stderr,
                     "core_matrix_storage::allocate: dimension product "
                     "%ld * %d * %d * %d = %ld exceeds INT_MAX\n",
                     static_cast<long>(d0), d1, d2, d3,
                     static_cast<long>(total64));
        std::abort();
      }
    }
    deallocate();
    dims_[0] = d0;
    dims_[1] = d1;
    dims_[2] = d2;
    dims_[3] = d3;
    const std::size_t total = static_cast<std::size_t>(d0)
                              * static_cast<std::size_t>(d1)
                              * static_cast<std::size_t>(d2)
                              * static_cast<std::size_t>(d3);
    data_   = new double[total]();
    status_ = storage_status::allocated;
  }

  void deallocate()
  {
    delete[] data_;
    data_    = nullptr;
    status_  = storage_status::unallocated;
    dims_[0] = 0;
    dims_[1] = 0;
    dims_[2] = 0;
    dims_[3] = 0;
  }

  storage_status get_status() const
  {
    return status_;
  }

  void zero_clear()
  {
    const std::size_t total = static_cast<std::size_t>(dims_[0])
                              * static_cast<std::size_t>(dims_[1])
                              * static_cast<std::size_t>(dims_[2])
                              * static_cast<std::size_t>(dims_[3]);
    std::memset(data_, 0, total * sizeof(double));
  }

  int get_dim(int axis) const
  {
    return dims_[axis];
  }

  int get_size() const
  {
    return dims_[0] * dims_[1] * dims_[2] * dims_[3];
  }

  double* get_data()
  {
    return data_;
  }

  const double* get_data() const
  {
    return data_;
  }

  // Element access: (a, i, j, b).
  double& operator()(int a, int i, int j, int b)
  {
    const std::size_t idx =
      ((static_cast<std::size_t>(a) * dims_[1]
        + static_cast<std::size_t>(i)) * dims_[2]
        + static_cast<std::size_t>(j)) * dims_[3]
      + static_cast<std::size_t>(b);
    return data_[idx];
  }

  double operator()(int a, int i, int j, int b) const
  {
    const std::size_t idx =
      ((static_cast<std::size_t>(a) * dims_[1]
        + static_cast<std::size_t>(i)) * dims_[2]
        + static_cast<std::size_t>(j)) * dims_[3]
      + static_cast<std::size_t>(b);
    return data_[idx];
  }

  private:
  double* data_         = nullptr;
  int     dims_[4]      = {0, 0, 0, 0};
  storage_status status_ = storage_status::unallocated;
};

}  // namespace detail
}  // namespace tensor_train
}  // namespace mva

#endif  // TENSOR_TRAIN_USE_NARRAY
//
// :D
//
