/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Zero-copy Eigen::Map views over a tt_core's underlying buffer
*/
#pragma once

#include <Eigen/Core>

#include "../types/tt_core.hpp"
#include "../types/tt_eigen_bridge.hpp"

namespace mva
{
namespace tensor_train
{
namespace detail
{

using row_matrix       = eigen_bridge::row_matrix;
using map_row_matrix   = Eigen::Map<row_matrix>;
using map_c_row_matrix = Eigen::Map<const row_matrix>;

// (r_left * n_phys, r_right) view, mutable.
inline map_row_matrix left_unfold(tt_core& c)
{
  return map_row_matrix(c.data(),
                        static_cast<Eigen::Index>(c.r_left()) * c.n_phys(),
                        c.r_right());
}

// (r_left * n_phys, r_right) view, const.
inline map_c_row_matrix left_unfold(const tt_core& c)
{
  return map_c_row_matrix(c.data(),
                          static_cast<Eigen::Index>(c.r_left()) * c.n_phys(),
                          c.r_right());
}

// (r_left, n_phys * r_right) view, mutable.
inline map_row_matrix right_unfold(tt_core& c)
{
  return map_row_matrix(c.data(),
                        c.r_left(),
                        static_cast<Eigen::Index>(c.n_phys()) * c.r_right());
}

// (r_left, n_phys * r_right) view, const.
inline map_c_row_matrix right_unfold(const tt_core& c)
{
  return map_c_row_matrix(c.data(),
                          c.r_left(),
                          static_cast<Eigen::Index>(c.n_phys()) * c.r_right());
}

}  // namespace detail
}  // namespace tensor_train
}  // namespace mva
//
// :D
//
