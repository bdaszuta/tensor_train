/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Single home for tt <-> tt_matrix repackaging
*/
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>
#include <limits>

#include "../types/tt.hpp"
#include "../types/tt_core.hpp"
#include "../types/tt_matrix.hpp"

namespace mva
{
namespace tensor_train
{
namespace detail
{

// Pure memcpy: tt_matrix_core -> tt_core vector.  No allocations beyond
// the output vector (length d).
inline std::vector<tt_core> pack_matrix_cores(const tt_matrix& a)
{
  const int d = a.d();
  std::vector<tt_core> out;
  out.reserve(static_cast<std::size_t>(d));
  for (int k = 0; k < d; ++k)
  {
    const tt_matrix_core& mc = a.core(k);
    const int64_t mn = static_cast<int64_t>(mc.m_phys()) * mc.n_phys();
    if (mn > std::numeric_limits<int>::max())
    {
      std::fprintf(stderr,
                   "pack_matrix_cores: mode product %ld exceeds INT_MAX; "
                   "aborting.\n",
                   static_cast<long>(mn));
      std::abort();
    }
    tt_core c(mc.r_left(), static_cast<int>(mn), mc.r_right());
    std::memcpy(c.data(),
                mc.data(),
                sizeof(double) * static_cast<std::size_t>(c.size()));
    out.push_back(std::move(c));
  }
  return out;
}

// Pack into a tt with mode size m_k * n_k AND record (m, n) per site.
inline tt pack_matrix_as_tt(const tt_matrix& a,
                            std::vector<int>& ms,
                            std::vector<int>& ns)
{
  const int d = a.d();
  ms.assign(static_cast<std::size_t>(d), 0);
  ns.assign(static_cast<std::size_t>(d), 0);
  for (int k = 0; k < d; ++k)
  {
    const tt_matrix_core& mc        = a.core(k);
    ms[static_cast<std::size_t>(k)] = mc.m_phys();
    ns[static_cast<std::size_t>(k)] = mc.n_phys();
  }
  return tt(pack_matrix_cores(a));
}

// Inverse of pack_matrix_as_tt.
inline tt_matrix unpack_tt_as_matrix(const tt& packed,
                                     const std::vector<int>& ms,
                                     const std::vector<int>& ns)
{
  const int d = packed.d();
  std::vector<tt_matrix_core> out;
  out.reserve(static_cast<std::size_t>(d));
  for (int k = 0; k < d; ++k)
  {
    const tt_core& c = packed.core(k);
    tt_matrix_core mc(c.r_left(),
                      ms[static_cast<std::size_t>(k)],
                      ns[static_cast<std::size_t>(k)],
                      c.r_right());
    std::memcpy(mc.data(),
                c.data(),
                sizeof(double) * static_cast<std::size_t>(c.size()));
    out.push_back(std::move(mc));
  }
  return tt_matrix(std::move(out));
}

}  // namespace detail
}  // namespace tensor_train
}  // namespace mva
//
// :D
//
