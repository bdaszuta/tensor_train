/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Warm-start compatibility checks for ALS / DMRG round paths
*/
#pragma once

#include <vector>

#include "../types/tt.hpp"
#include "../types/tt_matrix.hpp"

namespace mva
{
namespace tensor_train
{
namespace detail
{

// True iff ``warm`` is non-null, has the same length as ``ref``, and
// every site shares the physical mode size with ``ref``.  Used by the
// tt-valued warm-start paths (als_round, dmrg_round, matvec_*_round).
inline bool warm_start_tt_compatible(const tt* warm, const tt& ref)
{
  if (warm == nullptr || warm->d() != ref.d())
    return false;
  const int d = ref.d();
  for (int k = 0; k < d; ++k)
  {
    if (warm->core(k).n_phys() != ref.core(k).n_phys())
      return false;
  }
  return true;
}

// True iff ``warm`` is non-null, has length ``ms.size()``, and every
// site matches ``(ms[k], ns[k])``.  Used by the tt_matrix-valued
// warm-start paths (matrix_*_round, matmat_*_round).
inline bool warm_start_matrix_compatible(const tt_matrix* warm,
                                         const std::vector<int>& ms,
                                         const std::vector<int>& ns)
{
  if (warm == nullptr)
    return false;
  const int d = static_cast<int>(ms.size());
  if (static_cast<int>(ns.size()) != d || warm->d() != d)
    return false;
  for (int k = 0; k < d; ++k)
  {
    const tt_matrix_core& wc = warm->core(k);
    if (wc.m_phys() != ms[static_cast<std::size_t>(k)] ||
        wc.n_phys() != ns[static_cast<std::size_t>(k)])
      return false;
  }
  return true;
}

// Deep-copy a tt by copying its core vector.
inline tt copy_tt(const tt& src)
{
  return tt(std::vector<tt_core>(src.cores().begin(), src.cores().end()));
}

}  // namespace detail
}  // namespace tensor_train
}  // namespace mva
//
// :D
//
