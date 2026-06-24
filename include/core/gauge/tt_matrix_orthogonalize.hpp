/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Right-to-left QR sweep for a TT-matrix (MPO)
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
#include "tt_orthogonalize.hpp"

namespace mva
{
namespace tensor_train
{
/**
 * @brief Right-orthogonalize a TT-matrix in-place.
 * Packs each 4-axis core to 3-axis, applies the TT right-orthogonalize,
 * then unpacks.  cores[1..d-1] become right-orthogonal.
 * @param a TT-matrix to orthogonalize (modified in-place).
 */

inline void right_orthogonalize(tt_matrix& a)
{
  const int d = a.d();
  if (d <= 1)
    return;

  // Pack into a tt with mode size m_k * n_k.
  std::vector<tt_core> packed;
  packed.reserve(static_cast<std::size_t>(d));
  std::vector<int> ms(static_cast<std::size_t>(d));
  std::vector<int> ns(static_cast<std::size_t>(d));
  for (int k = 0; k < d; ++k)
  {
    const tt_matrix_core& mc = a.core(k);
    ms[k]                    = mc.m_phys();
    ns[k]                    = mc.n_phys();
    const int64_t mn = static_cast<int64_t>(mc.m_phys()) * mc.n_phys();
    if (mn > std::numeric_limits<int>::max())
    {
      std::fprintf(stderr,
                   "right_orthogonalize(tt_matrix): mode product %ld "
                   "exceeds INT_MAX; aborting.\n",
                   static_cast<long>(mn));
      std::abort();
    }
    tt_core c(mc.r_left(), static_cast<int>(mn), mc.r_right());
    std::memcpy(c.data(),
                mc.data(),
                sizeof(double) * static_cast<std::size_t>(c.size()));
    packed.push_back(std::move(c));
  }
  tt v(std::move(packed));

  // Reuse the 3-axis right-orthogonalisation in place.
  right_orthogonalize(v);

  // Unpack back into a's cores.  Ranks may have shrunk (QR on the
  // transpose truncates to r_eff = min(r, n*r_right) via
  // HouseholderQR), so we replace each matrix-core.
  std::vector<tt_matrix_core> out;
  out.reserve(static_cast<std::size_t>(d));
  for (int k = 0; k < d; ++k)
  {
    const tt_core& c = v.core(k);
    tt_matrix_core mc(c.r_left(), ms[k], ns[k], c.r_right());
    std::memcpy(mc.data(),
                c.data(),
                sizeof(double) * static_cast<std::size_t>(c.size()));
    out.push_back(std::move(mc));
  }
  a = tt_matrix(std::move(out));
}

}  // namespace tensor_train
}  // namespace mva
//
// :D
//
