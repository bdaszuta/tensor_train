/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Function-sampling TT construction API
*/
#pragma once

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>
#include <limits>

#include "../algebra/tt_eval.hpp"
#include "../types/tt.hpp"
#include "../types/tt_eigen_bridge.hpp"
#include "../types/tt_matrix.hpp"
#include "from_dense.hpp"
#include "tt_als_cross.hpp"
#include "tt_amen_cross.hpp"
#include "tt_cross.hpp"
#include "tt_dmrg_cross.hpp"

namespace mva
{
namespace tensor_train
{

enum class from_samples_method
{
  dmrg_cross,
  tt_cross,
  als_cross,
  amen_cross,
  qtt_dmrg_cross,
  qtt_tt_cross,
  qtt_als_cross,
  qtt_amen_cross
};

struct from_samples_options
{
  double eps                  = 1.0e-10;
  int max_rank                = 0;
  int rmin                    = 1;
  from_samples_method method  = from_samples_method::dmrg_cross;
  int max_sweeps              = 10;
  int init_rank               = 2;
  std::uint64_t seed          = 0;
  int maxvol_resolve_interval = 16;
  int enrich_rank             = 4;
  bool use_qr_pivots          = false;

  int qtt_base                 = 2;
  std::vector<int> qtt_maxbits = {};
  int qtt_pool_modes           = 3;

  // Residual enrichment
  int enrich_rounds  = 0;
  int enrich_samples = 1024;
  int enrich_k       = 32;

  std::vector<int> qtt_row_maxbits = {};
  std::vector<int> qtt_col_maxbits = {};
};

namespace detail
{

inline std::vector<int> qtt_shape(int base, const std::vector<int>& maxbits)
{
  int nbits = 0;
  for (int b : maxbits)
    nbits += b;
  return std::vector<int>(static_cast<std::size_t>(nbits), base);
}

inline std::vector<int> pooled_meta_shape(int base,
                                          const std::vector<int>& maxbits,
                                          int pool)
{
  int total = 0;
  for (int b : maxbits)
    total += b;
  if (pool < 2 || total == 0)
    return {};
  std::vector<int> meta;
  int remaining = total;
  while (remaining > 0)
  {
    const int w = std::min(pool, remaining);
    int sz      = 1;
    for (int i = 0; i < w; ++i)
      sz *= base;
    meta.push_back(sz);
    remaining -= w;
  }
  return meta;
}

// QTT dispatcher with pooled meta-cross.
template <typename F>
inline tt qtt_wrap(F&& func,
                   const std::vector<int>& maxbits,
                   int base,
                   const from_samples_options& opts,
                   const std::vector<std::vector<int>>* seed_pivots = nullptr)
{
  const int n_dims               = static_cast<int>(maxbits.size());
  const int pool                 = opts.qtt_pool_modes;
  const std::vector<int> ttshape = qtt_shape(base, maxbits);
  const std::vector<int> meta    = pooled_meta_shape(base, maxbits, pool);

  if (meta.empty())
    return tt();

  // Validate that base is a power of 2 (required for the bitmask
  // extraction in the grid-index decoder below).
  if (base < 2 || (base & (base - 1)) != 0)
  {
    std::fprintf(stderr,
                 "qtt_wrap: base=%d must be a power of 2, "
                 ">= 2; aborting.\n", base);
    std::abort();
  }

  auto grid_fn = [&fn = func, maxbits, base, n_dims](
                   const int* gi) -> double { return fn(gi); };

  std::vector<int> group_bits;
  {
    int rem = 0;
    for (int b : maxbits)
      rem += b;
    while (rem > 0)
    {
      int w = std::min(pool, rem);
      group_bits.push_back(w);
      rem -= w;
    }
  }
  // Cumulative group boundaries.
  std::vector<int> cum(static_cast<std::size_t>(group_bits.size()) + 1, 0);
  for (std::size_t i = 0; i < group_bits.size(); ++i)
    cum[i + 1] = cum[i] + group_bits[i];

  auto inner =
    [fn2 = std::move(grid_fn), maxbits, base, group_bits, cum, n_dims](
      const int* meta_idx) -> double
  {
    std::vector<int> grid_idx(static_cast<std::size_t>(n_dims));
    int gb = 0;
    for (int dim = 0; dim < n_dims; ++dim)
    {
      int64_t val = 0;
      for (int b = 0; b < maxbits[dim]; ++b, ++gb)
      {
        int g = 0;
        while (gb >= cum[static_cast<std::size_t>(g + 1)])
          ++g;
        const int pos_in_g  = gb - cum[static_cast<std::size_t>(g)];
        const int bits_in_g = group_bits[static_cast<std::size_t>(g)];
        const int mp        = meta_idx[g];
        const int shift     = bits_in_g - 1 - pos_in_g;
        val                 = val * base + ((mp >> shift) & (base - 1));
      }
      if (val > std::numeric_limits<int>::max())
      {
        std::fprintf(stderr,
                     "qtt_wrap: grid index %ld exceeds INT_MAX\n",
                     static_cast<long>(val));
        std::abort();
      }
      grid_idx[static_cast<std::size_t>(dim)] = static_cast<int>(val);
    }
    return fn2(grid_idx.data());
  };

  tt meta_tt;
  if (opts.method == from_samples_method::qtt_dmrg_cross)
  {
    dmrg_cross_options co;
    co.eps                     = opts.eps;
    co.max_rank                = opts.max_rank;
    co.max_sweeps              = opts.max_sweeps;
    co.init_rank               = opts.init_rank;
    co.seed                    = opts.seed;
    co.maxvol_resolve_interval = opts.maxvol_resolve_interval;
    co.rmin                    = opts.rmin;
    co.enrich_rank             = opts.enrich_rank;
    co.use_qr_pivots           = opts.use_qr_pivots;
    co.seed_pivots             = seed_pivots;
    meta_tt                    = dmrg_cross(std::move(inner), meta, co);
  }
  else if (opts.method == from_samples_method::qtt_als_cross)
  {
    als_cross_options co;
    co.eps                     = opts.eps;
    co.max_rank                = opts.max_rank;
    co.rmin                    = opts.rmin;
    co.max_sweeps              = opts.max_sweeps;
    co.init_rank               = opts.init_rank;
    co.seed                    = opts.seed;
    co.maxvol_resolve_interval = opts.maxvol_resolve_interval;
    co.enrich_rank             = opts.enrich_rank;
    co.use_qr_pivots           = opts.use_qr_pivots;
    co.seed_pivots             = seed_pivots;
    meta_tt                    = als_cross(std::move(inner), meta, co);
  }
  else if (opts.method == from_samples_method::qtt_amen_cross)
  {
    amen_cross_options co;
    co.eps                     = opts.eps;
    co.max_rank                = opts.max_rank;
    co.rmin                    = opts.rmin;
    co.max_sweeps              = opts.max_sweeps;
    co.init_rank               = opts.init_rank;
    co.seed                    = opts.seed;
    co.maxvol_resolve_interval = opts.maxvol_resolve_interval;
    co.enrich_rank             = opts.enrich_rank;
    co.use_qr_pivots           = opts.use_qr_pivots;
    co.seed_pivots             = seed_pivots;
    meta_tt                    = amen_cross(std::move(inner), meta, co);
  }
  else
  {
    tt_cross_options co;
    co.max_rank                = opts.max_rank;
    co.init_rank               = opts.init_rank;
    co.seed                    = opts.seed;
    co.maxvol_resolve_interval = opts.maxvol_resolve_interval;
    co.seed_pivots             = seed_pivots;
    meta_tt                    = tt_cross(std::move(inner), meta, co);
  }

  // Convert the meta-TT to a dense buffer at the original binary
  // resolution and re-compress via from_dense.  The dense
  // intermediate is base^{total_bits} entries -- small for typical
  // QTT (e.g. 2^{18} = 262K for 6x6x6 bits).  The meta-TT itself
  // was built from the cross algorithm which samples adaptively.
  const int64_t total = [&]
  {
    int64_t p = 1;
    for (int s : ttshape)
      p *= static_cast<int64_t>(s);
    return p;
  }();
  std::vector<double> dense(static_cast<std::size_t>(total));
  std::vector<int> bidx(static_cast<std::size_t>(ttshape.size()));
  const int d_tt   = static_cast<int>(ttshape.size());
  const int d_meta = static_cast<int>(meta.size());

  for (int64_t pos = 0; pos < total; ++pos)
  {
    int64_t tmp = pos;
    for (int i = d_tt - 1; i >= 0; --i)
    {
      const int sz                      = ttshape[static_cast<std::size_t>(i)];
      bidx[static_cast<std::size_t>(i)] = tmp % sz;
      tmp /= sz;
    }
    std::vector<int> mi(static_cast<std::size_t>(d_meta), 0);
    for (int gb = 0; gb < d_tt; ++gb)
    {
      int g = 0;
      while (gb >= cum[static_cast<std::size_t>(g + 1)])
        ++g;
      mi[static_cast<std::size_t>(g)] =
        mi[static_cast<std::size_t>(g)] * base +
        bidx[static_cast<std::size_t>(gb)];
    }
    dense[static_cast<std::size_t>(pos)] = eval_at(meta_tt, mi.data());
  }

  from_dense_options fo;
  fo.eps      = opts.eps;
  fo.max_rank = opts.max_rank;
  fo.method   = from_dense_method::svd;
  return from_dense(dense.data(), ttshape, fo);
}

template <typename F>
inline tt_matrix qtt_matrix_wrap(F&& func,
                                 const std::vector<int>& row_maxbits,
                                 const std::vector<int>& col_maxbits,
                                 int base,
                                 const from_samples_options& opts)
{
  (void)func;
  (void)row_maxbits;
  (void)col_maxbits;
  (void)base;
  (void)opts;
  std::fprintf(stderr,
               "from_samples(tt_matrix): QTT matrix cross not "
               "yet implemented.  Aborting.\n");
  std::abort();
}

// Function adapter: wraps a two-argument func(row_idx, col_idx) -> double
// into a single-argument func(packed_idx) -> double where
// packed_idx[k] = i_k * n_k + j_k.
template <typename F>
inline auto pack_matrix_func(F&& func,
                             const std::vector<int>& row_shape,
                             const std::vector<int>& col_shape)
{
  const int d = static_cast<int>(row_shape.size());
  return
    [fn      = std::forward<F>(func),
     n_cols  = col_shape,
     d,
     row_buf = std::vector<int>(static_cast<std::size_t>(d)),
     col_buf = std::vector<int>(static_cast<std::size_t>(d))]
    (const int* packed_idx) mutable -> double
  {
    for (int k = 0; k < d; ++k)
    {
      const int nk   = n_cols[k];
      row_buf[k] = packed_idx[k] / nk;
      col_buf[k] = packed_idx[k] % nk;
    }
    return fn(row_buf.data(), col_buf.data());
  };
}

}  // namespace detail

// -------- tt (vector / tensor) --------

template <typename F>
/**
 * @brief Build a TT via cross-interpolation.
 *
 * Evaluates the callable func at adaptively chosen multi-indices
 * without ever materialising the full dense tensor.  Supports
 * dmrg_cross, tt_cross, als_cross, amen_cross, and their QTT variants.
 *
 * @param func  Callable with signature double(const int*).
 * @param shape Mode sizes n_0, ..., n_{d-1}.
 * @param opts  from_samples_options controlling method, max rank,
 *              enrichment, and QTT parameters.
 * @return TT approximant.
 * @note  The func is evaluated on-the-fly; no dense storage needed.
 *        QTT methods require qtt_base, qtt_maxbits in opts.
 */
inline tt from_samples(F&& func,
                       const std::vector<int>& shape,
                       const from_samples_options& opts = {})
{
  // Validate shape entries are positive (prevents UB in cross engines
  // which construct uniform_int_distribution with nm-1).
  for (std::size_t k = 0; k < shape.size(); ++k)
  {
    if (shape[k] <= 0)
    {
      std::fprintf(stderr,
                   "from_samples: shape[%zu]=%d must be positive\n",
                   k, shape[k]);
      std::abort();
    }
  }

  const bool is_qtt = (opts.method == from_samples_method::qtt_dmrg_cross ||
                       opts.method == from_samples_method::qtt_tt_cross ||
                       opts.method == from_samples_method::qtt_als_cross ||
                       opts.method == from_samples_method::qtt_amen_cross);

  // Helper: run one cross pass with given parameters + seed pivots.
  auto run_pass = [&](const from_samples_options& o,
                      const std::vector<std::vector<int>>* sp) -> tt
  {
    if (is_qtt)
    {
      return detail::qtt_wrap(func, o.qtt_maxbits, o.qtt_base, o, sp);
    }
    if (o.method == from_samples_method::dmrg_cross)
    {
      dmrg_cross_options co;
      co.eps                     = o.eps;
      co.max_rank                = o.max_rank;
      co.rmin                    = o.rmin;
      co.max_sweeps              = o.max_sweeps;
      co.init_rank               = o.init_rank;
      co.seed                    = o.seed;
      co.maxvol_resolve_interval = o.maxvol_resolve_interval;
      co.enrich_rank             = o.enrich_rank;
      co.use_qr_pivots           = o.use_qr_pivots;
      co.seed_pivots             = sp;
      return dmrg_cross(func, shape, co);
    }
    if (o.method == from_samples_method::als_cross)
    {
      als_cross_options co;
      co.eps                     = o.eps;
      co.max_rank                = o.max_rank;
      co.rmin                    = o.rmin;
      co.max_sweeps              = o.max_sweeps;
      co.init_rank               = o.init_rank;
      co.seed                    = o.seed;
      co.maxvol_resolve_interval = o.maxvol_resolve_interval;
      co.enrich_rank             = o.enrich_rank;
      co.use_qr_pivots           = o.use_qr_pivots;
      co.seed_pivots             = sp;
      return als_cross(func, shape, co);
    }
    if (o.method == from_samples_method::amen_cross)
    {
      amen_cross_options co;
      co.eps                     = o.eps;
      co.max_rank                = o.max_rank;
      co.rmin                    = o.rmin;
      co.max_sweeps              = o.max_sweeps;
      co.init_rank               = o.init_rank;
      co.seed                    = o.seed;
      co.maxvol_resolve_interval = o.maxvol_resolve_interval;
      co.enrich_rank             = o.enrich_rank;
      co.use_qr_pivots           = o.use_qr_pivots;
      co.seed_pivots             = sp;
      return amen_cross(func, shape, co);
    }
    tt_cross_options co;
    co.max_rank                = o.max_rank;
    co.init_rank               = o.init_rank;
    co.seed                    = o.seed;
    co.maxvol_resolve_interval = o.maxvol_resolve_interval;
    co.seed_pivots             = sp;
    return tt_cross(func, shape, co);
  };

  tt result = run_pass(opts, nullptr);

  // Residual enrichment: sample random points, find worst offenders,
  // inject their pivot indices as seed_pivots, re-run cross.
  if (opts.enrich_rounds <= 0 || opts.enrich_samples <= 0 ||
      opts.enrich_k <= 0)
  {
    return result;
  }

  const int d = result.d();
  std::vector<int> shp;
  if (is_qtt)
    shp = detail::qtt_shape(opts.qtt_base, opts.qtt_maxbits);
  else
    shp = shape;

  if (shp.empty())
    return result;

  int64_t total_modes = 1;
  for (int s : shp)
    total_modes *= s;

  // QTT path: result TT lives in binary-index space; func expects
  // original grid-scale indices.  Build a one-shot converter.
  const int n_dims = static_cast<int>(opts.qtt_maxbits.size());
  std::vector<int> grid_idx(static_cast<std::size_t>(n_dims));

  std::mt19937_64 rng(opts.seed + 0xE007ULL);
  std::uniform_int_distribution<int64_t> dist(0, total_modes - 1);
  std::vector<int> idx(static_cast<std::size_t>(d));

  // Buffer: (err, linear_pos) pairs for tracking worst offenders.
  std::vector<std::pair<double, int64_t>> worst(
    static_cast<std::size_t>(opts.enrich_k));

  for (int round = 0; round < opts.enrich_rounds; ++round)
  {
    // Reset worst-offender buffer (sorted ascending by error at end).
    worst.assign(worst.size(), { 0.0, 0 });

    // Sample and find worst offenders using sorted-list tracking.
    // Maintain worst[0..stored-1] sorted ascending; always keep the
    // top enrich_k by error.
    int stored = 0;
    for (int s = 0; s < opts.enrich_samples; ++s)
    {
      int64_t pos = dist(rng);
      int64_t tmp = pos;
      for (int i = d - 1; i >= 0; --i)
      {
        const int sz                     = shp[static_cast<std::size_t>(i)];
        idx[static_cast<std::size_t>(i)] = tmp % sz;
        tmp /= sz;
      }
      double func_val;
      if (is_qtt)
      {
        // Convert binary TT index to grid-scale index, then call func.
        // Binary indices are packed dimension-first: bits for dim 0,
        // then dim 1, ..., with MSB first within each dimension.
        int bit_pos = 0;
        for (int dim = 0; dim < n_dims; ++dim)
        {
          int64_t val = 0;
          for (int b = 0; b < opts.qtt_maxbits[dim]; ++b, ++bit_pos)
            val = val * opts.qtt_base + idx[static_cast<std::size_t>(bit_pos)];
          if (val > std::numeric_limits<int>::max())
          {
            std::fprintf(stderr,
                         "from_samples enrichment: grid index %ld "
                         "exceeds INT_MAX\n",
                         static_cast<long>(val));
            std::abort();
          }
          grid_idx[static_cast<std::size_t>(dim)] = static_cast<int>(val);
        }
        func_val = func(grid_idx.data());
      }
      else
      {
        func_val = func(idx.data());
      }
      double err = std::abs(func_val - eval_at(result, idx.data()));

      if (stored < opts.enrich_k)
      {
        worst[static_cast<std::size_t>(stored)] = { err, pos };
        ++stored;
        if (stored == opts.enrich_k)
          std::sort(worst.begin(), worst.end());
      }
      else if (err > worst[0].first)
      {
        worst[0] = { err, pos };
        // Bubble the new element to its correct sorted position.
        for (int j = 1; j < opts.enrich_k &&
                        worst[static_cast<std::size_t>(j - 1)].first >
                          worst[static_cast<std::size_t>(j)].first;
             ++j)
          std::swap(worst[static_cast<std::size_t>(j - 1)],
                    worst[static_cast<std::size_t>(j)]);
      }
    }

    // Check if error is acceptable.
    double max_err = 0.0;
    for (auto& w : worst)
      if (w.first > max_err)
        max_err = w.first;
    // max_err is absolute pointwise error; opts.eps is a relative
    // Frobenius tolerance.  The comparison is an early-stop heuristic
    // that works well when the function is O(1).  For functions with
    // magnitude far from 1, the heuristic may converge too early
    // (small function) or too late (large function); increase
    // max_sweeps as a hard upper bound if needed.
    if (max_err <= opts.eps)
      break;

    // Build seed_pivots from worst-offender linear positions.
    // seed_pivots[k] stores k indices per pivot (modes 0..k-1).
    std::vector<std::vector<int>> seed_pivots(static_cast<std::size_t>(d));
    for (auto& w : worst)
    {
      int64_t tmp = w.second;
      for (int i = 0; i < d; ++i)
      {
        const int sz                     = shp[static_cast<std::size_t>(i)];
        idx[static_cast<std::size_t>(i)] = tmp % sz;
        tmp /= sz;
      }
      for (int k = 1; k < d; ++k)
      {
        seed_pivots[static_cast<std::size_t>(k)].insert(
          seed_pivots[static_cast<std::size_t>(k)].end(),
          idx.begin(),
          idx.begin() + k);
      }
    }

    auto next_opts          = opts;
    next_opts.enrich_rounds = 0;
    next_opts.seed += 1;
    result = run_pass(next_opts, is_qtt ? nullptr : &seed_pivots);
  }

  return result;
}

// -------- tt_matrix (operator) --------

/**
 * @brief Build a TT-matrix via cross-interpolation.
 *
 * Packs (m_k, n_k) into a combined mode, cross-approximates
 * as a TT, then unpacks cores to TT-matrix form.
 *
 * @param func      Callable with signature double(const int* row_idx,
 *                  const int* col_idx).
 * @param row_shape Row mode sizes m_0, ..., m_{d-1}.
 * @param col_shape Column mode sizes n_0, ..., n_{d-1}.
 * @param opts      from_samples_options controlling method, max rank,
 *                  enrichment, and QTT parameters.
 * @return TT-matrix approximant.
 * @note  QTT methods are not yet implemented for TT-matrices.
 */
template <typename F>
inline tt_matrix from_samples(F&& func,
                              const std::vector<int>& row_shape,
                              const std::vector<int>& col_shape,
                              const from_samples_options& opts = {})
{
  // Validate shape entries are positive.
  for (std::size_t k = 0; k < row_shape.size(); ++k)
  {
    if (row_shape[k] <= 0)
    {
      std::fprintf(stderr,
                   "from_samples(tt_matrix): row_shape[%zu]=%d must be positive\n",
                   k, row_shape[k]);
      std::abort();
    }
  }
  for (std::size_t k = 0; k < col_shape.size(); ++k)
  {
    if (col_shape[k] <= 0)
    {
      std::fprintf(stderr,
                   "from_samples(tt_matrix): col_shape[%zu]=%d must be positive\n",
                   k, col_shape[k]);
      std::abort();
    }
  }

  const bool is_qtt = (opts.method == from_samples_method::qtt_dmrg_cross ||
                       opts.method == from_samples_method::qtt_tt_cross ||
                       opts.method == from_samples_method::qtt_als_cross ||
                       opts.method == from_samples_method::qtt_amen_cross);

  if (is_qtt)
  {
    return detail::qtt_matrix_wrap(std::forward<F>(func),
                                   opts.qtt_row_maxbits,
                                   opts.qtt_col_maxbits,
                                   opts.qtt_base,
                                   opts);
  }

  // Non-QTT path: pack (m_k, n_k) -> m_k * n_k, cross-approximate
  // the packed function as a tt, then unpack cores to tt_matrix.
  const int d = static_cast<int>(row_shape.size());
  std::vector<int> mn_shape(static_cast<std::size_t>(d));
  for (int k = 0; k < d; ++k)
  {
    const int64_t prod =
      static_cast<int64_t>(row_shape[k]) * col_shape[k];
    if (prod > std::numeric_limits<int>::max())
    {
      std::fprintf(stderr,
                   "from_samples(tt_matrix): mn product %ld exceeds "
                   "INT_MAX at axis %d; aborting.\n",
                   static_cast<long>(prod), k);
      std::abort();
    }
    mn_shape[k] = static_cast<int>(prod);
  }

  auto packed_func =
    detail::pack_matrix_func(std::forward<F>(func), row_shape, col_shape);
  tt packed = from_samples(packed_func, mn_shape, opts);

  std::vector<tt_matrix_core> mcs;
  mcs.reserve(static_cast<std::size_t>(d));
  for (int k = 0; k < d; ++k)
  {
    const tt_core& tc = packed.core(k);
    tt_matrix_core mc(
      tc.r_left(), row_shape[k], col_shape[k], tc.r_right());
    std::memcpy(mc.data(),
                tc.data(),
                sizeof(double) * static_cast<std::size_t>(tc.size()));
    mcs.push_back(std::move(mc));
  }
  return tt_matrix(std::move(mcs));
}

}  // namespace tensor_train
}  // namespace mva
//
// :D
//
