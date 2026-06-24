/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Test: dmrg_cross / tt_cross -- function-based TT construction via
*/
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "tensor_train.hpp"
#include "tt_test_utils.hpp"

namespace tt_ns = mva::tensor_train;
namespace tu    = tt_ns::test_utils;

// Separable rank-1 function: f(idx) = prod_k (idx[k] + 1) / n_k.
static double separable_func(const int* idx)
{
  double val = 1.0;
  for (int k = 0; k < 3; ++k)
    val *= (idx[k] + 1.0) / 4.0;
  return val;
}

// Sum of two separable functions: rank-2 TT.
static double rank2_func(const int* idx)
{
  double a = 1.0, b = 1.0;
  for (int k = 0; k < 3; ++k)
  {
    a *= (idx[k] + 1.0) / 4.0;
    b *= (4.0 - idx[k]) / 4.0;
  }
  return a + b;
}

// Sum-of-products rank-2 function on 4 modes: f = (i0+1)(i1+1) + (i2+1)(i3+1)
static double d4_rank2_func(const int* idx)
{
  return static_cast<double>((idx[0] + 1) * (idx[1] + 1)
                           + (idx[2] + 1) * (idx[3] + 1));
}

int main()
{
  // ---------- dmrg_cross ----------

  std::printf("--- dmrg_cross: rank-1 function (3 modes, 4x4x4) ---\n");

  {
    std::vector<int> shape = { 4, 4, 4 };

    const int total = 64;
    std::vector<double> dense(static_cast<std::size_t>(total));
    int idx[3];
    for (int i0 = 0; i0 < 4; ++i0)
    {
      idx[0] = i0;
      for (int i1 = 0; i1 < 4; ++i1)
      {
        idx[1] = i1;
        for (int i2 = 0; i2 < 4; ++i2)
        {
          idx[2] = i2;
          dense[static_cast<std::size_t>((i0 * 4 + i1) * 4 + i2)] =
            separable_func(idx);
        }
      }
    }

    tt_ns::from_samples_options opts;
    opts.method     = tt_ns::from_samples_method::dmrg_cross;
    opts.eps        = 1.0e-12;
    opts.max_sweeps = 4;
    opts.init_rank  = 2;
    opts.seed       = 42;

    auto A_cross = tt_ns::from_samples(separable_func, shape, opts);
    auto A_dense = tt_ns::from_dense(dense.data(), shape, 1.0e-12);

    double err = tu::max_abs_diff(A_cross.to_dense(), A_dense.to_dense());
    std::printf("  cross vs dense ref:  sup-err = %8.2e\n", err);

    auto ranks = A_cross.ranks();
    int max_r  = 0;
    for (int r : ranks)
      if (r > max_r)
        max_r = r;
    std::printf("  max rank: %d  (expect 1)\n", max_r);

    bool ok = (err < 1.0e-6 && max_r == 1);
    std::printf("  dmrg_cross rank-1: %s\n", ok ? "PASS" : "FAIL");
    if (!ok)
      return 1;
  }

  std::printf("--- dmrg_cross: rank-2 function (3 modes, 4x4x4) ---\n");

  {
    std::vector<int> shape = { 4, 4, 4 };

    const int total = 64;
    std::vector<double> dense(static_cast<std::size_t>(total));
    int idx[3];
    for (int i0 = 0; i0 < 4; ++i0)
    {
      idx[0] = i0;
      for (int i1 = 0; i1 < 4; ++i1)
      {
        idx[1] = i1;
        for (int i2 = 0; i2 < 4; ++i2)
        {
          idx[2] = i2;
          dense[static_cast<std::size_t>((i0 * 4 + i1) * 4 + i2)] =
            rank2_func(idx);
        }
      }
    }

    tt_ns::from_samples_options opts;
    opts.method     = tt_ns::from_samples_method::dmrg_cross;
    opts.eps        = 1.0e-10;
    opts.max_sweeps = 6;
    opts.init_rank  = 3;
    opts.seed       = 12345;

    auto A_cross = tt_ns::from_samples(rank2_func, shape, opts);
    auto A_dense = tt_ns::from_dense(dense.data(), shape, 1.0e-12);

    double err = tu::max_abs_diff(A_cross.to_dense(), A_dense.to_dense());
    std::printf("  cross vs dense ref:  sup-err = %8.2e\n", err);

    auto ranks = A_cross.ranks();
    int max_r  = 0;
    for (int r : ranks)
      if (r > max_r)
        max_r = r;
    std::printf("  max rank: %d  (expect 2)\n", max_r);

    bool ok = (err < 1.0e-6 && max_r == 2);
    std::printf("  dmrg_cross rank-2: %s\n", ok ? "PASS" : "FAIL");
    if (!ok)
      return 1;
  }

  // ---------- tt_cross ----------

  std::printf("--- tt_cross: rank-1 function (3 modes, 4x4x4) ---\n");

  {
    std::vector<int> shape = { 4, 4, 4 };

    const int total = 64;
    std::vector<double> dense(static_cast<std::size_t>(total));
    int idx[3];
    for (int i0 = 0; i0 < 4; ++i0)
    {
      idx[0] = i0;
      for (int i1 = 0; i1 < 4; ++i1)
      {
        idx[1] = i1;
        for (int i2 = 0; i2 < 4; ++i2)
        {
          idx[2] = i2;
          dense[static_cast<std::size_t>((i0 * 4 + i1) * 4 + i2)] =
            separable_func(idx);
        }
      }
    }

    tt_ns::from_samples_options opts;
    opts.method    = tt_ns::from_samples_method::tt_cross;
    opts.max_rank  = 1;
    opts.init_rank = 2;
    opts.seed      = 42;

    auto A_cross = tt_ns::from_samples(separable_func, shape, opts);
    auto A_dense = tt_ns::from_dense(dense.data(), shape, 1.0e-12);

    double err = tu::max_abs_diff(A_cross.to_dense(), A_dense.to_dense());
    std::printf("  cross vs dense ref:  sup-err = %8.2e\n", err);

    auto ranks = A_cross.ranks();
    int max_r  = 0;
    for (int r : ranks)
      if (r > max_r)
        max_r = r;
    std::printf("  max rank: %d  (expect 1)\n", max_r);

    bool ok = (err < 1.0e-6 && max_r == 1);
    std::printf("  tt_cross rank-1: %s\n", ok ? "PASS" : "FAIL");
    if (!ok)
      return 1;
  }

  std::printf("--- tt_cross: rank-2 function (3 modes, 4x4x4) ---\n");

  {
    std::vector<int> shape = { 4, 4, 4 };

    const int total = 64;
    std::vector<double> dense(static_cast<std::size_t>(total));
    int idx[3];
    for (int i0 = 0; i0 < 4; ++i0)
    {
      idx[0] = i0;
      for (int i1 = 0; i1 < 4; ++i1)
      {
        idx[1] = i1;
        for (int i2 = 0; i2 < 4; ++i2)
        {
          idx[2] = i2;
          dense[static_cast<std::size_t>((i0 * 4 + i1) * 4 + i2)] =
            rank2_func(idx);
        }
      }
    }

    tt_ns::from_samples_options opts;
    opts.method    = tt_ns::from_samples_method::tt_cross;
    opts.max_rank  = 2;
    opts.init_rank = 3;
    opts.seed      = 12345;

    auto A_cross = tt_ns::from_samples(rank2_func, shape, opts);
    auto A_dense = tt_ns::from_dense(dense.data(), shape, 1.0e-12);

    double err = tu::max_abs_diff(A_cross.to_dense(), A_dense.to_dense());
    std::printf("  cross vs dense ref:  sup-err = %8.2e\n", err);

    auto ranks = A_cross.ranks();
    int max_r  = 0;
    for (int r : ranks)
      if (r > max_r)
        max_r = r;
    std::printf("  max rank: %d  (expect 2)\n", max_r);

    bool ok = (err < 1.0e-6 && max_r == 2);
    std::printf("  tt_cross rank-2: %s\n", ok ? "PASS" : "FAIL");
    if (!ok)
      return 1;
  }

  // ---------- tt_cross d=4 regression ----------
  std::printf("--- tt_cross d=4: rank-2 function (3x3x3x3) ---\n");

  {
    std::vector<int> shape = { 3, 3, 3, 3 };
    const int total = 81;
    std::vector<double> dense(static_cast<std::size_t>(total));
    int idx2[4];
    for (int i0 = 0; i0 < 3; ++i0) {
      idx2[0] = i0;
      for (int i1 = 0; i1 < 3; ++i1) {
        idx2[1] = i1;
        for (int i2 = 0; i2 < 3; ++i2) {
          idx2[2] = i2;
          for (int i3 = 0; i3 < 3; ++i3) {
            idx2[3] = i3;
            dense[static_cast<std::size_t>(((i0*3 + i1)*3 + i2)*3 + i3)] =
              d4_rank2_func(idx2);
          }
        }
      }
    }

    tt_ns::from_samples_options opts;
    opts.method    = tt_ns::from_samples_method::tt_cross;
    opts.max_rank  = 3;
    opts.init_rank = 4;
    opts.seed      = 8675309;

    auto A_cross = tt_ns::from_samples(d4_rank2_func, shape, opts);
    auto A_ref   = tt_ns::from_dense(dense.data(), shape, 1.0e-12);

    double err = tu::max_abs_diff(A_cross.to_dense(), A_ref.to_dense());
    std::printf("  cross vs ref: sup-err = %8.2e\n", err);
    bool ok = (err < 1.0e-6);
    std::printf("  tt_cross d=4: %s\n", ok ? "PASS" : "FAIL");
    if (!ok) return 1;
  }

  // ---------- d=1 (use default dmrg_cross) ----------

  std::printf("--- d=1 function ---\n");

  {
    std::vector<int> shape = { 7 };

    auto cross =
      tt_ns::from_samples([](const int* ii) { return ii[0] + 1.0; }, shape);

    std::vector<double> c_d = cross.to_dense();
    double ref_seven[7];
    for (int i = 0; i < 7; ++i)
      ref_seven[i] = i + 1.0;

    double err =
      tu::max_abs_diff(c_d, std::vector<double>(ref_seven, ref_seven + 7));
    std::printf("  sup-err = %8.2e\n", err);
    bool ok = (err < 1.0e-12);
    std::printf("  d=1 test: %s\n", ok ? "PASS" : "FAIL");
    if (!ok)
      return 1;
  }

  // ---------- QTT: dmrg_cross, 1D rank-1 ----------

  std::printf("--- qtt_dmrg_cross: 1D, maxbits=5 ---\n");

  {
    tt_ns::from_samples_options opts;
    opts.method      = tt_ns::from_samples_method::qtt_dmrg_cross;
    opts.eps         = 1.0e-6;
    opts.max_rank    = 4;
    opts.max_sweeps  = 10;
    opts.init_rank   = 4;
    opts.seed        = 99;
    opts.qtt_maxbits = { 5 };

    auto linear = [](const int* gi) { return (gi[0] + 1.0); };

    auto t = tt_ns::from_samples(linear, {}, opts);

    std::vector<double> dense(32);
    for (int i = 0; i < 32; ++i)
    {
      int gi[]                           = { i };
      dense[static_cast<std::size_t>(i)] = linear(gi);
    }
    auto ref = tt_ns::from_dense(
      dense.data(), std::vector<int>{ 2, 2, 2, 2, 2 }, 1.0e-12);
    double err = tu::max_abs_diff(t.to_dense(), ref.to_dense());
    std::printf("  cross vs dense:  sup-err = %8.2e\n", err);

    auto rks  = t.ranks();
    int max_r = 0;
    for (int r : rks)
      if (r > max_r)
        max_r = r;
    std::printf("  max rank: %d\n", max_r);

    bool ok = (err < 1.0e-6 && t.d() == 5);
    std::printf("  qtt_dmrg_cross rank-1: %s\n", ok ? "PASS" : "FAIL");
    if (!ok)
      return 1;
  }

  // ---------- QTT: tt_cross, 1D linear ----------

  std::printf("--- qtt_tt_cross: 1D linear, maxbits=4 ---\n");

  {
    tt_ns::from_samples_options opts;
    opts.method      = tt_ns::from_samples_method::qtt_tt_cross;
    opts.max_rank    = 3;
    opts.init_rank   = 3;
    opts.seed        = 77;
    opts.qtt_maxbits = { 4 };

    auto t =
      tt_ns::from_samples([](const int* gi) { return gi[0] + 1.0; }, {}, opts);

    auto d_v        = t.to_dense();
    double max_diff = 0.0;
    for (int i = 0; i < 16; ++i)
    {
      double diff = std::abs(d_v[static_cast<std::size_t>(i)] - (i + 1.0));
      if (diff > max_diff)
        max_diff = diff;
    }
    std::printf("  sup-err = %8.2e\n", max_diff);

    auto rks  = t.ranks();
    int max_r = 0;
    for (int r : rks)
      if (r > max_r)
        max_r = r;
    std::printf("  max rank: %d\n", max_r);

    bool ok = (max_diff < 1.0e-6);
    std::printf("  qtt_tt_cross 1D: %s\n", ok ? "PASS" : "FAIL");
    if (!ok)
      return 1;
  }

  // ---------- QTT: dmrg_cross, 1D rank-2 ----------

  std::printf("--- qtt_dmrg_cross: 1D rank-2, maxbits=5 ---\n");

  {
    tt_ns::from_samples_options opts;
    opts.method      = tt_ns::from_samples_method::qtt_dmrg_cross;
    opts.eps         = 1.0e-4;
    opts.max_rank    = 8;
    opts.max_sweeps  = 10;
    opts.init_rank   = 4;
    opts.seed        = 314;
    opts.qtt_maxbits = { 5 };

    auto r2_func = [](const int* gi)
    {
      double x = gi[0] / 31.0;
      return (gi[0] + 1.0) + 16.0 * std::sin(x * 3.1415926535);
    };

    auto t = tt_ns::from_samples(r2_func, {}, opts);

    auto rks  = t.ranks();
    int max_r = 0;
    for (int r : rks)
      if (r > max_r)
        max_r = r;

    bool ok = (t.d() == 5 && max_r >= 2 && max_r <= 8);
    std::printf("  max rank: %d\n", max_r);
    std::printf("  qtt_dmrg_cross rank-2: %s\n", ok ? "PASS" : "FAIL");
    if (!ok)
      return 1;
  }

  // ---------- QTT: d=1 single-mode ----------

  std::printf("--- qtt: d=1 (maxbits=1) ---\n");

  {
    tt_ns::from_samples_options opts;
    opts.method      = tt_ns::from_samples_method::qtt_dmrg_cross;
    opts.qtt_maxbits = { 1 };

    auto t =
      tt_ns::from_samples([](const int* gi) { return gi[0] + 2.0; }, {}, opts);

    auto d_v = t.to_dense();
    bool ok =
      (std::abs(d_v[0] - 2.0) < 1.0e-15 && std::abs(d_v[1] - 3.0) < 1.0e-15);
    std::printf("  d=1 qtt: %s\n", ok ? "PASS" : "FAIL");
    if (!ok)
      return 1;
  }

  // ---------- QTT: 3D Gaussian, maxbits={5,5,5} ----------

  std::printf("--- qtt_dmrg_cross: 3D Gaussian, maxbits={5,5,5} ---\n");

  {
    constexpr int bits      = 5;
    constexpr int N         = 1 << bits;  // 32
    constexpr double cx     = (N - 1) * 0.5;
    constexpr double sigma  = N / 6.0;
    constexpr double inv2s2 = 1.0 / (2.0 * sigma * sigma);

    auto gauss = [=](const int* gi) -> double
    {
      double dx = static_cast<double>(gi[0]) - cx;
      double dy = static_cast<double>(gi[1]) - cx;
      double dz = static_cast<double>(gi[2]) - cx;
      return std::exp(-(dx * dx + dy * dy + dz * dz) * inv2s2);
    };

    tt_ns::from_samples_options opts;
    opts.method      = tt_ns::from_samples_method::qtt_dmrg_cross;
    opts.eps         = 1.0e-6;
    opts.max_rank    = 30;
    opts.max_sweeps  = 20;
    opts.init_rank   = 8;
    opts.seed        = 42;
    opts.qtt_maxbits = { bits, bits, bits };
    opts.qtt_base    = 2;

    auto t = tt_ns::from_samples(gauss, {}, opts);

    const int d = t.d();
    auto rks    = t.ranks();
    int max_r   = 0;
    for (int r : rks)
      if (r > max_r)
        max_r = r;

    // Sample 512 random grid indices and compare against function.
    std::mt19937_64 rng(99);
    std::uniform_int_distribution<int> dist(0, N - 1);
    double max_err = 0.0;
    for (int s = 0; s < 512; ++s)
    {
      int gi[3]  = { dist(rng), dist(rng), dist(rng) };
      double ref = gauss(gi);

      // Convert grid indices to TT multi-index.
      int tt_idx[15];
      for (int dim = 0; dim < 3; ++dim)
      {
        for (int b = 0; b < bits; ++b)
          tt_idx[dim * bits + b] = (gi[dim] >> (bits - 1 - b)) & 1;
      }
      double tt_val = tt_ns::eval_at(t, tt_idx);
      double diff   = std::abs(tt_val - ref);
      if (diff > max_err)
        max_err = diff;
    }

    std::printf("  d=%d  max-internal-rank=%d\n", d, max_r);
    std::printf("  max-abs-error (512 samples) = %8.2e\n", max_err);

    bool ok = (max_err < 1.0e-6 && max_r >= 4);
    std::printf("  qtt_dmrg_cross 3D Gaussian: %s\n", ok ? "PASS" : "FAIL");
    if (!ok)
      return 1;
  }

  // ---------- Residual enrichment ----------

  std::printf("--- dmrg_cross + enrichment: weak eps -> tightened ---\n");

  {
    auto func = [](const int* idx) -> double
    { return (idx[0] + 1.0) / 16.0 * (idx[1] + 1.0) / 16.0; };

    // Run with weak eps -> should produce mediocre accuracy.
    tt_ns::from_samples_options wopts;
    wopts.method     = tt_ns::from_samples_method::dmrg_cross;
    wopts.eps        = 1.0e-2;
    wopts.max_sweeps = 5;
    wopts.init_rank  = 2;
    wopts.seed       = 123;

    auto weak       = tt_ns::from_samples(func, { 16, 16 }, wopts);
    double weak_err = 0.0;
    {
      auto dv = weak.to_dense();
      for (int i = 0; i < 256; ++i)
      {
        int mi[]    = { i / 16, i % 16 };
        double diff = std::abs(dv[static_cast<std::size_t>(i)] - func(mi));
        if (diff > weak_err)
          weak_err = diff;
      }
    }
    std::printf("  weak (eps=1e-2, no enrich): max-err = %8.2e\n", weak_err);

    // Same params but with enrichment: should converge to 1e-2.
    tt_ns::from_samples_options eopts;
    eopts.method         = tt_ns::from_samples_method::dmrg_cross;
    eopts.eps            = 1.0e-2;
    eopts.max_sweeps     = 5;
    eopts.init_rank      = 2;
    eopts.seed           = 123;
    eopts.enrich_rounds  = 3;
    eopts.enrich_samples = 512;
    eopts.enrich_k       = 32;

    auto enriched     = tt_ns::from_samples(func, { 16, 16 }, eopts);
    double enrich_err = 0.0;
    {
      auto dv = enriched.to_dense();
      for (int i = 0; i < 256; ++i)
      {
        int mi[]    = { i / 16, i % 16 };
        double diff = std::abs(dv[static_cast<std::size_t>(i)] - func(mi));
        if (diff > enrich_err)
          enrich_err = diff;
      }
    }
    std::printf(
      "  enriched (eps=1e-2, rounds=3): "
      "max-err = %8.2e\n",
      enrich_err);

    auto rks  = enriched.ranks();
    int max_r = 0;
    for (int r : rks)
      if (r > max_r)
        max_r = r;
    std::printf("  enriched max-rank=%d\n", max_r);

    bool ok = (enrich_err <= std::max(weak_err, 1.0e-2));
    std::printf("  enrichment test: %s\n", ok ? "PASS" : "FAIL");
    if (!ok)
      return 1;
  }

  // ---------- als_cross: rank-1 function (3 modes, 4x4x4) ----------

  std::printf("--- als_cross: rank-1 function (3 modes, 4x4x4) ---\n");

  {
    auto sep_func = [](const int* ii) -> double
    { return (ii[0] + 1.0) * (ii[1] + 1.0) * (ii[2] + 1.0); };

    tt_ns::from_samples_options opts;
    opts.method     = tt_ns::from_samples_method::als_cross;
    opts.eps        = 1.0e-10;
    opts.max_rank   = 0;
    opts.max_sweeps = 20;
    opts.init_rank  = 2;
    opts.seed       = 99;

    auto t = tt_ns::from_samples(sep_func, { 4, 4, 4 }, opts);

    std::vector<double> dense(4 * 4 * 4);
    for (int i0 = 0; i0 < 4; ++i0)
      for (int i1 = 0; i1 < 4; ++i1)
        for (int i2 = 0; i2 < 4; ++i2)
        {
          int ii[] = { i0, i1, i2 };
          dense[static_cast<std::size_t>((i0 * 4 + i1) * 4 + i2)] =
            sep_func(ii);
        }
    auto ref = tt_ns::from_dense(dense.data(), { 4, 4, 4 }, 1.0e-12);

    double err = tu::max_abs_diff(t.to_dense(), ref.to_dense());
    std::printf("  cross vs dense ref:  sup-err = %8.2e\n", err);

    auto rks  = t.ranks();
    int max_r = 0;
    for (int r : rks)
      if (r > max_r)
        max_r = r;
    std::printf("  max rank: %d  (expect 1)\n", max_r);

    bool ok = (err < 1.0e-8 && max_r == 1);
    std::printf("  als_cross rank-1: %s\n", ok ? "PASS" : "FAIL");
    if (!ok)
      return 1;
  }

  // ---------- als_cross: rank-2 function (3 modes, 4x4x4) ----------

  std::printf("--- als_cross: rank-2 function (3 modes, 4x4x4) ---\n");

  {
    auto r2_func = [](const int* ii) -> double
    {
      return (ii[0] + 1.0) * (ii[1] + 1.0) * (ii[2] + 1.0) +
             0.5 * ii[0] * ii[1] * ii[2];
    };

    tt_ns::from_samples_options opts;
    opts.method     = tt_ns::from_samples_method::als_cross;
    opts.eps        = 1.0e-10;
    opts.max_rank   = 0;
    opts.max_sweeps = 20;
    opts.init_rank  = 4;
    opts.seed       = 123;

    auto t = tt_ns::from_samples(r2_func, { 4, 4, 4 }, opts);

    std::vector<double> dense(4 * 4 * 4);
    for (int i0 = 0; i0 < 4; ++i0)
      for (int i1 = 0; i1 < 4; ++i1)
        for (int i2 = 0; i2 < 4; ++i2)
        {
          int ii[] = { i0, i1, i2 };
          dense[static_cast<std::size_t>((i0 * 4 + i1) * 4 + i2)] =
            r2_func(ii);
        }
    auto ref = tt_ns::from_dense(dense.data(), { 4, 4, 4 }, 1.0e-12);

    double err = tu::max_abs_diff(t.to_dense(), ref.to_dense());
    std::printf("  cross vs dense ref:  sup-err = %8.2e\n", err);

    auto rks  = t.ranks();
    int max_r = 0;
    for (int r : rks)
      if (r > max_r)
        max_r = r;
    std::printf("  max rank: %d  (expect 2)\n", max_r);

    bool ok = (err < 1.0e-6 && max_r == 2);
    std::printf("  als_cross rank-2: %s\n", ok ? "PASS" : "FAIL");
    if (!ok)
      return 1;
  }

  // ---------- amen_cross: rank-1 function ----------

  std::printf("--- amen_cross: rank-1 function (3 modes, 4x4x4) ---\n");

  {
    std::vector<int> shape = { 4, 4, 4 };

    const int total = 64;
    std::vector<double> dense(static_cast<std::size_t>(total));
    int idx[3];
    for (int i0 = 0; i0 < 4; ++i0)
    {
      idx[0] = i0;
      for (int i1 = 0; i1 < 4; ++i1)
      {
        idx[1] = i1;
        for (int i2 = 0; i2 < 4; ++i2)
        {
          idx[2] = i2;
          dense[static_cast<std::size_t>((i0 * 4 + i1) * 4 + i2)] =
            separable_func(idx);
        }
      }
    }

    tt_ns::from_samples_options opts;
    opts.method     = tt_ns::from_samples_method::amen_cross;
    opts.eps        = 1.0e-12;
    opts.max_rank   = 2;
    opts.max_sweeps = 8;
    opts.init_rank  = 2;
    opts.seed       = 42;

    auto A_cross = tt_ns::from_samples(separable_func, shape, opts);
    auto A_dense = tt_ns::from_dense(dense.data(), shape, 1.0e-12);

    double err = tu::max_abs_diff(A_cross.to_dense(), A_dense.to_dense());
    std::printf("  cross vs dense ref:  sup-err = %8.2e\n", err);

    auto ranks = A_cross.ranks();
    int max_r  = 0;
    for (int r : ranks)
      if (r > max_r)
        max_r = r;
    std::printf("  max rank: %d  (expect 1)\n", max_r);

    bool ok = (err < 1.0e-6 && max_r == 1);
    std::printf("  amen_cross rank-1: %s\n", ok ? "PASS" : "FAIL");
    if (!ok)
      return 1;
  }

  // ---------- amen_cross: rank-2 function ----------

  std::printf("--- amen_cross: rank-2 function (3 modes, 4x4x4) ---\n");

  {
    std::vector<int> shape = { 4, 4, 4 };

    const int total = 64;
    std::vector<double> dense(static_cast<std::size_t>(total));
    int idx[3];
    for (int i0 = 0; i0 < 4; ++i0)
    {
      idx[0] = i0;
      for (int i1 = 0; i1 < 4; ++i1)
      {
        idx[1] = i1;
        for (int i2 = 0; i2 < 4; ++i2)
        {
          idx[2] = i2;
          dense[static_cast<std::size_t>((i0 * 4 + i1) * 4 + i2)] =
            rank2_func(idx);
        }
      }
    }

    tt_ns::from_samples_options opts;
    opts.method     = tt_ns::from_samples_method::amen_cross;
    opts.eps        = 1.0e-12;
    opts.max_rank   = 4;
    opts.max_sweeps = 8;
    opts.init_rank  = 3;
    opts.seed       = 314;

    auto A_cross = tt_ns::from_samples(rank2_func, shape, opts);
    auto A_dense = tt_ns::from_dense(dense.data(), shape, 1.0e-12);

    double err = tu::max_abs_diff(A_cross.to_dense(), A_dense.to_dense());
    std::printf("  cross vs dense ref:  sup-err = %8.2e\n", err);

    auto ranks = A_cross.ranks();
    int max_r  = 0;
    for (int r : ranks)
      if (r > max_r)
        max_r = r;
    std::printf("  max rank: %d  (expect 2)\n", max_r);

    bool ok = (err < 1.0e-6 && max_r == 2);
    std::printf("  amen_cross rank-2: %s\n", ok ? "PASS" : "FAIL");
    if (!ok)
      return 1;
  }

  // ---------- amen_cross: enrichment test ----------

  std::printf("--- amen_cross: enrichment (rank-2, weak eps) ---\n");

  {
    std::vector<int> shape = { 4, 4, 4 };

    const int total = 64;
    std::vector<double> dense(static_cast<std::size_t>(total));
    int idx[3];
    for (int i0 = 0; i0 < 4; ++i0)
    {
      idx[0] = i0;
      for (int i1 = 0; i1 < 4; ++i1)
      {
        idx[1] = i1;
        for (int i2 = 0; i2 < 4; ++i2)
        {
          idx[2] = i2;
          dense[static_cast<std::size_t>((i0 * 4 + i1) * 4 + i2)] =
            rank2_func(idx);
        }
      }
    }

    tt_ns::from_samples_options opts;
    opts.method         = tt_ns::from_samples_method::amen_cross;
    opts.eps            = 1.0e-2;
    opts.max_rank       = 4;
    opts.max_sweeps     = 8;
    opts.init_rank      = 3;
    opts.seed           = 99;
    opts.enrich_rounds  = 3;
    opts.enrich_samples = 512;
    opts.enrich_k       = 32;

    auto A_cross = tt_ns::from_samples(rank2_func, shape, opts);
    auto A_dense = tt_ns::from_dense(dense.data(), shape, 1.0e-12);

    double err = tu::max_abs_diff(A_cross.to_dense(), A_dense.to_dense());
    std::printf("  cross vs dense ref:  sup-err = %8.2e\n", err);

    auto ranks = A_cross.ranks();
    int max_r  = 0;
    for (int r : ranks)
      if (r > max_r)
        max_r = r;
    std::printf("  max rank: %d\n", max_r);

    bool ok = (err < 1.0e-6);
    std::printf("  amen_cross enrichment: %s\n", ok ? "PASS" : "FAIL");
    if (!ok)
      return 1;
  }

  // ---------- from_samples (tt_matrix) ----------

  std::printf("--- from_samples(tt_matrix): rank-2 matrix (2 modes, 3x3 x 3x3) ---\n");

  {
    // A(row, col) = prod_k ((row[k]+1)*(col[k]+1)) + prod_k ((4-row[k])*(4-col[k]))
    auto mat_func = [](const int* row_idx, const int* col_idx) -> double
    {
      double a = 1.0, b = 1.0;
      for (int k = 0; k < 2; ++k)
      {
        a *= (row_idx[k] + 1.0) * (col_idx[k] + 1.0) / 9.0;
        b *= (4.0 - row_idx[k]) * (4.0 - col_idx[k]) / 9.0;
      }
      return a + b;
    };

    std::vector<int> rs = { 3, 3 };
    std::vector<int> cs = { 3, 3 };
    const int M = 9, N = 9;
    std::vector<double> dense(static_cast<std::size_t>(M * N));
    int ri[2], ci[2];
    for (int i0 = 0; i0 < 3; ++i0)
    {
      ri[0] = i0;
      for (int i1 = 0; i1 < 3; ++i1)
      {
        ri[1] = i1;
        const int I = i0 * 3 + i1;
        for (int j0 = 0; j0 < 3; ++j0)
        {
          ci[0] = j0;
          for (int j1 = 0; j1 < 3; ++j1)
          {
            ci[1] = j1;
            const int J = j0 * 3 + j1;
            dense[static_cast<std::size_t>(I * N + J)] =
              mat_func(ri, ci);
          }
        }
      }
    }

    auto A_dense = tt_ns::from_dense(dense.data(), rs, cs, 1.0e-12);

    tt_ns::from_samples_options opts;
    opts.method     = tt_ns::from_samples_method::dmrg_cross;
    opts.eps        = 1.0e-12;
    opts.max_sweeps = 4;
    opts.init_rank  = 2;
    opts.seed       = 42;

    auto A_cross = tt_ns::from_samples(mat_func, rs, cs, opts);

    auto dense_cross = A_cross.to_dense();
    auto dense_ref   = A_dense.to_dense();
    double err = tu::max_abs_diff(dense_cross, dense_ref);
    std::printf("  cross vs dense ref:  sup-err = %8.2e\n", err);

    auto ranks = A_cross.ranks();
    int max_r  = 0;
    for (int r : ranks)
      if (r > max_r)
        max_r = r;
    std::printf("  max rank: %d  (expect 2)\n", max_r);

    bool ok = (err < 1.0e-6);
    std::printf("  from_samples(tt_matrix) dmrg_cross rank-2: %s\n",
                ok ? "PASS" : "FAIL");
    if (!ok)
      return 1;
  }

  std::printf("test_tt_cross: OK\n");
  return 0;
}
//
// :D
//
