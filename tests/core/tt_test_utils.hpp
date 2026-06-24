/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Shared helpers for tensor_train test executables (norm, diff, dense construction, tolerance checks)
*/
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace mva
{
namespace tensor_train
{
namespace test_utils
{

// Sup-norm of a flat vector.
inline double max_abs(const std::vector<double>& v)
{
  double m = 0.0;
  for (double x : v)
  {
    const double a = std::fabs(x);
    if (a > m)
      m = a;
  }
  return m;
}

// Sup-norm of (a - b).  Returns a huge sentinel on size mismatch and
// prints a diagnostic so the caller can surface the failure.
inline double max_abs_diff(const std::vector<double>& a,
                           const std::vector<double>& b)
{
  if (a.size() != b.size())
  {
    std::fprintf(stderr, "max_abs_diff: size mismatch: %zu vs %zu\n",
                 a.size(), b.size());
    return 1e300;
  }
  double m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
  {
    const double d = std::fabs(a[i] - b[i]);
    if (d > m)
      m = d;
  }
  return m;
}

// Frobenius difference between two flat vectors.
inline double frob_diff(const std::vector<double>& a,
                        const std::vector<double>& b)
{
  if (a.size() != b.size())
  {
    std::fprintf(stderr,
                 "frob_diff: size mismatch (a=%zu, b=%zu)\n",
                 a.size(), b.size());
    return 1e300;
  }
  double s            = 0.0;
  const std::size_t n = a.size();
  for (std::size_t i = 0; i < n; ++i)
  {
    const double d = a[i] - b[i];
    s += d * d;
  }
  return std::sqrt(s);
}

inline double frob_norm(const std::vector<double>& a)
{
  double s = 0.0;
  for (double v : a)
    s += v * v;
  return std::sqrt(s);
}

// Build a deterministic-pseudorandom dense tensor of given shape, with
// uniform [-1, 1] entries.
inline std::vector<double> make_dense(const std::vector<int>& shape,
                                      unsigned seed)
{
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  long long total = shape.empty() ? 0 : 1;
  for (int n : shape)
    total *= n;
  std::vector<double> v(static_cast<std::size_t>(total));
  for (auto& x : v)
    x = dist(rng);
  return v;
}

// Print sup-error vs tolerance and return whether the bound is met.
inline bool check_close(const char* tag,
                        const std::vector<double>& got,
                        const std::vector<double>& want,
                        double tol)
{
  const double err = max_abs_diff(got, want);
  std::printf("  [%s] sup-err = %.3e (tol %.3e)\n", tag, err, tol);
  return err <= tol;
}

}  // namespace test_utils
}  // namespace tensor_train
}  // namespace mva
//
// :D
//
