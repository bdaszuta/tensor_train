/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: nanobind Python bindings for mva::tensor_train
*/
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <utility>

#include "tensor_train.hpp"

namespace nb = nanobind;
using namespace nb::literals;

using namespace mva::tensor_train;

namespace
{

nb::ndarray<nb::numpy, double> to_numpy(std::vector<double>&& data)
{
  if (data.empty())
  {
    return nb::ndarray<nb::numpy, double>(nullptr, { static_cast<size_t>(0) });
  }
  auto* vec = new std::vector<double>(std::move(data));
  nb::capsule owner(vec,
                    [](void* p) noexcept
                    { delete static_cast<std::vector<double>*>(p); });
  return nb::ndarray<nb::numpy, double>(vec->data(), { vec->size() }, owner);
}

}  // namespace

// Return a read-only 3D numpy view into tt_core data.
// The optional capsule keeps the backing storage alive.
nb::ndarray<nb::numpy, const double> core_view(
    const tt_core& c,
    nb::capsule owner = nb::capsule())
{
  auto dims = c.dims();
  return nb::ndarray<nb::numpy, const double>(
    c.data(),
    { static_cast<size_t>(dims[0]),
      static_cast<size_t>(dims[1]),
      static_cast<size_t>(dims[2]) },
    std::move(owner));
}

NB_MODULE(tt_core, m)
{
  m.doc() = "mva::tensor_train C++ library bindings";

  // ==================================================================
  // Enums
  // ==================================================================

  nb::enum_<round_method>(m, "round_method",
    "Compression method. svd_streaming is matvec_round only; "
    "svd_naive is matmat_round only; both are rejected by round().")
    .value("svd", round_method::svd)
    .value("svd_streaming", round_method::svd_streaming)
    .value("svd_naive", round_method::svd_naive)
    .value("als", round_method::als)
    .value("dmrg", round_method::dmrg);

  nb::enum_<round_gauge>(m, "round_gauge", "Output gauge for round().")
    .value("none", round_gauge::none)
    .value("right_canonical", round_gauge::right_canonical);

  nb::enum_<from_dense_method>(
    m, "from_dense_method", "Compression method for from_dense().")
    .value("svd", from_dense_method::svd)
    .value("qtt", from_dense_method::qtt);

  // ==================================================================
  // Options / result structs
  // ==================================================================

  nb::class_<round_options>(m, "round_options")
    .def(nb::init<>())
    .def_rw("eps", &round_options::eps)
    .def_rw("max_rank", &round_options::max_rank)
    .def_rw("method", &round_options::method)
    .def_rw("max_iters", &round_options::max_iters)
    .def_rw("tol", &round_options::tol)
    .def_rw("seed", &round_options::seed)
    .def_rw("gauge", &round_options::gauge);

  nb::class_<from_dense_options>(m, "from_dense_options")
    .def(nb::init<>())
    .def_rw("eps", &from_dense_options::eps)
    .def_rw("max_rank", &from_dense_options::max_rank)
    .def_rw("method", &from_dense_options::method)
    .def_rw("qtt_base", &from_dense_options::qtt_base)
    .def_rw("validate", &from_dense_options::validate);

  nb::class_<round_result>(m, "round_result")
    .def(nb::init<>())
    .def_ro("iters_run", &round_result::iters_run)
    .def_ro("final_resid", &round_result::final_resid)
    .def_ro("converged", &round_result::converged);

  // ==================================================================
  // tt_core class (single TT core)
  // ==================================================================

  nb::class_<tt_core>(m, "tt_core")
    .def(nb::init<int, int, int>(),
         "r_left"_a,
         "n_phys"_a,
         "r_right"_a,
         "Allocate a zero-filled core of shape (r_left, n_phys, r_right).")
    .def_prop_ro("r_left", &tt_core::r_left)
    .def_prop_ro("n_phys", &tt_core::n_phys)
    .def_prop_ro("r_right", &tt_core::r_right)
    .def_prop_ro("dims", &tt_core::dims)
    .def_prop_ro("size", &tt_core::size)
    .def("data_view",
         [](const tt_core& c) { return core_view(c); },
         "Read-only 3D numpy view of shape (r_left, n_phys, r_right) "
         "into the core data (zero-copy).  "
         "The view is valid only as long as this TTCore object exists.")
    .def("fill",
         [](tt_core& c,
            nb::ndarray<nb::numpy, const double, nb::ndim<3>> src)
         {
           if (static_cast<int>(src.shape(0)) != c.r_left() ||
               static_cast<int>(src.shape(1)) != c.n_phys() ||
               static_cast<int>(src.shape(2)) != c.r_right())
           {
             throw std::runtime_error(
               "tt_core.fill: shape mismatch");
           }
           std::memcpy(c.data(),
                       src.data(),
                       sizeof(double) * static_cast<std::size_t>(c.size()));
         },
         "src"_a,
         "Copy data from a 3D numpy array into this core.  "
         "src must have shape (r_left, n_phys, r_right).");

  // ==================================================================
  // tt class
  // ==================================================================

  nb::class_<tt>(m, "tt")
    .def_prop_ro("shape", &tt::shape)
    .def_prop_ro("ranks", &tt::ranks)
    .def_prop_ro("d", &tt::d)
    .def_prop_ro("max_rank", &tt::max_rank)
    .def("to_dense", [](const tt& a) { return to_numpy(a.to_dense()); })
    .def("core",
         [](const tt& a, int k)
         { return a.core(k); },
         "k"_a,
         "Return a copy of core k.")
    .def_prop_ro("num_params", &tt::num_params)
    .def(
      "cores",
      [](const tt& a)
      {
        nb::list out;
        const int d = a.d();
        if (d == 0)
          return out;
        struct shared_ref
        {
          std::shared_ptr<tt> data;
          int refs;
        };
        auto* ref = new shared_ref{std::make_shared<tt>(a), d};
        for (int k = 0; k < d; ++k)
        {
          nb::capsule owner(ref,
                            [](void* p) noexcept
                            {
                              auto* r = static_cast<shared_ref*>(p);
                              if (--r->refs == 0)
                                delete r;
                            });
          out.append(core_view(ref->data->core(k), std::move(owner)));
        }
        return out;
      },
      "Return list of read-only 3D numpy views, one per core. "
      "Zero-copy -- views share a single TT copy kept alive "
      "until all arrays are garbage-collected.");

  // ==================================================================
  // Factory: from_dense
  // ==================================================================

  m.def(
    "from_dense",
    [](nb::ndarray<nb::numpy, const double> data,
       const std::vector<int>& shape,
       double eps) -> tt
    { return mva::tensor_train::from_dense(data.data(), shape, eps); },
    "data"_a,
    "shape"_a,
    "eps"_a);

  m.def(
    "from_dense",
    [](nb::ndarray<nb::numpy, const double> data,
       const std::vector<int>& shape,
       const from_dense_options& opts) -> tt
    { return mva::tensor_train::from_dense(data.data(), shape, opts); },
    "data"_a,
    "shape"_a,
    "opts"_a);


  // ==================================================================
  // Factory: tt constructors (zeros, ones, canonical_unit, random)
  // ==================================================================

  m.def(
    "zeros",
    [](const std::vector<int>& shape) -> tt
    { return mva::tensor_train::zeros(shape); },
    "shape"_a,
    "All-zero rank-1 TT with given mode sizes.");

  m.def(
    "ones",
    [](const std::vector<int>& shape) -> tt
    { return mva::tensor_train::ones(shape); },
    "shape"_a,
    "All-one rank-1 TT (outer product of ones vectors).");

  m.def(
    "canonical_unit",
    [](const std::vector<int>& shape,
       const std::vector<int>& idx) -> tt
    { return mva::tensor_train::canonical_unit(shape, idx); },
    "shape"_a,
    "idx"_a,
    "Canonical basis TT with a single 1 at multi-index idx.");

  m.def(
    "random",
    [](const std::vector<int>& shape,
       int max_rank,
       std::uint64_t seed) -> tt
    { return mva::tensor_train::random(shape, max_rank, seed); },
    "shape"_a,
    "max_rank"_a,
    "seed"_a,
    "Random TT with N(0,1) entries, bond ranks capped by max_rank.");

  // ==================================================================
  // Factory: from_cores
  // ==================================================================

  m.def(
    "from_cores",
    [](nb::list core_list) -> tt
    {
      std::vector<tt_core> cores;
      const int d = nb::len(core_list);
      cores.reserve(d);
      for (int k = 0; k < d; ++k)
      {
        auto arr =
          nb::cast<nb::ndarray<nb::numpy, const double, nb::ndim<3>>>(
            core_list[k]);
        int r_left  = static_cast<int>(arr.shape(0));
        int n_phys  = static_cast<int>(arr.shape(1));
        int r_right = static_cast<int>(arr.shape(2));
        // Enforce C-contiguity: stride[2]==1, stride[1]==r_right,
        // stride[0]==n_phys*r_right.
        if (arr.stride(2) != 1 ||
            arr.stride(1) != static_cast<int64_t>(r_right) ||
            arr.stride(0) != static_cast<int64_t>(n_phys) * r_right)
        {
          throw std::runtime_error(
            "from_cores: core " + std::to_string(k) +
            " is not C-contiguous; "
            "pass np.ascontiguousarray(core) first.");
        }
        tt_core c(r_left, n_phys, r_right);
        std::memcpy(c.data(),
                    arr.data(),
                    sizeof(double) * r_left * n_phys * r_right);
        cores.push_back(std::move(c));
      }
      // Validate boundary ranks and inter-core consistency.
      if (d > 0)
      {
        if (cores[0].r_left() != 1)
        {
          throw std::runtime_error(
            "from_cores: core 0 r_left must be 1, got " +
            std::to_string(cores[0].r_left()));
        }
        if (cores.back().r_right() != 1)
        {
          throw std::runtime_error(
            "from_cores: last core r_right must be 1, got " +
            std::to_string(cores.back().r_right()));
        }
        for (int k = 1; k < d; ++k)
        {
          if (cores[static_cast<std::size_t>(k - 1)].r_right()
              != cores[static_cast<std::size_t>(k)].r_left())
          {
            throw std::runtime_error(
              "from_cores: rank mismatch core[" +
              std::to_string(k - 1) + "].r_right=" +
              std::to_string(cores[static_cast<std::size_t>(k - 1)].r_right()) +
              " != core[" + std::to_string(k) + "].r_left=" +
              std::to_string(cores[static_cast<std::size_t>(k)].r_left()));
          }
        }
      }
      return tt(std::move(cores));
    },
    "cores"_a,
    "Build a TensorTrain from a list of 3D numpy arrays, one per core. "
    "Core k must have shape (r_k, n_k, r_{k+1}) with r_0 = r_d = 1.");

  // ==================================================================
  // Compression: round
  // ==================================================================

  m.def(
    "round",
    [](const tt& a, const round_options& opts) -> tt
    { return mva::tensor_train::round(a, opts); },
    "a"_a,
    "opts"_a);

  m.def(
    "round_ex",
    [](const tt& a, const round_options& opts)
    {
      round_result info;
      tt result = mva::tensor_train::round(a, opts, &info);
      return nb::make_tuple(std::move(result), std::move(info));
    },
    "a"_a,
    "opts"_a);

  m.def(
    "round",
    [](const tt& a, const round_options& opts, nb::object warm) -> tt
    {
      round_options local_opts = opts;
      if (!warm.is_none())
        local_opts.warm_start_tt = nb::inst_ptr<tt>(warm);
      return mva::tensor_train::round(a, local_opts);
    },
    "a"_a,
    "opts"_a,
    "warm"_a.none());

  m.def(
    "round_ex",
    [](const tt& a, const round_options& opts, nb::object warm)
    {
      round_options local_opts = opts;
      if (!warm.is_none())
        local_opts.warm_start_tt = nb::inst_ptr<tt>(warm);
      round_result info;
      tt result = mva::tensor_train::round(a, local_opts, &info);
      return nb::make_tuple(std::move(result), std::move(info));
    },
    "a"_a,
    "opts"_a,
    "warm"_a.none());

  // ==================================================================
  // Algebra
  // ==================================================================

  m.def(
    "add",
    [](const tt& a, const tt& b) -> tt
    { return mva::tensor_train::add(a, b); },
    "a"_a,
    "b"_a);
  m.def(
    "sub",
    [](const tt& a, const tt& b) -> tt
    { return mva::tensor_train::sub(a, b); },
    "a"_a,
    "b"_a);
  m.def(
    "scale",
    [](const tt& a, double alpha) -> tt
    { return mva::tensor_train::scale(a, alpha); },
    "a"_a,
    "alpha"_a);
  m.def(
    "axpy",
    [](double alpha, const tt& a, const tt& b) -> tt
    { return mva::tensor_train::axpy(alpha, a, b); },
    "alpha"_a,
    "a"_a,
    "b"_a);
  m.def(
    "axpby",
    [](double alpha, const tt& a, double beta, const tt& b) -> tt
    { return mva::tensor_train::axpby(alpha, a, beta, b); },
    "alpha"_a,
    "a"_a,
    "beta"_a,
    "b"_a);
  m.def(
    "hadamard",
    [](const tt& a, const tt& b) -> tt
    { return mva::tensor_train::hadamard(a, b); },
    "a"_a,
    "b"_a);

  m.def(
    "neg",
    [](const tt& a) -> tt { return mva::tensor_train::neg(a); },
    "a"_a);

  // ==================================================================
  // Inner products and norms
  // ==================================================================

  m.def(
    "inner",
    [](const tt& a, const tt& b) -> double
    { return mva::tensor_train::inner(a, b); },
    "a"_a,
    "b"_a);
  m.def(
    "norm",
    [](const tt& a) -> double { return mva::tensor_train::norm(a); },
    "a"_a);

  // ==================================================================
  // Element evaluation
  // ==================================================================

  m.def(
    "eval_at",
    [](const tt& a, const std::vector<int>& idx) -> double
    { return mva::tensor_train::eval_at(a, idx.data()); },
    "a"_a,
    "idx"_a);

  m.def(
    "eval_batch",
    [](const tt& a, nb::ndarray<nb::numpy, const int> idx, int M)
    { return to_numpy(mva::tensor_train::eval_batch(a, idx.data(), M)); },
    "a"_a,
    "idx"_a,
    "M"_a);

  // ==================================================================
  // Sampler-based construction (cross interpolation)
  // ==================================================================

  nb::enum_<from_samples_method>(
    m, "from_samples_method", "Cross-interpolation method for from_samples().")
    .value("dmrg_cross", from_samples_method::dmrg_cross)
    .value("tt_cross", from_samples_method::tt_cross)
    .value("als_cross", from_samples_method::als_cross)
    .value("amen_cross", from_samples_method::amen_cross)
    .value("qtt_dmrg_cross", from_samples_method::qtt_dmrg_cross)
    .value("qtt_tt_cross", from_samples_method::qtt_tt_cross)
    .value("qtt_als_cross", from_samples_method::qtt_als_cross)
    .value("qtt_amen_cross", from_samples_method::qtt_amen_cross);

  nb::class_<from_samples_options>(m, "from_samples_options")
    .def(nb::init<>())
    .def_rw("eps", &from_samples_options::eps)
    .def_rw("max_rank", &from_samples_options::max_rank)
    .def_rw("rmin", &from_samples_options::rmin)
    .def_rw("method", &from_samples_options::method)
    .def_rw("max_sweeps", &from_samples_options::max_sweeps)
    .def_rw("init_rank", &from_samples_options::init_rank)
    .def_rw("seed", &from_samples_options::seed)
    .def_rw("maxvol_resolve_interval",
            &from_samples_options::maxvol_resolve_interval)
    .def_rw("enrich_rank", &from_samples_options::enrich_rank)
    .def_rw("use_qr_pivots", &from_samples_options::use_qr_pivots)
    .def_rw("enrich_rounds", &from_samples_options::enrich_rounds)
    .def_rw("enrich_samples", &from_samples_options::enrich_samples)
    .def_rw("enrich_k", &from_samples_options::enrich_k)
    .def_rw("qtt_base", &from_samples_options::qtt_base)
    .def_rw("qtt_maxbits", &from_samples_options::qtt_maxbits)
    .def_rw("qtt_pool_modes", &from_samples_options::qtt_pool_modes)
    .def_rw("qtt_row_maxbits", &from_samples_options::qtt_row_maxbits)
    .def_rw("qtt_col_maxbits", &from_samples_options::qtt_col_maxbits);

  m.def(
    "from_samples",
    [](nb::callable func,
       const std::vector<int>& shape,
       const from_samples_options& opts) -> tt
    {
      int d = static_cast<int>(shape.size());
      if (d == 0)
      {
        d = static_cast<int>(opts.qtt_maxbits.size());
      }
      if (d <= 0)
      {
        throw std::runtime_error(
          "from_samples: cannot determine dimension d -- "
          "both shape and qtt_maxbits are empty");
      }
      return mva::tensor_train::from_samples(
        [&func, d](const int* idx) -> double
        {
          nb::object tup = nb::steal(PyTuple_New(d));
          if (!tup.is_valid())
            throw std::runtime_error(
              "from_samples: PyTuple_New failed for d=" +
              std::to_string(d));
          for (int i = 0; i < d; ++i)
          {
            PyObject* py_i = PyLong_FromLong(idx[i]);
            if (py_i == nullptr)
            {
              tup.release();
              throw std::runtime_error(
                "from_samples: PyLong_FromLong failed for idx["
                + std::to_string(i) + "]=" + std::to_string(idx[i]));
            }
            PyTuple_SET_ITEM(tup.ptr(), i, py_i);
          }
          return nb::cast<double>(func(tup));
        },
        shape,
        opts);
    },
    "func"_a,
    "shape"_a,
    "opts"_a = from_samples_options{});

  // ==================================================================
  // Factory: from_samples for tt_matrix
  // ==================================================================

  m.def(
    "from_samples_matrix",
    [](nb::callable func,
       const std::vector<int>& row_shape,
       const std::vector<int>& col_shape,
       const from_samples_options& opts) -> tt_matrix
    {
      if (opts.method == from_samples_method::qtt_dmrg_cross ||
          opts.method == from_samples_method::qtt_tt_cross ||
          opts.method == from_samples_method::qtt_als_cross ||
          opts.method == from_samples_method::qtt_amen_cross)
      {
        throw std::runtime_error(
          "from_samples(tt_matrix): QTT matrix cross is not yet "
          "implemented; use dmrg_cross, tt_cross, als_cross, "
          "or amen_cross");
      }
      int d = static_cast<int>(row_shape.size());
      if (d == 0)
        d = static_cast<int>(opts.qtt_row_maxbits.size());
      if (d <= 0)
        throw std::runtime_error(
          "from_samples(tt_matrix): cannot determine dimension -- "
          "both row_shape and qtt_row_maxbits are empty");
      return mva::tensor_train::from_samples(
        [&func, d](const int* row_idx, const int* col_idx) -> double
        {
          nb::object tup_row = nb::steal(PyTuple_New(d));
          nb::object tup_col = nb::steal(PyTuple_New(d));
          if (!tup_row.is_valid() || !tup_col.is_valid())
            throw std::runtime_error(
              "from_samples(tt_matrix): PyTuple_New failed for d="
              + std::to_string(d));
          for (int i = 0; i < d; ++i)
          {
            PyObject* py_r = PyLong_FromLong(row_idx[i]);
            PyObject* py_c = PyLong_FromLong(col_idx[i]);
            if (py_r == nullptr || py_c == nullptr)
            {
              if (py_r != nullptr) Py_DECREF(py_r);
              if (py_c != nullptr) Py_DECREF(py_c);
              tup_row.release();
              tup_col.release();
              throw std::runtime_error(
                "from_samples(tt_matrix): PyLong_FromLong failed");
            }
            PyTuple_SET_ITEM(tup_row.ptr(), i, py_r);
            PyTuple_SET_ITEM(tup_col.ptr(), i, py_c);
          }
          return nb::cast<double>(func(tup_row, tup_col));
        },
        row_shape, col_shape, opts);
    },
    "func"_a, "row_shape"_a, "col_shape"_a,
    "opts"_a = from_samples_options{});

  // ==================================================================
  // tt_matrix_core class (4-axis core for MPO)
  // ==================================================================

  nb::class_<tt_matrix_core>(m, "tt_matrix_core")
    .def(nb::init<int, int, int, int>(),
         "r_left"_a, "m_phys"_a, "n_phys"_a, "r_right"_a,
         "Allocate a zero-filled core of shape "
         "(r_left, m_phys, n_phys, r_right).")
    .def_prop_ro("r_left", &tt_matrix_core::r_left)
    .def_prop_ro("m_phys", &tt_matrix_core::m_phys)
    .def_prop_ro("n_phys", &tt_matrix_core::n_phys)
    .def_prop_ro("r_right", &tt_matrix_core::r_right)
    .def_prop_ro("dims", &tt_matrix_core::dims)
    .def_prop_ro("size", &tt_matrix_core::size)
    .def("data_view",
         [](const tt_matrix_core& c) {
           auto dims = c.dims();
           return nb::ndarray<nb::numpy, const double>(
             c.data(),
             { static_cast<size_t>(dims[0]),
               static_cast<size_t>(dims[1]),
               static_cast<size_t>(dims[2]),
               static_cast<size_t>(dims[3]) });
         },
         "Read-only 4D numpy view of shape "
         "(r_left, m_phys, n_phys, r_right) "
         "into the core data (zero-copy).  "
         "The view is valid only as long as this "
         "TTCoreMatrix object exists.")
    .def("fill",
         [](tt_matrix_core& c,
            nb::ndarray<nb::numpy, const double, nb::ndim<4>> src)
         {
           if (static_cast<int>(src.shape(0)) != c.r_left() ||
               static_cast<int>(src.shape(1)) != c.m_phys() ||
               static_cast<int>(src.shape(2)) != c.n_phys() ||
               static_cast<int>(src.shape(3)) != c.r_right())
           {
             throw std::runtime_error(
               "tt_matrix_core.fill: shape mismatch");
           }
           std::memcpy(c.data(),
                       src.data(),
                       sizeof(double) * static_cast<std::size_t>(c.size()));
         },
         "src"_a,
         "Copy data from a 4D numpy array into this core.  "
         "src must have shape (r_left, m_phys, n_phys, r_right).");

  // ==================================================================
  // tt_matrix class (MPO)
  // ==================================================================

  nb::class_<tt_matrix>(m, "tt_matrix")
    .def_prop_ro("d", &tt_matrix::d)
    .def_prop_ro("row_shape", &tt_matrix::row_shape)
    .def_prop_ro("col_shape", &tt_matrix::col_shape)
    .def_prop_ro("ranks", &tt_matrix::ranks)
    .def_prop_ro("max_rank", &tt_matrix::max_rank)
    .def_prop_ro("total_rows", &tt_matrix::total_rows)
    .def_prop_ro("total_cols", &tt_matrix::total_cols)
    .def_prop_ro("num_params", &tt_matrix::num_params)
    .def("to_dense",
         [](const tt_matrix& a) { return to_numpy(a.to_dense()); })
    .def("core",
         [](const tt_matrix& a, int k)
         { return a.core(k); },
         "k"_a,
         "Return a copy of core k.")
    .def("cores",
         [](const tt_matrix& a)
         {
           nb::list out;
           const int d = a.d();
           if (d == 0)
             return out;
           struct shared_ref
           {
             std::shared_ptr<tt_matrix> data;
             int refs;
           };
           auto* ref = new shared_ref{std::make_shared<tt_matrix>(a), d};
           for (int k = 0; k < d; ++k)
           {
             nb::capsule owner(ref,
                               [](void* p) noexcept
                               {
                                 auto* r = static_cast<shared_ref*>(p);
                                 if (--r->refs == 0)
                                   delete r;
                               });
             auto dims = ref->data->core(k).dims();
             out.append(
               nb::ndarray<nb::numpy, const double>(
                 ref->data->core(k).data(),
                 { static_cast<size_t>(dims[0]),
                   static_cast<size_t>(dims[1]),
                   static_cast<size_t>(dims[2]),
                   static_cast<size_t>(dims[3]) },
                 std::move(owner)));
           }
           return out;
         },
         "Return list of read-only 4D numpy views, one per core. "
         "Zero-copy -- views share a single TT-matrix copy kept alive "
         "until all arrays are garbage-collected.");

  // ==================================================================
  // Factory: from_matrix_cores
  // ==================================================================

  m.def(
    "from_matrix_cores",
    [](nb::list core_list) -> tt_matrix
    {
      std::vector<tt_matrix_core> cores;
      const int d = nb::len(core_list);
      cores.reserve(d);
      for (int k = 0; k < d; ++k)
      {
        auto arr =
          nb::cast<nb::ndarray<nb::numpy, const double, nb::ndim<4>>>(
            core_list[k]);
        int r_left  = static_cast<int>(arr.shape(0));
        int m_phys  = static_cast<int>(arr.shape(1));
        int n_phys  = static_cast<int>(arr.shape(2));
        int r_right = static_cast<int>(arr.shape(3));
        // Enforce C-contiguity: stride[3]==1, stride[2]==r_right,
        // stride[1]==n_phys*r_right, stride[0]==m_phys*n_phys*r_right.
        if (arr.stride(3) != 1 ||
            arr.stride(2) != static_cast<int64_t>(r_right) ||
            arr.stride(1) != static_cast<int64_t>(n_phys) * r_right ||
            arr.stride(0) != static_cast<int64_t>(m_phys) * n_phys * r_right)
        {
          throw std::runtime_error(
            "from_matrix_cores: core " + std::to_string(k) +
            " is not C-contiguous; "
            "pass np.ascontiguousarray(core) first.");
        }
        tt_matrix_core c(r_left, m_phys, n_phys, r_right);
        std::memcpy(c.data(),
                    arr.data(),
                    sizeof(double) * static_cast<std::size_t>(c.size()));
        cores.push_back(std::move(c));
      }
      if (d > 0)
      {
        if (cores[0].r_left() != 1)
          throw std::runtime_error(
            "from_matrix_cores: core 0 r_left must be 1, got " +
            std::to_string(cores[0].r_left()));
        if (cores.back().r_right() != 1)
          throw std::runtime_error(
            "from_matrix_cores: last core r_right must be 1, got " +
            std::to_string(cores.back().r_right()));
        for (int k = 1; k < d; ++k)
        {
          if (cores[static_cast<std::size_t>(k - 1)].r_right()
              != cores[static_cast<std::size_t>(k)].r_left())
            throw std::runtime_error(
              "from_matrix_cores: rank mismatch at bond " +
              std::to_string(k));
        }
      }
      return tt_matrix(std::move(cores));
    },
    "cores"_a,
    "Build a TensorTrainMatrix from a list of 4D numpy arrays, "
    "one per core. Core k must have shape (r_k, m_k, n_k, r_{k+1}) "
    "with r_0 = r_d = 1.");

  // ==================================================================
  // Factory: tt_matrix constructors
  // ==================================================================

  m.def(
    "zeros_matrix",
    [](const std::vector<int>& rs, const std::vector<int>& cs) -> tt_matrix
    { return mva::tensor_train::zeros(rs, cs); },
    "row_shape"_a, "col_shape"_a,
    "All-zero rank-1 TT-matrix.");

  m.def(
    "identity_matrix",
    [](const std::vector<int>& shape) -> tt_matrix
    { return mva::tensor_train::identity(shape); },
    "shape"_a,
    "Block-diagonal identity TT-matrix (rank 1).");

  m.def(
    "random_matrix",
    [](const std::vector<int>& rs, const std::vector<int>& cs,
       int max_rank, std::uint64_t seed) -> tt_matrix
    { return mva::tensor_train::random(rs, cs, max_rank, seed); },
    "row_shape"_a, "col_shape"_a, "max_rank"_a, "seed"_a,
    "Random TT-matrix with N(0,1) entries.");

  m.def(
    "diag_from_tt",
    [](const tt& v) -> tt_matrix
    { return mva::tensor_train::diag_from_tt(v); },
    "diag"_a,
    "Diagonal TT-matrix from a TT vector.");

  m.def(
    "matrix_from_dense",
    [](nb::ndarray<nb::numpy, const double> data,
       const std::vector<int>& rs, const std::vector<int>& cs,
       double eps) -> tt_matrix
    {
      from_dense_options opts;
      opts.eps = eps;
      return mva::tensor_train::from_dense(data.data(), rs, cs, opts);
    },
    "data"_a, "row_shape"_a, "col_shape"_a, "eps"_a,
    "Build TT-matrix from dense data via TT-SVD compression.");

  m.def(
    "matrix_from_dense",
    [](nb::ndarray<nb::numpy, const double> data,
       const std::vector<int>& row_shape,
       const std::vector<int>& col_shape,
       const from_dense_options& opts) -> tt_matrix
    {
      if (opts.method == from_dense_method::qtt)
      {
        throw std::runtime_error(
          "matrix_from_dense: QTT compression is not yet "
          "implemented for tt_matrix; use method='svd'.");
      }
      return mva::tensor_train::from_dense(
        data.data(), row_shape, col_shape, opts);
    },
    "data"_a, "row_shape"_a, "col_shape"_a, "opts"_a,
    "Build TT-matrix from dense data via TT-SVD compression. "
    "(QTT method is not yet implemented for tt_matrix.)");

  // ==================================================================
  // TT-matrix algebra
  // ==================================================================

  m.def(
    "add_matrix",
    [](const tt_matrix& a, const tt_matrix& b) -> tt_matrix
    { return mva::tensor_train::add(a, b); },
    "a"_a, "b"_a);
  m.def(
    "sub_matrix",
    [](const tt_matrix& a, const tt_matrix& b) -> tt_matrix
    { return mva::tensor_train::sub(a, b); },
    "a"_a, "b"_a);
  m.def(
    "scale_matrix",
    [](const tt_matrix& a, double alpha) -> tt_matrix
    { return mva::tensor_train::scale(a, alpha); },
    "a"_a, "alpha"_a);
  m.def(
    "axpy_matrix",
    [](double alpha, const tt_matrix& a, const tt_matrix& b) -> tt_matrix
    { return mva::tensor_train::axpy(alpha, a, b); },
    "alpha"_a, "a"_a, "b"_a);
  m.def(
    "axpby_matrix",
    [](double alpha, const tt_matrix& a,
       double beta, const tt_matrix& b) -> tt_matrix
    { return mva::tensor_train::axpby(alpha, a, beta, b); },
    "alpha"_a, "a"_a, "beta"_a, "b"_a);

  m.def(
    "neg_matrix",
    [](const tt_matrix& a) -> tt_matrix
    { return mva::tensor_train::neg(a); },
    "a"_a);

  // ==================================================================
  // TT-matrix inner products
  // ==================================================================

  m.def(
    "frob_inner",
    [](const tt_matrix& a, const tt_matrix& b) -> double
    { return mva::tensor_train::frob_inner(a, b); },
    "a"_a, "b"_a);
  m.def(
    "frob_norm_matrix",
    [](const tt_matrix& a) -> double
    { return mva::tensor_train::frob_norm(a); },
    "a"_a);

  // ==================================================================
  // TT-matrix apply (matvec, matmat without rounding)
  // ==================================================================

  m.def(
    "matvec",
    [](const tt_matrix& a, const tt& x) -> tt
    { return mva::tensor_train::matvec(a, x); },
    "a"_a, "x"_a);
  m.def(
    "matmat",
    [](const tt_matrix& a, const tt_matrix& b) -> tt_matrix
    { return mva::tensor_train::matmat(a, b); },
    "a"_a, "b"_a);

  // ==================================================================
  // TT-matrix compression (round, matvec_round, matmat_round)
  // ==================================================================

  m.def(
    "round_matrix",
    [](const tt_matrix& a, const round_options& opts) -> tt_matrix
    { return mva::tensor_train::round(a, opts); },
    "a"_a, "opts"_a);

  m.def(
    "round_matrix",
    [](const tt_matrix& a, const round_options& opts,
       nb::object warm) -> tt_matrix
    {
      round_options local_opts = opts;
      if (!warm.is_none())
        local_opts.warm_start_tt_matrix = nb::inst_ptr<tt_matrix>(warm);
      return mva::tensor_train::round(a, local_opts);
    },
    "a"_a, "opts"_a, "warm"_a.none());

  m.def(
    "matvec_round",
    [](const tt_matrix& a, const tt& x,
       const round_options& opts) -> tt
    { return mva::tensor_train::matvec_round(a, x, opts); },
    "a"_a, "x"_a, "opts"_a);

  m.def(
    "matvec_round",
    [](const tt_matrix& a, const tt& x,
       const round_options& opts, nb::object warm) -> tt
    {
      round_options local_opts = opts;
      if (!warm.is_none())
        local_opts.warm_start_tt = nb::inst_ptr<tt>(warm);
      return mva::tensor_train::matvec_round(a, x, local_opts);
    },
    "a"_a, "x"_a, "opts"_a, "warm"_a.none());

  m.def(
    "matvec_round_ex",
    [](const tt_matrix& a, const tt& x, const round_options& opts)
    {
      round_result info;
      tt result = mva::tensor_train::matvec_round(a, x, opts, &info);
      return nb::make_tuple(std::move(result), std::move(info));
    },
    "a"_a, "x"_a, "opts"_a);

  m.def(
    "matvec_round_ex",
    [](const tt_matrix& a, const tt& x, const round_options& opts,
       nb::object warm)
    {
      round_options local_opts = opts;
      if (!warm.is_none())
        local_opts.warm_start_tt = nb::inst_ptr<tt>(warm);
      round_result info;
      tt result = mva::tensor_train::matvec_round(a, x, local_opts, &info);
      return nb::make_tuple(std::move(result), std::move(info));
    },
    "a"_a, "x"_a, "opts"_a, "warm"_a.none());

  m.def(
    "matmat_round",
    [](const tt_matrix& a, const tt_matrix& b,
       const round_options& opts) -> tt_matrix
    { return mva::tensor_train::matmat_round(a, b, opts); },
    "a"_a, "b"_a, "opts"_a);

  m.def(
    "matmat_round",
    [](const tt_matrix& a, const tt_matrix& b,
       const round_options& opts, nb::object warm) -> tt_matrix
    {
      round_options local_opts = opts;
      if (!warm.is_none())
        local_opts.warm_start_tt_matrix = nb::inst_ptr<tt_matrix>(warm);
      return mva::tensor_train::matmat_round(a, b, local_opts);
    },
    "a"_a, "b"_a, "opts"_a, "warm"_a.none());

  m.def(
    "matmat_round_ex",
    [](const tt_matrix& a, const tt_matrix& b, const round_options& opts)
    {
      round_result info;
      tt_matrix result = mva::tensor_train::matmat_round(a, b, opts, &info);
      return nb::make_tuple(std::move(result), std::move(info));
    },
    "a"_a, "b"_a, "opts"_a);

  m.def(
    "matmat_round_ex",
    [](const tt_matrix& a, const tt_matrix& b, const round_options& opts,
       nb::object warm)
    {
      round_options local_opts = opts;
      if (!warm.is_none())
        local_opts.warm_start_tt_matrix = nb::inst_ptr<tt_matrix>(warm);
      round_result info;
      tt_matrix result = mva::tensor_train::matmat_round(
        a, b, local_opts, &info);
      return nb::make_tuple(std::move(result), std::move(info));
    },
    "a"_a, "b"_a, "opts"_a, "warm"_a.none());

  // ==================================================================
  // TT-matrix round with diagnostics
  // ==================================================================

  m.def(
    "round_matrix_ex",
    [](const tt_matrix& a, const round_options& opts)
    {
      round_result info;
      tt_matrix result = mva::tensor_train::round(a, opts, &info);
      return nb::make_tuple(std::move(result), std::move(info));
    },
    "a"_a, "opts"_a);

  m.def(
    "round_matrix_ex",
    [](const tt_matrix& a, const round_options& opts,
       nb::object warm)
    {
      round_options local_opts = opts;
      if (!warm.is_none())
        local_opts.warm_start_tt_matrix =
          nb::inst_ptr<tt_matrix>(warm);
      round_result info;
      tt_matrix result = mva::tensor_train::round(a, local_opts, &info);
      return nb::make_tuple(std::move(result), std::move(info));
    },
    "a"_a, "opts"_a, "warm"_a.none());

  // ==================================================================
  // Gauge / orthogonalization
  // ==================================================================

  m.def(
    "right_orthogonalize",
    [](const tt& a) -> tt
    {
      tt copy = a;
      mva::tensor_train::right_orthogonalize(copy);
      return copy;
    },
    "a"_a,
    "Return a copy of a with cores[1..d-1] right-orthogonal.");

  m.def(
    "left_orthogonalize",
    [](const tt& a) -> tt
    {
      tt copy = a;
      mva::tensor_train::detail::left_orthogonalize(copy);
      return copy;
    },
    "a"_a,
    "Return a copy of a with cores[0..d-2] left-orthogonal.");

  m.def(
    "right_orthogonalize_matrix",
    [](const tt_matrix& a) -> tt_matrix
    {
      tt_matrix copy = a;
      mva::tensor_train::right_orthogonalize(copy);
      return copy;
    },
    "a"_a,
    "Return a copy of a with cores[1..d-1] right-orthogonal.");

  // ==================================================================
  // Soft thresholding
  // ==================================================================

  m.def(
    "soft_threshold",
    [](const tt& a, double tau) -> tt
    { return mva::tensor_train::soft_threshold(a, tau); },
    "a"_a, "tau"_a,
    "Singular-value soft-thresholding: new_SV = max(SV - tau, 0).");

  // ==================================================================
  // Frobenius norm of apply (fast path)
  // ==================================================================

  m.def(
    "frob_norm_apply_matvec",
    [](const tt_matrix& a, const tt& x) -> double
    { return mva::tensor_train::frob_norm_apply(a, x); },
    "a"_a, "x"_a,
    "||A * x||_F without materializing the full result.");

  m.def(
    "frob_norm_apply_matmat",
    [](const tt_matrix& a, const tt_matrix& b) -> double
    { return mva::tensor_train::frob_norm_apply(a, b); },
    "a"_a, "b"_a,
    "||A * B||_F without materializing the full result.");

}
//
// :D
//
