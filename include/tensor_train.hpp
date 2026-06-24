/*!
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Umbrella header for the mva::tensor_train library
*/
#pragma once

#include "core/types/tt.hpp"
#include "core/types/tt_core.hpp"
#include "core/types/tt_eigen_bridge.hpp"
#include "core/types/tt_matrix.hpp"

// Factories / construction
#include "core/factory/from_dense.hpp"
#include "core/factory/from_samples.hpp"
#include "core/factory/tt_factory.hpp"
#include "core/factory/tt_matrix_factory.hpp"
#include "core/factory/tt_svd.hpp"

// Algebra (linear ops, inner products, evaluation)
#include "core/algebra/tt_eval.hpp"
#include "core/algebra/tt_inner.hpp"
#include "core/algebra/tt_matrix_frob_norm_apply.hpp"
#include "core/algebra/tt_matrix_inner.hpp"
#include "core/algebra/tt_matrix_ops.hpp"
#include "core/algebra/tt_ops.hpp"

// Gauge / orthogonalization
#include "core/gauge/tt_matrix_orthogonalize.hpp"
#include "core/gauge/tt_orthogonalize.hpp"

// Apply (matvec, matmat without rounding)
#include "core/apply/tt_matrix_apply.hpp"

// Compression engines (kept in detail::; surfaced via the dispatchers below)
#include "core/round/tt_als.hpp"
#include "core/round/tt_dmrg.hpp"
#include "core/round/tt_matmat_als_round.hpp"
#include "core/round/tt_matmat_dmrg_round.hpp"
#include "core/round/tt_matrix_als_round.hpp"
#include "core/round/tt_matrix_apply_round.hpp"
#include "core/round/tt_matrix_dmrg_round.hpp"
#include "core/round/tt_matrix_round.hpp"
#include "core/round/tt_matvec_als_round.hpp"
#include "core/round/tt_matvec_dmrg_round.hpp"
#include "core/round/tt_round.hpp"

// Public unified round / matvec_round / matmat_round (round_options-driven).
#include "core/round/matmat_round.hpp"
#include "core/round/matvec_round.hpp"
#include "core/round/round.hpp"
#include "core/round/tt_soft_threshold.hpp"
//
// :D
//
