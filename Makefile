include config.mk

CXXFLAGS = -std=c++20 -O2 -Wall -Wextra -pedantic -Werror \
           -Wshadow -Wcast-align \
           -Wdouble-promotion -Wformat=2 \
           -I./include \
           -I./libs/ext_eigen

# Optional: enable mva::containers::narray storage backend.
#   TENSOR_TRAIN_USE_NARRAY=1 make test
ifdef TENSOR_TRAIN_USE_NARRAY
  CXXFLAGS += -DTENSOR_TRAIN_USE_NARRAY -I./libs/mva_containers_narray/include
endif

BUILD_DIR = build

# Per-module test groups.  Append new module groups (e.g. MATRIX_TESTS)
# here and add their pattern rule + group target below.
CORE_TESTS = test_tt_core test_tt_svd test_tt_round \
             test_tt_eval test_tt_ops test_tt_factory \
             test_tt_orthogonalize \
             test_tt_invariants \
             test_tt_soft_threshold \
             test_tt_matvec_round \
             test_tt_fused_apply_core \
             test_tt_cross \
             test_tt_matrix test_tt_matrix_factory test_tt_matrix_ops \
             test_tt_matrix_round \
             test_tt_matrix_frob_norm_apply \
             test_tt_matrix_apply test_tt_matrix_apply_round \
             test_tt_qtt_from_dense \
             test_tt_cross_direct test_tt_stress test_tt_validation

BENCH_TESTS = bench_tt_matrix_apply bench_tt_round \
              bench_streaming_matvec bench_tt_round_methods

TESTS = $(CORE_TESTS)
TEST_EXECUTABLES = $(addprefix $(BUILD_DIR)/, $(TESTS))
BENCH_EXECUTABLES = $(addprefix $(BUILD_DIR)/, $(BENCH_TESTS))

.PHONY: all clean test test-core run help bench bind clean-bind \
         test-bindings docs doc clean-docs \
         $(addprefix test-, $(TESTS)) \
         $(addprefix run-, $(BENCH_TESTS))

all: $(TEST_EXECUTABLES)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Pattern rules: compile to .o then link.  Two-step so ccache can
# cache the object-file compilation (single-step compile+link is
# rejected by ccache as uncacheable).

$(BUILD_DIR)/test_%.o: tests/core/test_%.cpp | $(BUILD_DIR)
	$(CXX) -c $(CXXFLAGS) -MMD -MP -o $@ $<

$(BUILD_DIR)/test_%: $(BUILD_DIR)/test_%.o
	$(CXX) -o $@ $<

$(BUILD_DIR)/bench_%.o: tests/bench/bench_%.cpp | $(BUILD_DIR)
	$(CXX) -c $(CXXFLAGS) -MMD -MP -I./tests/bench -o $@ $<

$(BUILD_DIR)/bench_%: $(BUILD_DIR)/bench_%.o
	$(CXX) -o $@ $<

# Auto-generated header dependencies (-MMD -MP).  On a fresh tree no .d
# files exist yet; the leading '-' tells make to silently ignore that.
-include $(wildcard $(BUILD_DIR)/*.d)

# ------------------------------------------------------------------
# Aggregate run targets.
# ------------------------------------------------------------------

test-core: $(addprefix $(BUILD_DIR)/, $(CORE_TESTS))
	@echo "=========================================="
	@echo "Running core tests"
	@echo "=========================================="
	@failed=0; \
	for t in $(CORE_TESTS); do \
		echo ""; \
		echo "Running $$t..."; \
		./$(BUILD_DIR)/$$t || failed=$$((failed + 1)); \
	done; \
	echo ""; \
	echo "=========================================="; \
	if [ $$failed -eq 0 ]; then \
		echo "ALL CORE TESTS PASSED!"; \
	else \
		echo "$$failed CORE TEST(S) FAILED!"; \
		exit 1; \
	fi

test: test-core

run: test

# Per-test convenience targets.
test-tt_core:    $(BUILD_DIR)/test_tt_core    ; ./$(BUILD_DIR)/test_tt_core
test-tt_svd:     $(BUILD_DIR)/test_tt_svd     ; ./$(BUILD_DIR)/test_tt_svd
test-tt_round:   $(BUILD_DIR)/test_tt_round   ; ./$(BUILD_DIR)/test_tt_round
test-tt_eval:    $(BUILD_DIR)/test_tt_eval    ; ./$(BUILD_DIR)/test_tt_eval
test-tt_ops:     $(BUILD_DIR)/test_tt_ops     ; ./$(BUILD_DIR)/test_tt_ops
test-tt_factory: $(BUILD_DIR)/test_tt_factory ; ./$(BUILD_DIR)/test_tt_factory
test-tt_orthogonalize: $(BUILD_DIR)/test_tt_orthogonalize ; ./$(BUILD_DIR)/test_tt_orthogonalize
test-tt_invariants: $(BUILD_DIR)/test_tt_invariants ; ./$(BUILD_DIR)/test_tt_invariants
test-tt_soft_threshold: $(BUILD_DIR)/test_tt_soft_threshold ; ./$(BUILD_DIR)/test_tt_soft_threshold
test-tt_matvec_round:  $(BUILD_DIR)/test_tt_matvec_round  ; ./$(BUILD_DIR)/test_tt_matvec_round
test-tt_fused_apply_core: $(BUILD_DIR)/test_tt_fused_apply_core ; ./$(BUILD_DIR)/test_tt_fused_apply_core
test-tt_cross:    $(BUILD_DIR)/test_tt_cross    ; ./$(BUILD_DIR)/test_tt_cross
test-tt_matrix_frob_norm_apply: $(BUILD_DIR)/test_tt_matrix_frob_norm_apply ; ./$(BUILD_DIR)/test_tt_matrix_frob_norm_apply
test-tt_matrix:  $(BUILD_DIR)/test_tt_matrix  ; ./$(BUILD_DIR)/test_tt_matrix
test-tt_matrix_factory: $(BUILD_DIR)/test_tt_matrix_factory ; ./$(BUILD_DIR)/test_tt_matrix_factory
test-tt_matrix_ops: $(BUILD_DIR)/test_tt_matrix_ops ; ./$(BUILD_DIR)/test_tt_matrix_ops
test-tt_matrix_round: $(BUILD_DIR)/test_tt_matrix_round ; ./$(BUILD_DIR)/test_tt_matrix_round
test-tt_matrix_apply: $(BUILD_DIR)/test_tt_matrix_apply ; ./$(BUILD_DIR)/test_tt_matrix_apply
test-tt_matrix_apply_round: $(BUILD_DIR)/test_tt_matrix_apply_round ; ./$(BUILD_DIR)/test_tt_matrix_apply_round
test-tt_qtt_from_dense: $(BUILD_DIR)/test_tt_qtt_from_dense ; ./$(BUILD_DIR)/test_tt_qtt_from_dense
test-tt_cross_direct: $(BUILD_DIR)/test_tt_cross_direct ; ./$(BUILD_DIR)/test_tt_cross_direct
test-tt_stress:   $(BUILD_DIR)/test_tt_stress   ; ./$(BUILD_DIR)/test_tt_stress
test-tt_validation: $(BUILD_DIR)/test_tt_validation ; ./$(BUILD_DIR)/test_tt_validation

# ------------------------------------------------------------------
# Bench targets.  Decoupled from `make test`; bench executables never
# run as part of the correctness suite.  `make bench` builds and runs
# all benches; `make run-<name>` runs a single one.
# ------------------------------------------------------------------

bench: $(BENCH_EXECUTABLES)
	@echo "=========================================="
	@echo "Running benches"
	@echo "=========================================="
	@for b in $(BENCH_TESTS); do \
		echo ""; \
		./$(BUILD_DIR)/$$b; \
	done

run-bench_tt_matrix_apply: $(BUILD_DIR)/bench_tt_matrix_apply ; ./$(BUILD_DIR)/bench_tt_matrix_apply
run-bench_tt_round:       $(BUILD_DIR)/bench_tt_round       ; ./$(BUILD_DIR)/bench_tt_round
run-bench_streaming_matvec: $(BUILD_DIR)/bench_streaming_matvec ; ./$(BUILD_DIR)/bench_streaming_matvec
run-bench_tt_round_methods: $(BUILD_DIR)/bench_tt_round_methods ; ./$(BUILD_DIR)/bench_tt_round_methods

clean:
	rm -rf $(BUILD_DIR)

clean-docs:
	rm -rf docs/_doxygen docs/_build

help:
	@echo "Targets:"
	@echo "  all          build all test executables"
	@echo "  test         run all tests (currently == test-core)"
	@echo "  test-core    run all core tests"
	@echo "  test-<name>  build + run a single test"
	@echo "  run          alias for test"
	@echo "  clean        rm -rf $(BUILD_DIR)"
	@echo "  bind         build Python bindings (nanobind)"
	@echo "  clean-bind   rm -rf $(BIND_BUILD_DIR)"
	@echo "  test-bindings run Python binding tests"
	@echo "  docs         build Sphinx docs (doxygen + sphinx-build)"
	@echo "  doc          alias for docs"
	@echo "  clean-docs   rm -rf docs/_doxygen docs/_build"
	@echo ""
	@echo "  TENSOR_TRAIN_USE_NARRAY=1 make <target>   use narray backend"
	@echo "Core tests:   $(CORE_TESTS)"

# ------------------------------------------------------------------
# Documentation (Sphinx + Breathe + Doxygen)
# ------------------------------------------------------------------

docs:
	doxygen docs/Doxyfile
	$(PYTHON) -m sphinx -b html docs docs/_build/html

doc: docs

# ------------------------------------------------------------------
# Python bindings (nanobind)
# ------------------------------------------------------------------

BIND_BUILD_DIR = bind_py/build
NANOBIND_CMAKE_DIR := $(shell $(PYTHON) -c "import nanobind; print(nanobind.cmake_dir())")

bind:
	cmake -S bind_py -B $(BIND_BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=Release \
		-DPython_EXECUTABLE=$(PYTHON) \
		-Dnanobind_DIR=$(NANOBIND_CMAKE_DIR)
	cmake --build $(BIND_BUILD_DIR)

clean-bind:
	rm -rf $(BIND_BUILD_DIR)

test-bindings:
	@echo "=========================================="
	@echo "Running Python binding tests"
	@echo "=========================================="
	$(PYTHON) tests/bind_py/test_smoke.py
	$(PYTHON) tests/bind_py/test_bindings.py
	$(PYTHON) tests/bind_py/test_roundtrip.py
	$(PYTHON) tests/bind_py/test_tt_matrix_bindings.py
	$(PYTHON) tests/bind_py/test_from_samples.py
