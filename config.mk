# Compiler: prefer $CXX from spack env, fall back to ccache+g++ via PATH,
# then hardcoded system ccache.
ifneq ($(CXX),)
  # CXX already set in environment (e.g. by activate_spack)
else ifneq ($(shell which ccache 2>/dev/null),)
  CXX = ccache g++
else
  CXX = /usr/bin/ccache g++
endif

# Python for bindings: prefer $CONDA_PREFIX from conda activate,
# fall back to PATH.
ifdef CONDA_PREFIX
  PYTHON = $(CONDA_PREFIX)/bin/python3
else
  PYTHON := $(shell which python3 2>/dev/null)
endif
