cmake_minimum_required(VERSION 3.18)
project(tt_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(tt_core tt_bindings.cpp)

target_include_directories(tt_core PRIVATE
  ${CMAKE_SOURCE_DIR}/../include
  ${CMAKE_SOURCE_DIR}/../libs/ext_eigen
)

# Optional: enable mva::containers::narray storage backend.
#   TENSOR_TRAIN_USE_NARRAY=1 cmake ...
if(DEFINED ENV{TENSOR_TRAIN_USE_NARRAY})
  target_include_directories(tt_core PRIVATE
    ${CMAKE_SOURCE_DIR}/../libs/mva_containers_narray/include)
  target_compile_definitions(tt_core PRIVATE TENSOR_TRAIN_USE_NARRAY)
endif()

target_compile_options(tt_core PRIVATE
  -O2 -Wall -Wextra -pedantic -Werror
  # -Wno-shadow: nanobind/stl headers shadow names (intentional;
  # the main library headers are shadow-free per project conventions).
  -Wno-shadow -Wcast-align -Wdouble-promotion -Wformat=2
)

# -Wno-array-bounds: GCC 13 false-positive in vendored narray.
# Only needed when the narray backend is active.
if(DEFINED ENV{TENSOR_TRAIN_USE_NARRAY})
  target_compile_options(tt_core PRIVATE -Wno-array-bounds)
endif()
