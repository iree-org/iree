# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(AbseilConfigureCopts)

#-------------------------------------------------------------------------------
# C++ used within IREE
#-------------------------------------------------------------------------------

set(IREE_CXX_STANDARD 14)

set(IREE_ROOT_DIR ${PROJECT_SOURCE_DIR})
list(APPEND IREE_COMMON_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_BINARY_DIR}
)

iree_select_compiler_opts(IREE_DEFAULT_COPTS
  CLANG
    "-Wno-strict-prototypes"
    "-Wno-shadow-uncaptured-local"
    "-Wno-gnu-zero-variadic-macro-arguments"
    "-Wno-shadow-field-in-constructor"
    "-Wno-unreachable-code-return"
    "-Wno-unused-private-field"
    "-Wno-missing-variable-declarations"
    "-Wno-gnu-label-as-value"
    "-Wno-unused-local-typedef"
    "-Wno-gnu-zero-variadic-macro-arguments"
  CLANG_OR_GCC
    "-Wno-unused-parameter"
    "-Wno-undef"
  MSVC_OR_CLANG_CL
    "/DWIN32_LEAN_AND_MEAN"
    "/wd4624"
    # TODO(benvanik): figure out if really required or accidentally enabled.
    "/EHsc"
    "/bigobj"
)
set(IREE_DEFAULT_LINKOPTS "${ABSL_DEFAULT_LINKOPTS}")
set(IREE_TEST_COPTS "${ABSL_TEST_COPTS}")

if(${IREE_ENABLE_TRACING})
  list(APPEND IREE_DEFAULT_COPTS
    "-DGLOBAL_WTF_ENABLE=1"
  )
endif()

#-------------------------------------------------------------------------------
# Compiler: Clang/LLVM
#-------------------------------------------------------------------------------

# TODO(benvanik): Clang/LLVM options.

#-------------------------------------------------------------------------------
# Compiler: GCC
#-------------------------------------------------------------------------------

# TODO(benvanik): GCC options.

#-------------------------------------------------------------------------------
# Compiler: MSVC
#-------------------------------------------------------------------------------

# TODO(benvanik): MSVC options.

#-------------------------------------------------------------------------------
# Third party: benchmark
#-------------------------------------------------------------------------------

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

#-------------------------------------------------------------------------------
# Third party: flatbuffers
#-------------------------------------------------------------------------------

set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_INSTALL OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATC ON CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATHASH OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_GRPCTEST OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_INCLUDE_DIRS
  "${PROJECT_SOURCE_DIR}/third_party/flatbuffers/include/"
)
iree_select_compiler_opts(FLATBUFFERS_COPTS
  CLANG
    # Flatbuffers has a bunch of incorrect documentation annotations.
    "-Wno-documentation"
    "-Wno-documentation-unknown-command"
)
list(APPEND IREE_DEFAULT_COPTS ${FLATBUFFERS_COPTS})

#-------------------------------------------------------------------------------
# Third party: glslang
#-------------------------------------------------------------------------------

set(ENABLE_CTEST OFF CACHE BOOL "" FORCE)

#-------------------------------------------------------------------------------
# Third party: gtest
#-------------------------------------------------------------------------------

set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

#-------------------------------------------------------------------------------
# Third party: llvm/mlir
#-------------------------------------------------------------------------------

set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(LLVM_APPEND_VC_REV OFF CACHE BOOL "" FORCE)
set(LLVM_ENABLE_IDE ON CACHE BOOL "" FORCE)
set(LLVM_ENABLE_RTTI ON CACHE BOOL "" FORCE)

set(LLVM_TARGETS_TO_BUILD "WebAssembly;X86" CACHE STRING "" FORCE)

set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "" FORCE)
set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "" FORCE)

list(APPEND IREE_COMMON_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}/third_party/llvm-project/llvm/include
  ${PROJECT_BINARY_DIR}/third_party/llvm-project/llvm/include
  ${PROJECT_SOURCE_DIR}/third_party/llvm-project/mlir/include
  ${PROJECT_BINARY_DIR}/third_party/llvm-project/llvm/tools/mlir/include
)

set(MLIR_TABLEGEN_EXE mlir-tblgen)
set(IREE_TABLEGEN_EXE iree-tblgen)

#-------------------------------------------------------------------------------
# Third party: tensorflow
#-------------------------------------------------------------------------------

list(APPEND IREE_COMMON_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}/third_party/tensorflow
  ${PROJECT_BINARY_DIR}/build_tools/third_party/tensorflow
)

#-------------------------------------------------------------------------------
# Third party: tracing
#-------------------------------------------------------------------------------

list(APPEND IREE_COMMON_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}/third_party/google_tracing_framework/bindings/cpp/include
)
