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

#-------------------------------------------------------------------------------
# Abseil configuration
#-------------------------------------------------------------------------------

include(AbseilConfigureCopts)

# By default Abseil strips string literals on mobile platforms, which means
# we cannot run IREE binaries via command-line with proper options. Turn off
# the stripping.
# TODO: we might still want to strip when compiling IREE into Android Java apps.
if(ANDROID)
  add_definitions(-DABSL_FLAGS_STRIP_NAMES=0)
endif()

#-------------------------------------------------------------------------------
# C++ used within IREE
#-------------------------------------------------------------------------------

set(IREE_CXX_STANDARD ${CMAKE_CXX_STANDARD})

set(IREE_ROOT_DIR ${PROJECT_SOURCE_DIR})
list(APPEND IREE_COMMON_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_BINARY_DIR}
)

if(${IREE_ENABLE_RUNTIME_TRACING})
  set (CMAKE_EXE_LINKER_FLAGS -ldl)
endif()

iree_select_compiler_opts(IREE_DEFAULT_COPTS
  CLANG
    # LINT.IfChange(clang_diagnostics)
    # Set clang diagnostics. These largely match the set of warnings used within
    # Google. They have not been audited super carefully by the IREE team but
    # are generally thought to be a good set and consistency with those used
    # internally is very useful when importing. If you feel hat some of these
    # should be different, please raise an issue!
    "-Wall"

    # Disable warnings we don't care about or that generally have a low
    # signal/noise ratio.
    "-Wno-ambiguous-member-template"
    "-Wno-char-subscripts"
    "-Wno-error=deprecated-declarations"
    "-Wno-extern-c-compat" # Matches upstream. Cannot impact due to extern C inclusion method.
    "-Wno-gnu-alignof-expression"
    "-Wno-gnu-variable-sized-type-not-at-end"
    "-Wno-ignored-optimization-argument"
    "-Wno-invalid-offsetof" # Technically UB but needed for intrusive ptrs
    "-Wno-invalid-source-encoding"
    "-Wno-mismatched-tags"
    "-Wno-pointer-sign"
    "-Wno-reserved-user-defined-literal"
    "-Wno-return-type-c-linkage"
    "-Wno-self-assign-overloaded"
    "-Wno-sign-compare"
    "-Wno-signed-unsigned-wchar"
    "-Wno-strict-overflow"
    "-Wno-trigraphs"
    "-Wno-unknown-pragmas"
    "-Wno-unknown-warning-option"
    "-Wno-unused-command-line-argument"
    "-Wno-unused-const-variable"
    "-Wno-unused-function"
    "-Wno-unused-local-typedef"
    "-Wno-unused-private-field"
    "-Wno-user-defined-warnings"
    "-Wno-macro-redefined" # TODO(GH-2556): Re-enable (IREE and TF both define LOG)
    # Explicitly enable some additional warnings.
    # Some of these aren't on by default, or under -Wall, or are subsets of
    # warnings turned off above.
    "-Wno-ambiguous-member-template"
    "-Wctad-maybe-unsupported"
    "-Wfloat-overflow-conversion"
    "-Wfloat-zero-conversion"
    "-Wfor-loop-analysis"
    "-Wformat-security"
    "-Wgnu-redeclared-enum"
    "-Wimplicit-fallthrough"
    "-Winfinite-recursion"
    "-Wliteral-conversion"
    "-Wnon-virtual-dtor"
    "-Woverloaded-virtual"
    "-Wself-assign"
    "-Wstring-conversion"
    "-Wtautological-overlap-compare"
    "-Wthread-safety"
    "-Wthread-safety-beta"
    "-Wunused-comparison"
    "-Wunused-variable"
    "-Wvla"
    # LINT.ThenChange(https://github.com/google/iree/tree/main/.bazelrc:clang_diagnostics)

    # Turn off some additional warnings (CMake only)
    "-Wno-strict-prototypes"
    "-Wno-shadow-uncaptured-local"
    "-Wno-gnu-zero-variadic-macro-arguments"
    "-Wno-shadow-field-in-constructor"
    "-Wno-unreachable-code-return"
    "-Wno-missing-variable-declarations"
    "-Wno-gnu-label-as-value"
  CLANG_OR_GCC
    "-Wno-unused-parameter"
    "-Wno-undef"
  MSVC_OR_CLANG_CL
    "/DWIN32_LEAN_AND_MEAN"
    "/wd4624"
    # 'inline': used more than once
    "/wd4141"
    # 'WIN32_LEAN_AND_MEAN': macro redefinition
    "/wd4005"
    # TODO(benvanik): figure out if really required or accidentally enabled.
    "/EHsc"
    "/bigobj"
)
set(IREE_DEFAULT_LINKOPTS "${ABSL_DEFAULT_LINKOPTS}")
set(IREE_TEST_COPTS "${ABSL_TEST_COPTS}")

#-------------------------------------------------------------------------------
# Size-optimized build flags
#-------------------------------------------------------------------------------

  # TODO(#898): add a dedicated size-constrained configuration.
if(${IREE_SIZE_OPTIMIZED})
  iree_select_compiler_opts(IREE_SIZE_OPTIMIZED_DEFAULT_COPTS
    MSVC_OR_CLANG_CL
      "/GS-"
      "/GL"
      "/Gw"
      "/Gy"
      "/DNDEBUG"
      "/DIREE_STATUS_MODE=0"
  )
  iree_select_compiler_opts(IREE_SIZE_OPTIMIZED_DEFAULT_LINKOPTS
    MSVC_OR_CLANG_CL
      "/LTCG"
      "/opt:ref,icf"
  )
  # TODO(#898): make this only impact the runtime (IREE_RUNTIME_DEFAULT_...).
  set(IREE_DEFAULT_COPTS
      "${IREE_DEFAULT_COPTS}"
      "${IREE_SIZE_OPTIMIZED_DEFAULT_COPTS}")
  set(IREE_DEFAULT_LINKOPTS
      "${IREE_DEFAULT_LINKOPTS}"
      "${IREE_SIZE_OPTIMIZED_DEFAULT_LINKOPTS}")
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
set(FLATBUFFERS_BUILD_FLATHASH OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_GRPCTEST OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_INSTALL OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_INCLUDE_DIRS
  "${PROJECT_SOURCE_DIR}/third_party/flatbuffers/include/"
)

if(CMAKE_CROSSCOMPILING)
  set(FLATBUFFERS_BUILD_FLATC OFF CACHE BOOL "" FORCE)
else()
  set(FLATBUFFERS_BUILD_FLATC ON CACHE BOOL "" FORCE)
endif()

iree_select_compiler_opts(FLATBUFFERS_COPTS
  CLANG
    # Flatbuffers has a bunch of incorrect documentation annotations.
    "-Wno-documentation"
    "-Wno-documentation-unknown-command"
)
list(APPEND IREE_DEFAULT_COPTS ${FLATBUFFERS_COPTS})

#-------------------------------------------------------------------------------
# Third party: flatcc
#-------------------------------------------------------------------------------

set(FLATCC_TEST OFF CACHE BOOL "" FORCE)
set(FLATCC_CXX_TEST OFF CACHE BOOL "" FORCE)
set(FLATCC_REFLECTION OFF CACHE BOOL "" FORCE)
set(FLATCC_ALLOW_WERROR OFF CACHE BOOL "" FORCE)

if(CMAKE_CROSSCOMPILING)
  set(FLATCC_RTONLY ON CACHE BOOL "" FORCE)
else()
  set(FLATCC_RTONLY OFF CACHE BOOL "" FORCE)
endif()

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

# TODO(ataei): Use optional build time targets selection for LLVMAOT.
set(LLVM_TARGETS_TO_BUILD "WebAssembly;X86;ARM;AArch64" CACHE STRING "" FORCE)

set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "" FORCE)
set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "" FORCE)

if(IREE_USE_LINKER)
  set(LLVM_USE_LINKER ${IREE_USE_LINKER} CACHE STRING "" FORCE)
endif()

# TODO: This should go in add_iree_mlir_src_dep at the top level.
if(IREE_MLIR_DEP_MODE STREQUAL "BUNDLED")
  list(APPEND IREE_COMMON_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/third_party/llvm-project/llvm/include
    ${PROJECT_BINARY_DIR}/third_party/llvm-project/llvm/include
    ${PROJECT_SOURCE_DIR}/third_party/llvm-project/mlir/include
    ${PROJECT_BINARY_DIR}/third_party/llvm-project/llvm/tools/mlir/include
  )
endif()

set(MLIR_TABLEGEN_EXE mlir-tblgen)
# iree-tblgen is not defined using the add_tablegen mechanism as other TableGen
# tools in LLVM.
iree_get_executable_path(IREE_TABLEGEN_EXE iree-tblgen)

#-------------------------------------------------------------------------------
# Third party: tensorflow
#-------------------------------------------------------------------------------

list(APPEND IREE_COMMON_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}/third_party/tensorflow
  ${PROJECT_SOURCE_DIR}/third_party/tensorflow/tensorflow/compiler/mlir/hlo/include/
  ${PROJECT_BINARY_DIR}/build_tools/third_party/tensorflow
  ${PROJECT_BINARY_DIR}/build_tools/third_party/tensorflow/tensorflow/compiler/mlir/hlo/include/
  ${PROJECT_BINARY_DIR}/build_tools/third_party/tensorflow/tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/IR/
  ${PROJECT_BINARY_DIR}/build_tools/third_party/tensorflow/tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms
)

#-------------------------------------------------------------------------------
# Third party: mlir-emitc
#-------------------------------------------------------------------------------

if(IREE_ENABLE_EMITC)
  list(APPEND IREE_COMMON_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/third_party/mlir-emitc/include
    ${PROJECT_BINARY_DIR}/third_party/mlir-emitc/include
  )
  add_definitions(-DIREE_HAVE_EMITC_DIALECT)
endif()
