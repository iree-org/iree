# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#-------------------------------------------------------------------------------
# Options affecting third-party libraries that IREE depends on.
#-------------------------------------------------------------------------------

macro(iree_set_benchmark_cmake_options)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
endmacro()

macro(iree_set_cpuinfo_cmake_options)
  set(CPUINFO_BUILD_TOOLS ON CACHE BOOL "" FORCE)

  set(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
  set(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
  set(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "" FORCE)
endmacro()

macro(iree_set_googletest_cmake_options)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endmacro()

macro(iree_set_llvm_cmake_options)
  # When enabling an IREE CPU backend, automatically enable these targets.
  set(IREE_DEFAULT_CPU_LLVM_TARGETS "X86;ARM;AArch64;RISCV"
      CACHE STRING "Initialization value for default LLVM CPU targets.")

  # These defaults are moderately important to us, but the user *can*
  # override them (enabling some of these brings in deps that will conflict,
  # so ymmv).
  set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "")
  set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "")
  set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "")
  set(LLVM_APPEND_VC_REV OFF CACHE BOOL "")
  set(LLVM_ENABLE_IDE ON CACHE BOOL "")
  set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "")

  # LLVM defaults to building all targets. We always enable targets that we need
  # as we need them, so default to none. The user can override this as needed,
  # which is fine.
  set(LLVM_TARGETS_TO_BUILD "" CACHE STRING "")

  # We enable LLVM projects as needed. The user can override this.
  set(LLVM_ENABLE_PROJECTS "" CACHE STRING "")
  set(LLVM_EXTERNAL_PROJECTS "" CACHE STRING "")

  # Default Python bindings to off (for all sub-projects).
  set(MLIR_ENABLE_BINDINGS_PYTHON OFF CACHE BOOL "")
  set(MHLO_ENABLE_BINDINGS_PYTHON OFF CACHE BOOL "")

  # If we are building LLD, this will be the target. Otherwise, empty.
  set(IREE_LLD_TARGET)

  # Unconditionally enable mlir.
  list(APPEND LLVM_ENABLE_PROJECTS mlir)

  # Configure LLVM based on enabled IREE target backends.
  message(STATUS "IREE compiler target backends:")
  if(IREE_TARGET_BACKEND_CUDA)
    message(STATUS "  - cuda")
    list(APPEND LLVM_TARGETS_TO_BUILD NVPTX)
  endif()
  if(IREE_TARGET_BACKEND_LLVM_CPU)
    message(STATUS "  - llvm-cpu")
    list(APPEND LLVM_TARGETS_TO_BUILD "${IREE_DEFAULT_CPU_LLVM_TARGETS}")
    set(IREE_LLD_TARGET lld)
  endif()
  if(IREE_TARGET_BACKEND_LLVM_CPU_WASM)
    message(STATUS "  - llvm-cpu (wasm)")
    list(APPEND LLVM_TARGETS_TO_BUILD WebAssembly)
    set(IREE_LLD_TARGET lld)
  endif()
  if(IREE_TARGET_BACKEND_METAL_SPIRV)
    message(STATUS "  - metal-spirv")
  endif()
  if(IREE_TARGET_BACKEND_OPENCL_SPIRV)
    message(STATUS "  - opencl-spirv")
  endif()
  if(IREE_TARGET_BACKEND_ROCM)
    message(STATUS "  - rocm")
    list(APPEND LLVM_TARGETS_TO_BUILD AMDGPU)
  endif()
  if(IREE_TARGET_BACKEND_VULKAN_SPIRV)
    message(STATUS "  - vulkan-spirv")
  endif()
  if(IREE_TARGET_BACKEND_VMVX)
    message(STATUS "  - vmvx")
  endif()
  if(IREE_TARGET_BACKEND_WEBGPU)
    message(STATUS "  - webgpu")
  endif()

  if(IREE_LLD_TARGET)
    list(APPEND LLVM_ENABLE_PROJECTS lld)
  endif()

  list(REMOVE_DUPLICATES LLVM_ENABLE_PROJECTS)
  list(REMOVE_DUPLICATES LLVM_TARGETS_TO_BUILD)
  message(VERBOSE "Building LLVM Targets: ${LLVM_TARGETS_TO_BUILD}")
  message(VERBOSE "Building LLVM Projects: ${LLVM_ENABLE_PROJECTS}")
endmacro()

macro(iree_add_llvm_external_project name identifier location)
  message(STATUS "Adding LLVM external project ${name} (${identifier}) -> ${location}")
  if(NOT EXISTS "${location}/CMakeLists.txt")
    message(FATAL_ERROR "External project location ${location} is not valid")
  endif()
  list(APPEND LLVM_EXTERNAL_PROJECTS ${name})
  list(REMOVE_DUPLICATES LLVM_EXTERNAL_PROJECTS)
  set(LLVM_EXTERNAL_${identifier}_SOURCE_DIR ${location})
endmacro()

macro(iree_set_spirv_headers_cmake_options)
  set(SPIRV_HEADERS_SKIP_EXAMPLES ON CACHE BOOL "" FORCE)
  set(SPIRV_HEADERS_SKIP_INSTALL ON CACHE BOOL "" FORCE)
endmacro()

macro(iree_set_spirv_cross_cmake_options)
  set(SPIRV_CROSS_ENABLE_MSL ON CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_ENABLE_GLSL ON CACHE BOOL "" FORCE) # Required to enable MSL

  set(SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS OFF CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_CLI OFF CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_SKIP_INSTALL ON CACHE BOOL "" FORCE)

  set(SPIRV_CROSS_ENABLE_HLSL OFF CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_ENABLE_CPP OFF CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_ENABLE_REFLECT OFF CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_ENABLE_C_API OFF CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_ENABLE_UTIL OFF CACHE BOOL "" FORCE)
endmacro()
