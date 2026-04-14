# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#===------------------------------------------------------------------------===#
# Common Sources
#===------------------------------------------------------------------------===#

set(_BITCODE_SRCS
  "blit.c"
  "dispatch.c"
  "tracing.c"
)

set(_ABI_HDRS
  "${CMAKE_CURRENT_SOURCE_DIR}/../abi/command_buffer.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../abi/common.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../abi/kernel_args.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../abi/queue.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../abi/signal.h"
)

set(_BITCODE_HDRS
  "blit.h"
  "dispatch.h"
  "kernel_tables.h"
  "kernels.h"
  "support/common.h"
  "support/kernel.h"
  "support/queue.h"
  "support/signal.h"
  "tracing.h"
  ${_ABI_HDRS}
)

#===------------------------------------------------------------------------===#
# Exported Headers
#===------------------------------------------------------------------------===#

iree_cc_library(
  NAME
    headers
  HDRS
    "${_BITCODE_HDRS}"
  DEPS
    iree::base
    iree::base::internal
    iree::base::threading::thread
    iree::hal::drivers::amdgpu::abi
  PUBLIC
)

iree_cc_library(
  NAME
    blit
  SRCS
    "blit.c"
  DEPS
    ::headers
  PUBLIC
)

iree_cc_library(
  NAME
    dispatch
  SRCS
    "dispatch.c"
  DEPS
    ::headers
  PUBLIC
)

iree_cc_test(
  NAME
    blit_test
  SRCS
    "blit_test.cc"
  DEPS
    ::blit
    ::headers
    iree::testing::gtest
    iree::testing::gtest_main
  LABELS
    "driver=amdgpu"
    "nodocker"
  GROUP
    "iree-hal-drivers-amdgpu-tests"
)

iree_cc_test(
  NAME
    dispatch_test
  SRCS
    "dispatch_test.cc"
  DEPS
    ::dispatch
    ::headers
    iree::testing::gtest
    iree::testing::gtest_main
  LABELS
    "driver=amdgpu"
    "nodocker"
  GROUP
    "iree-hal-drivers-amdgpu-tests"
)

#===------------------------------------------------------------------------===#
# Architecture-specific Binaries
#===------------------------------------------------------------------------===#
# NOTE: the naming here matches what HSA_ISA_INFO_NAME returns so that we can
# match them at runtime without having to load and reflect each code object.

# The architecture list is currently hand-picked for this backend. If we align
# the naming scheme with TheRock's target families, this can become a
# user-facing CMake configuration surface fed by those command-line arguments:
# https://github.com/ROCm/TheRock/blob/main/cmake/therock_amdgpu_targets.cmake

set(IREE_HAL_AMDGPU_DEVICE_LIBRARY_TARGETS
    "gfx942;gfx1100"
    CACHE STRING
    "Bundled device library architectures included in the runtime binary.")

set(_ARCH_BINARIES)
foreach(_ARCH ${IREE_HAL_AMDGPU_DEVICE_LIBRARY_TARGETS})
  iree_amdgpu_binary(
    NAME
      amdgcn-amd-amdhsa--${_ARCH}
    TARGET
      amdgcn-amd-amdhsa
    ARCH
      ${_ARCH}
    SRCS
      "${_BITCODE_SRCS}"
    INTERNAL_HDRS
      "${_BITCODE_HDRS}"
  )
  list(APPEND _ARCH_BINARIES "amdgcn-amd-amdhsa--${_ARCH}.so")
endforeach()

#===------------------------------------------------------------------------===#
# Embedded Binary Table
#===------------------------------------------------------------------------===#

iree_c_embed_data(
  NAME
    binaries
  SRCS
    "${_ARCH_BINARIES}"
  C_FILE_OUTPUT
    "binaries.c"
  H_FILE_OUTPUT
    "binaries.h"
  IDENTIFIER
    "iree_hal_amdgpu_device_binaries"
  FLATTEN
  PUBLIC
)
