# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#===------------------------------------------------------------------------===#
# Common Sources
#===------------------------------------------------------------------------===#

set(_BITCODE_SRCS
  "dummy.c"
)

set(_BITCODE_HDRS
  "kernel_tables.h"
  "kernels.h"
  "support/common.h"
  "support/kernel_args.h"
)

#===------------------------------------------------------------------------===#
# Exported Headers
#===------------------------------------------------------------------------===#

iree_cc_library(
  NAME
    headers
  HDRS
    "${_BITCODE_HDRS}"
  PUBLIC
)

#===------------------------------------------------------------------------===#
# Architecture-specific Binaries
#===------------------------------------------------------------------------===#
# NOTE: the naming here matches what HSA_ISA_INFO_NAME returns so that we can
# match them at runtime without having to load and reflect each code object.

# TODO(benvanik): when TheRock stabilizes its naming convention we'll want to
# copy that and make it configurable. See:
# https://github.com/ROCm/TheRock/blob/main/cmake/therock_amdgpu_targets.cmake
# Matching their family naming scheme would allow us to directly source from
# their command line arguments.

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
