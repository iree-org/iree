# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Builtin AMDGPU device libraries embedded into the runtime.

set(IREE_HAL_AMDGPU_DEVICE_LIBRARY_TARGETS
    "all"
    CACHE STRING
    "AMDGPU device library targets, LLVM generic ISA targets, or TheRock-style target families to embed.")

include("${CMAKE_CURRENT_LIST_DIR}/target_map.cmake")

iree_cc_library(
  NAME
    target_map
  TEXTUAL_HDRS
    "target_map.inl"
  PUBLIC
)

function(_iree_hal_amdgpu_device_target_family_var out_var family)
  string(MAKE_C_IDENTIFIER "${family}" _family_identifier)
  set(${out_var} "_IREE_HAL_AMDGPU_DEVICE_TARGET_FAMILY_${_family_identifier}" PARENT_SCOPE)
endfunction()

function(_iree_hal_amdgpu_device_target_code_object out_var target)
  set(${out_var} "${_IREE_HAL_AMDGPU_DEVICE_TARGET_CODE_OBJECT_${target}}" PARENT_SCOPE)
endfunction()

function(_iree_hal_amdgpu_expand_device_library_targets out_targets)
  set(_expanded_targets)
  foreach(_selection ${IREE_HAL_AMDGPU_DEVICE_LIBRARY_TARGETS})
    if("${_selection}" IN_LIST _IREE_HAL_AMDGPU_DEVICE_CODE_OBJECT_TARGETS)
      list(APPEND _expanded_targets "${_selection}")
    elseif("${_selection}" IN_LIST _IREE_HAL_AMDGPU_DEVICE_TARGETS)
      _iree_hal_amdgpu_device_target_code_object(_code_object_target "${_selection}")
      list(APPEND _expanded_targets "${_code_object_target}")
    elseif("${_selection}" IN_LIST _IREE_HAL_AMDGPU_DEVICE_TARGET_FAMILIES)
      _iree_hal_amdgpu_device_target_family_var(_family_var "${_selection}")
      foreach(_exact_target ${${_family_var}})
        _iree_hal_amdgpu_device_target_code_object(_code_object_target "${_exact_target}")
        list(APPEND _expanded_targets "${_code_object_target}")
      endforeach()
    else()
      set(_available_selections
        ${_IREE_HAL_AMDGPU_DEVICE_TARGETS}
        ${_IREE_HAL_AMDGPU_DEVICE_CODE_OBJECT_TARGETS}
        ${_IREE_HAL_AMDGPU_DEVICE_TARGET_FAMILIES}
      )
      list(REMOVE_DUPLICATES _available_selections)
      list(SORT _available_selections)
      string(JOIN " " _available_pretty ${_available_selections})
      message(FATAL_ERROR
        "Unknown AMDGPU device library target or family '${_selection}'. "
        "Available: ${_available_pretty}"
      )
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _expanded_targets)
  set(${out_targets} "${_expanded_targets}" PARENT_SCOPE)
endfunction()

set(_BITCODE_SRCS
  "../blit.c"
  "../dispatch.c"
  "../tracing.c"
)

set(_ABI_HDRS
  "${CMAKE_CURRENT_SOURCE_DIR}/../../abi/command_buffer.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../abi/common.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../abi/kernel_args.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../abi/queue.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../abi/signal.h"
)

set(_BITCODE_HDRS
  "${CMAKE_CURRENT_SOURCE_DIR}/../blit.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../dispatch.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../kernel_tables.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../kernels.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../support/common.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../support/kernel.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../support/queue.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../support/signal.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../tracing.h"
  ${_ABI_HDRS}
)

_iree_hal_amdgpu_expand_device_library_targets(_DEVICE_LIBRARY_TARGETS)

set(_ARCH_BINARIES)
foreach(_ARCH ${_DEVICE_LIBRARY_TARGETS})
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

iree_c_embed_data(
  NAME
    toc
  SRCS
    "${_ARCH_BINARIES}"
  C_FILE_OUTPUT
    "toc.c"
  H_FILE_OUTPUT
    "toc.h"
  IDENTIFIER
    "iree_hal_amdgpu_device_binaries"
  FLATTEN
  PUBLIC
)
