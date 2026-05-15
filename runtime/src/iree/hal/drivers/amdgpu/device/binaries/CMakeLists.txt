# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Builtin AMDGPU device binaries embedded into the runtime.

set(IREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE
    "prebuilt"
    CACHE STRING
    "AMDGPU device binary producer: prebuilt or source.")
set_property(CACHE IREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE
             PROPERTY STRINGS prebuilt source)

set(IREE_HAL_AMDGPU_DEVICE_BINARY_TARGETS
    "gfx9-generic;gfx90a;gfx9-4-generic;gfx10-1-generic;gfx10-3-generic;gfx11-generic;gfx12-generic"
    CACHE STRING
    "AMDGPU device binary targets, LLVM generic ISA targets, or TheRock-style target families to embed.")

set(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN
    "auto"
    CACHE STRING
    "AMDGPU source-build toolchain: auto, rocm, llvm-tools, or llvm-project.")
set_property(CACHE IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN
             PROPERTY STRINGS auto rocm llvm-tools llvm-project)
set(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_ROCM_PATH "" CACHE PATH
    "ROCm/TheRock root used for AMDGPU device binary source builds.")
set(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLVM_TOOLS_DIR "" CACHE PATH
    "Directory containing clang, llvm-link, lld, and llvm-objcopy for AMDGPU device binary source builds.")
set(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_CLANG_BINARY "" CACHE FILEPATH
    "clang/amdclang executable for AMDGPU device binary source builds.")
set(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLVM_LINK_BINARY "" CACHE FILEPATH
    "llvm-link executable for AMDGPU device binary source builds.")
set(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLD_BINARY "" CACHE FILEPATH
    "lld executable for AMDGPU device binary source builds.")
set(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLVM_OBJCOPY_BINARY "" CACHE FILEPATH
    "llvm-objcopy executable for AMDGPU device binary source builds.")
set(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_CLANG_RESOURCE_INCLUDE "" CACHE PATH
    "Clang resource include directory for AMDGPU device binary source builds.")

include("${CMAKE_CURRENT_LIST_DIR}/target_map.cmake")

if(NOT IREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE STREQUAL "prebuilt" AND
   NOT IREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE STREQUAL "source")
  message(FATAL_ERROR
    "Unknown IREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE='${IREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE}'. "
    "Expected 'prebuilt' or 'source'.")
endif()

function(_iree_hal_amdgpu_device_target_family_var out_var family)
  string(MAKE_C_IDENTIFIER "${family}" _family_identifier)
  set(${out_var} "_IREE_HAL_AMDGPU_DEVICE_TARGET_FAMILY_${_family_identifier}" PARENT_SCOPE)
endfunction()

function(_iree_hal_amdgpu_device_target_code_object out_var target)
  set(${out_var} "${_IREE_HAL_AMDGPU_DEVICE_TARGET_CODE_OBJECT_${target}}" PARENT_SCOPE)
endfunction()

function(_iree_hal_amdgpu_expand_device_binary_targets out_targets)
  set(_expanded_targets)
  foreach(_selection ${IREE_HAL_AMDGPU_DEVICE_BINARY_TARGETS})
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
        "Unknown AMDGPU device binary target or family '${_selection}'. "
        "Available: ${_available_pretty}"
      )
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _expanded_targets)
  set(${out_targets} "${_expanded_targets}" PARENT_SCOPE)
endfunction()

function(_iree_hal_amdgpu_read_device_bitcode_sources out_var)
  set(_source_list_path "${CMAKE_CURRENT_SOURCE_DIR}/../device_bitcode_sources.bzl")
  set_property(DIRECTORY APPEND PROPERTY
    CMAKE_CONFIGURE_DEPENDS "${_source_list_path}")
  file(READ "${_source_list_path}" _source_list)
  string(REGEX MATCH
    "IREE_HAL_AMDGPU_DEVICE_BITCODE_SRCS[ \t\r\n]*=[ \t\r\n]*\\[([^]]*)\\]"
    _source_match "${_source_list}"
  )
  if(NOT _source_match)
    message(FATAL_ERROR
      "${_source_list_path} must assign IREE_HAL_AMDGPU_DEVICE_BITCODE_SRCS "
      "to a literal list")
  endif()

  set(_source_body "${CMAKE_MATCH_1}")
  string(REGEX REPLACE "#[^\r\n]*" "" _source_body "${_source_body}")
  string(REGEX MATCHALL "\"[^\"]+\"" _quoted_sources "${_source_body}")
  if(NOT _quoted_sources)
    message(FATAL_ERROR
      "${_source_list_path} must list at least one C source file")
  endif()

  set(_source_remainder "${_source_body}")
  set(_source_paths)
  foreach(_quoted_source ${_quoted_sources})
    string(REPLACE "${_quoted_source}" "" _source_remainder "${_source_remainder}")
    string(REGEX REPLACE "^\"(.*)\"$" "\\1" _source_name "${_quoted_source}")
    if(_source_name MATCHES "(^/|(^|/)\\.\\.(/|$))")
      message(FATAL_ERROR
        "${_source_list_path} entries must be package-relative: ${_source_name}")
    endif()
    if(NOT _source_name MATCHES "\\.c$")
      message(FATAL_ERROR
        "${_source_list_path} entry is not a C source file: ${_source_name}")
    endif()
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../${_source_name}")
      message(FATAL_ERROR
        "${_source_list_path} entry does not exist: ${_source_name}")
    endif()
    list(APPEND _source_paths "../${_source_name}")
  endforeach()
  string(REGEX REPLACE "[, \t\r\n]" "" _source_remainder "${_source_remainder}")
  if(_source_remainder)
    message(FATAL_ERROR
      "${_source_list_path} must contain only quoted source file names")
  endif()
  set(${out_var} "${_source_paths}" PARENT_SCOPE)
endfunction()

function(_iree_hal_amdgpu_append_generator_flag out_args flag value)
  if(NOT "${value}" STREQUAL "")
    set(${out_args} ${${out_args}} "${flag}" "${value}" PARENT_SCOPE)
  endif()
endfunction()

function(_iree_hal_amdgpu_source_generator_args out_args targets_arg)
  set(_args
    "${Python3_EXECUTABLE}"
    "${IREE_SOURCE_DIR}/build_tools/scripts/amdgpu_device_binaries.py"
    "--repo-root"
    "${IREE_SOURCE_DIR}"
    "--binary-root"
    "${IREE_BINARY_DIR}"
    "--output-dir"
    "${CMAKE_CURRENT_BINARY_DIR}"
    "--targets"
    "${targets_arg}"
  )

  _iree_hal_amdgpu_append_generator_flag(
    _args "--rocm-path" "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_ROCM_PATH}")
  _iree_hal_amdgpu_append_generator_flag(
    _args "--tool-dir" "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLVM_TOOLS_DIR}")
  _iree_hal_amdgpu_append_generator_flag(
    _args "--tool-dir" "${IREE_HOST_BIN_DIR}")
  _iree_hal_amdgpu_append_generator_flag(
    _args "--clang" "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_CLANG_BINARY}")
  _iree_hal_amdgpu_append_generator_flag(
    _args "--llvm-link" "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLVM_LINK_BINARY}")
  _iree_hal_amdgpu_append_generator_flag(
    _args "--lld" "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLD_BINARY}")
  _iree_hal_amdgpu_append_generator_flag(
    _args "--llvm-objcopy" "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLVM_OBJCOPY_BINARY}")
  _iree_hal_amdgpu_append_generator_flag(
    _args "--clang-resource-include"
    "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_CLANG_RESOURCE_INCLUDE}")

  if(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN STREQUAL "llvm-project")
    foreach(_var
        IREE_CLANG_BINARY
        IREE_LLVM_LINK_BINARY
        IREE_LLD_BINARY
        IREE_LLVM_OBJCOPY_BINARY
        IREE_CLANG_BUILTIN_HEADERS_PATH)
      if("${${_var}}" STREQUAL "")
        message(FATAL_ERROR
          "IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=llvm-project requires ${_var} to be "
          "configured by the containing build.")
      endif()
    endforeach()
    if("${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_CLANG_BINARY}" STREQUAL "")
      list(APPEND _args "--clang" "${IREE_CLANG_BINARY}")
    endif()
    if("${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLVM_LINK_BINARY}" STREQUAL "")
      list(APPEND _args "--llvm-link" "${IREE_LLVM_LINK_BINARY}")
    endif()
    if("${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLD_BINARY}" STREQUAL "")
      list(APPEND _args "--lld" "${IREE_LLD_BINARY}")
    endif()
    if("${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLVM_OBJCOPY_BINARY}" STREQUAL "")
      list(APPEND _args "--llvm-objcopy" "${IREE_LLVM_OBJCOPY_BINARY}")
    endif()
    if("${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_CLANG_RESOURCE_INCLUDE}" STREQUAL "")
      list(APPEND _args
        "--clang-resource-include" "${IREE_CLANG_BUILTIN_HEADERS_PATH}")
    endif()
  elseif(NOT IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN STREQUAL "auto" AND
         NOT IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN STREQUAL "rocm" AND
         NOT IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN STREQUAL "llvm-tools")
    message(FATAL_ERROR
      "Unknown IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN='${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN}'. "
      "Expected auto, rocm, llvm-tools, or llvm-project.")
  endif()

  set(${out_args} "${_args}" PARENT_SCOPE)
endfunction()

_iree_hal_amdgpu_read_device_bitcode_sources(_BITCODE_SRCS)

set(_ABI_HDRS
  "${CMAKE_CURRENT_SOURCE_DIR}/../../abi/command_buffer.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../abi/common.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../abi/kernel_args.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../abi/profile.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../abi/queue.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../abi/signal.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../abi/timestamp.h"
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
  "${CMAKE_CURRENT_SOURCE_DIR}/../timestamp.h"
  ${_ABI_HDRS}
)

_iree_hal_amdgpu_expand_device_binary_targets(_DEVICE_BINARY_TARGETS)

set(_ARCH_BINARIES)
if(IREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE STREQUAL "prebuilt")
  foreach(_ARCH ${_DEVICE_BINARY_TARGETS})
    set(_BINARY_PATH
        "${CMAKE_CURRENT_SOURCE_DIR}/prebuilt/amdgcn-amd-amdhsa--${_ARCH}.so")
    if(NOT EXISTS "${_BINARY_PATH}")
      message(FATAL_ERROR
        "AMDGPU device binary target '${_ARCH}' was requested in prebuilt "
        "mode, but ${_BINARY_PATH} does not exist. Use "
        "-DIREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE=source or regenerate and "
        "check in the prebuilt blob.")
    endif()
    list(APPEND _ARCH_BINARIES "${_BINARY_PATH}")
  endforeach()
else()
  string(JOIN "," _DEVICE_BINARY_TARGETS_ARG ${_DEVICE_BINARY_TARGETS})
  _iree_hal_amdgpu_source_generator_args(
    _SOURCE_GENERATOR_ARGS "${_DEVICE_BINARY_TARGETS_ARG}")
  foreach(_ARCH ${_DEVICE_BINARY_TARGETS})
    list(APPEND _ARCH_BINARIES
      "${CMAKE_CURRENT_BINARY_DIR}/amdgcn-amd-amdhsa--${_ARCH}.so"
    )
  endforeach()
  add_custom_command(
    OUTPUT
      ${_ARCH_BINARIES}
    COMMAND
      ${_SOURCE_GENERATOR_ARGS}
    DEPENDS
      "${IREE_SOURCE_DIR}/build_tools/scripts/amdgpu_device_binaries.py"
      "${IREE_SOURCE_DIR}/build_tools/scripts/amdgpu_target_map.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/../device_bitcode_sources.bzl"
      ${_BITCODE_SRCS}
      ${_BITCODE_HDRS}
    COMMENT
      "Generating AMDGPU builtin device binaries"
    VERBATIM
  )
endif()

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
