# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

function(iree_filter_test)
  cmake_parse_arguments(
    _FUNC
    ""
    "RESULT_VAR_ENABLED;SIZE;DRIVER"
    "LABELS"
    ${ARGN}
  )
  # Default to excluding; then exclusion cases are implemented as early-returns;
  # then at the end we will override that to TRUE.
  set("${_FUNC_RESULT_VAR_ENABLED}" FALSE PARENT_SCOPE)
  # Exclude hostonly tests when cross-compiling.
  if(CMAKE_CROSSCOMPILING AND "hostonly" IN_LIST _FUNC_LABELS)
    return()
  endif()
  # Sanitizer-based exclusions.
  # CUDA is considered incompatible with all sanitizers.
  if(IREE_ENABLE_ASAN)
    if (("noasan" IN_LIST _FUNC_LABELS) OR (_FUNC_DRIVER STREQUAL "cuda"))
      return()
    endif()
  endif()
  if(IREE_ENABLE_TSAN)
    if (("notsan" IN_LIST _FUNC_LABELS) OR (_FUNC_DRIVER STREQUAL "cuda"))
      return()
    endif()
  endif()
  # Architecture-based exclusions.
  if ((IREE_ARCH MATCHES "^riscv_") AND ("noriscv" IN_LIST _FUNC_LABELS))
    return()
  endif()
  # Size-based exclusions.
  set(_EXCLUDE_SIZE_LARGE OFF)
  set(_EXCLUDE_SIZE_ENORMOUS OFF)
  # On very slow configurations, exclude "large" tests.
  # Note: RISC-V is treated very slow because at the moment it's emulators on
  # CI. We should have a dedicated emulator setting instead.
  if (IREE_ARCH MATCHES "^riscv_")
    set(_EXCLUDE_SIZE_LARGE ON)
  endif()
  # On somewhat slow configurations, such as sanitizers, exclude only
  # "enormous" tests.
  if (_EXCLUDE_SIZE_LARGE OR
      CMAKE_CROSSCOMPILING OR
      IREE_ENABLE_ASAN OR
      IREE_ENABLE_TSAN)
    set(_EXCLUDE_SIZE_ENORMOUS ON)
  endif()
  if(_EXCLUDE_SIZE_LARGE AND _FUNC_SIZE STREQUAL "large")
    return()
  endif()
  if(_EXCLUDE_SIZE_ENORMOUS AND _FUNC_SIZE STREQUAL "enormous")
    return()
  endif()
  # Test is not excluded, communicate that to the caller.
  set("${_FUNC_RESULT_VAR_ENABLED}" TRUE PARENT_SCOPE)
endfunction()
