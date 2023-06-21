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
    "RESULT_VAR_ENABLED;TIMEOUT;DRIVER"
    "LABELS"
    ${ARGN}
  )

  set(_ENABLED TRUE)

  # Exclude hostonly tests when cross-compiling.
  if(CMAKE_CROSSCOMPILING AND "hostonly" IN_LIST _FUNC_LABELS)
    set(_ENABLED FALSE)
  endif()

  # Sanitizer-based exclusions.
  # CUDA is considered incompatible with all sanitizers.
  if(IREE_ENABLE_ASAN)
    if (("noasan" IN_LIST _FUNC_LABELS) OR (_FUNC_DRIVER STREQUAL "cuda"))
      set(_ENABLED FALSE)
    endif()
  endif()
  if(IREE_ENABLE_TSAN)
    if (("notsan" IN_LIST _FUNC_LABELS) OR (_FUNC_DRIVER STREQUAL "cuda"))
      set(_ENABLED FALSE)
    endif()
  endif()

  # Architecture-based exclusions.
  if ((IREE_ARCH MATCHES "^riscv_") AND ("noriscv" IN_LIST _FUNC_LABELS))
    set(_ENABLED FALSE)
  endif()

  # Timeout-based exclusions.
  #
  # Tests timeout values are "short", "moderate", "long". The IREE-wide default
  # is "short".
  # On very slow configurations, we only run "short" tests (the IREE-wide
  # default), excluding "moderate" and "long". On other somewhat-slow configs,
  # we run "short" and "moderate" tests, excluding "long".
  set(_EXCLUDE_TIMEOUT_MODERATE OFF)
  set(_EXCLUDE_TIMEOUT_LONG OFF)
  if (IREE_ARCH MATCHES "^riscv_")
    # Very slow, because it's emulators on CI. We should have a dedicated
    # setting for emulators instead of treating all RISC-V as very slow.
    set(_EXCLUDE_TIMEOUT_MODERATE ON)
  endif()
  if (_EXCLUDE_TIMEOUT_MODERATE OR
      CMAKE_CROSSCOMPILING OR
      IREE_ENABLE_ASAN OR
      IREE_ENABLE_TSAN)
    set(_EXCLUDE_TIMEOUT_LONG ON)
  endif()
  if(_EXCLUDE_TIMEOUT_MODERATE AND _FUNC_TIMEOUT STREQUAL "moderate")
    set(_ENABLED FALSE)
  endif()
  if(_EXCLUDE_TIMEOUT_LONG AND _FUNC_TIMEOUT STREQUAL "long")
    set(_ENABLED FALSE)
  endif()

  # Communicate the result to the caller.
  set("${_FUNC_RESULT_VAR_ENABLED}" "${_ENABLED}" PARENT_SCOPE)
endfunction()
