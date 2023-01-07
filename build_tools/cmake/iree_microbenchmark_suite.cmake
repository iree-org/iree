# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# iree_microbenchmark_suite()
#
# Generates microbenchmark suites for MLIR input modules.
# Parameters:
#   NAME: Name of target.
#   SRCS: Source files to compile into a bytecode module (list of strings).
#   FLAGS: Flags to pass to the compiler tool (list of strings).

function(iree_microbenchmark_suite)
  if(NOT IREE_BUILD_MICROBENCHMARKS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "SRCS;FLAGS"
    ${ARGN}
  )

  iree_package_name(PACKAGE_NAME)

  foreach(_SRC IN LISTS _RULE_SRCS)
    set(_COMPILE_TOOL "iree-compile")
    set(_TRANSLATE_SRC "${_SRC}")
    set(_MODULE_FILE_NAME "${_RULE_NAME}_${_SRC}.vmfb")
    set(_TARGET_NAME "${PACKAGE_NAME}_${_MODULE_FILE_NAME}")
    set(_ARGS "${_RULE_FLAGS}")
    get_filename_component(_TRANSLATE_SRC_PATH "${_TRANSLATE_SRC}" REALPATH)
    list(APPEND _ARGS "${_TRANSLATE_SRC_PATH}")
    list(APPEND _ARGS "-o")
    list(APPEND _ARGS "${_MODULE_FILE_NAME}")

    add_custom_command(
      OUTPUT
        "${_MODULE_FILE_NAME}"
      COMMAND
        ${_COMPILE_TOOL}
        ${_ARGS}
      DEPENDS
        ${_COMPILE_TOOL}
        ${_TRANSLATE_SRC}
      VERBATIM
    )
    add_custom_target("${_TARGET_NAME}"
      DEPENDS
        "${_MODULE_FILE_NAME}"
    )
    add_dependencies(iree-microbenchmark-suites "${_TARGET_NAME}")
  endforeach(_SRC IN LISTS _SRCS)
endfunction(iree_microbenchmark_suite)
