# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# iree_benchmark_suite_module()
#
# Compile bytecode modules for IREE benchmark suites. This is a wrapper of
# iree_bytecode_module to add benchmark-suite-specific CMake rules.
#
# Parameters:
# NAME: Name of target.
# SRC: Source file to compile into a bytecode module.
# MODULE_FILE_NAME: Output bytecode module file path.
# FRIENDLY_NAME: Optional. Name to use to display build progress info.
# FLAGS: Flags to pass to iree-compile (list of strings).
# PRESETS: Optional. Benchamrk presets to include this module target (list of
#     strings). For each preset, the target will be added as the dependencies of
#     "iree-benchmark-suites-${PRESET}".
function(iree_benchmark_suite_module)
  if(NOT IREE_BUILD_E2E_TEST_ARTIFACTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME;SRC;MODULE_FILE_NAME;FRIENDLY_NAME"
    "FLAGS;PRESETS"
    ${ARGN}
  )
  iree_validate_required_arguments(
    _RULE
    "NAME;SRC;MODULE_FILE_NAME"
    "FLAGS"
  )

  iree_bytecode_module(
    NAME
      "${_RULE_NAME}"
    SRC
      "${_RULE_SRC}"
    MODULE_FILE_NAME
      "${_RULE_MODULE_FILE_NAME}"
    FLAGS
      ${_RULE_FLAGS}
    FRIENDLY_NAME
      "${_RULE_FRIENDLY_NAME}"
    PUBLIC
  )

  # TODO(#10155): Dump the compile flags from iree_benchmark_suite_module into
  # flagfiles.

  iree_package_name(_PACKAGE_NAME)
  set(_MODULE_TARGET "${_PACKAGE_NAME}_${_RULE_NAME}")

  foreach(_PRESET IN LISTS _RULE_PRESETS)
    add_dependencies("iree-benchmark-suites-${_PRESET}" "${_MODULE_TARGET}")
  endforeach()
endfunction(iree_benchmark_suite)
