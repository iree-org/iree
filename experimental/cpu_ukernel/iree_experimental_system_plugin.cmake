# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# # iree_experimental_system_plugin
#
# Creates a system plugin library, that is built using the host toolchain for
# the host architecture and may be loaded with the system dynamic library
# loader.
#
# Contrast with: iree_experimental_standalone_plugin.
#
# Parameters:
# NAME: Name of the system plugin to create.
# SRCS: List of source files.
# DEPS: List of dependencies.
function(iree_experimental_system_plugin)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "SRCS;DEPS"
    ${ARGN}
  )

  iree_cc_library(
    NAME
      ${_RULE_NAME}
    SRCS
      ${_RULE_SRCS}
    DEPS
      ${_RULE_DEPS}
      iree::hal::local::executable_plugin
    INCLUDES
      "${IREE_SOURCE_DIR}/runtime/src/"
    SHARED
  )

  # NOTE: this is only required because we want this sample to run on all
  # platforms without needing to change the library name (libfoo.so/foo.dll).
  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")
  set_target_properties("${_NAME}"
    PROPERTIES
      WINDOWS_EXPORT_ALL_SYMBOLS ON
      PREFIX ""
      OUTPUT_NAME "${_RULE_NAME}"
  )

  add_dependencies(iree-test-deps "${_NAME}")
endfunction()
