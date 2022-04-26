# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

# iree_cc_binary_benchmarks()
#
# Creates a binary and a test for a cc benchmark target.
#
# It's good to test that benchmarks run, but it's really annoying to run a
# billion iterations of them every time you try to run tests. So we create these
# as binaries and then invoke them as tests with `--benchmark_min_time=0`.
#
# Mirrors the bzl function of the same name. See iree_cc_binary and iree_cc_test
# for more details on those rules
#
# Parameters:
# NAME: name for the binary target. The test target will be "${NAME}_test"
# SRCS: List of source files for the binary
# DATA: List of other targets and files required for the binary
# DEPS: List of other libraries to be linked in to the binary
# COPTS: List of private compile options for the binary
# DEFINES: List of public defines for the binary
# LINKOPTS: List of link options for the binary
# TESTONLY: whether the binary should only be compiled for tests
# HOSTONLY: whether the binary should be compiled using host toolchain when
#   cross-compiling
# LABELS: Additional labels to apply to the test. The package path is added
#     automatically.
#
# )

function(iree_cc_binary_benchmark)

  cmake_parse_arguments(
    _RULE
    "HOSTONLY;TESTONLY"
    "NAME"
    "SRCS;COPTS;DEFINES;LINKOPTS;DATA;DEPS;LABELS"
    ${ARGN}
  )

  set(_MAYBE_TESTONLY "")
  if(_RULE_TESTONLY)
    set(_MAYBE_TESTONLY "TESTONLY")
  endif()

  set(_MAYBE_HOSTONLY "")
  if(_RULE_HOSTONLY)
    set(_MAYBE_HOSTONLY "HOSTONLY")
  endif()

  iree_cc_binary(
    NAME
      ${_RULE_NAME}
    SRCS
      ${_RULE_SRCS}
    DEPS
      ${_RULE_DEPS}
    COPTS
      ${_RULE_COPTS}
    DEFINES
      ${_RULE_DEFINES}
    LINKOPTS
      ${_RULE_LINKOPTS}
    ${_MAYBE_TESTONLY}
    ${_MAYBE_HOSTONLY}
  )


  iree_native_test(
    NAME
      ${_RULE_NAME}_test
    ARGS
      "--benchmark_min_time=0"
    SRC
      ::${_RULE_NAME}
    LABELS
      ${_RULE_LABELS}
  )
endfunction()
