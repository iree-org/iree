# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Macros for defining tests that use a trace-runner."""

def iree_generated_trace_runner_test(**kwargs):
    # TODO: implement this. For now, it's only parsed by bazel_to_cmake.py, so
    # the iree_generated_check_test's are just omitted in bazel builds.
    pass
