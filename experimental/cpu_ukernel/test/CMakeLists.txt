# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_check_single_backend_test_suite(
  NAME
    builtin_ukernel_system_plugin_test
  SRCS
    "mmt4d.mlir"
  TARGET_BACKEND
    "llvm-cpu"
  DRIVER
    "local-sync"
  COMPILER_FLAGS
    "--iree-llvmcpu-enable-ukernels=all"
  RUNNER_ARGS
    "--executable_plugin=${PROJECT_BINARY_DIR}/experimental/cpu_ukernel/builtin_ukernel_system_plugin.so"
  LABELS
    "hostonly"
)
