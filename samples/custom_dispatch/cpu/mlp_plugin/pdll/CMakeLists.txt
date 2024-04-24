# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_lit_test_suite(
  NAME
    mlp_pdll_linalg_example
  SRCS
    "mlp_linalg_pdll.mlir"
  TOOLS
    FileCheck
    iree-compile
    iree-run-module
    iree_samples_custom_dispatch_cpu_mlp_plugin
  DATA
    "mlp_linalg_spec.pdll"
  LABELS
    "driver=local-sync"
    "hostonly"
)

if(IREE_INPUT_TORCH)
  iree_lit_test_suite(
    NAME
      mlp_pdll_torch_example
    SRCS
      "mlp_torch_pdll.mlir"
    TOOLS
      FileCheck
      iree-compile
      iree-run-module
      iree_samples_custom_dispatch_cpu_mlp_plugin
    DATA
      "mlp_torch_spec.pdll"
    LABELS
      "driver=local-sync"
      "hostonly"
  )
endif()

if(IREE_INPUT_TOSA)
  iree_lit_test_suite(
    NAME
      mlp_pdll_tosa_example
    SRCS
      "mlp_tosa_pdll.mlir"
    TOOLS
      FileCheck
      iree-compile
      iree-run-module
      iree_samples_custom_dispatch_cpu_mlp_plugin
    DATA
      "mlp_tosa_spec.pdll"
    LABELS
      "driver=local-sync"
      "hostonly"
  )
endif()
