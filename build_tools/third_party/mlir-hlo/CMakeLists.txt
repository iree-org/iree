# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(TF_MLIR_HLO_SOURCE_DIR
  "${IREE_SOURCE_DIR}/third_party/mlir-hlo/"
)
set(TF_MLIR_HLO_BINARY_DIR
  "${IREE_BINARY_DIR}/llvm-external-projects/mlir-hlo"
)

external_cc_library(
  PACKAGE
    tensorflow
  NAME
    external_mhlo_includes
  ROOT
    ${TF_MLIR_HLO_SOURCE_DIR}
  INCLUDES
    "${TF_MLIR_HLO_SOURCE_DIR}"
    "${TF_MLIR_HLO_SOURCE_DIR}/include/"
    "${TF_MLIR_HLO_SOURCE_DIR}/stablehlo/"
    "${TF_MLIR_HLO_BINARY_DIR}"
    "${TF_MLIR_HLO_BINARY_DIR}/include/"
    "${TF_MLIR_HLO_BINARY_DIR}/stablehlo/"
  PUBLIC
)
