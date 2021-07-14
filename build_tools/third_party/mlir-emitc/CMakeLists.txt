# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(MLIR_EMITC_SOURCE_DIR
  "${IREE_SOURCE_DIR}/third_party/mlir-emitc/"
)

external_cc_library(
  PACKAGE
    emitc
  NAME
    TranslateToCpp
  ROOT
    ${MLIR_EMITC_SOURCE_DIR}
    HDRS
      "include/emitc/Target/Cpp/CppEmitter.h"
    SRCS
      "lib/Target/Cpp/TranslateToCpp.cpp"
    DEPS
      MLIREmitC
      MLIRIR
      MLIRSCF
      MLIRStandard
  INCLUDES
    "${MLIR_EMITC_SOURCE_DIR}/include/"
  PUBLIC
)
