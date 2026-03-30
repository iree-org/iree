# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_tablegen_library(
  NAME
    PassesIncGen
  TD_FILE
    "Passes.td"
  OUTS
    --gen-pass-decls Passes.h.inc
)

iree_cc_library(
  NAME
    TransformsPassHeaders
  HDRS
    "Passes.h"
    "Passes.h.inc"
  DEPS
    ::PassesIncGen
    MLIRPass
  PUBLIC
)

iree_cc_library(
  NAME
    Transforms
  SRCS
    "IREEAffineExpandIndexOps.cpp"
    "IREELowerAffine.cpp"
    "Passes.cpp"
  DEPS
    ::PassesIncGen
    ::TransformsPassHeaders
    MLIRAffineDialect
    MLIRAffineToStandard
    MLIRAffineTransforms
    MLIRAffineUtils
    MLIRArithDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRPass
    MLIRSCFDialect
    MLIRSupport
    MLIRTransforms
    MLIRVectorDialect
  PUBLIC
)

iree_tablegen_doc(
  NAME
    CodegenTransformsPassesDocGen
  CATEGORY
    "Passes"
  TD_FILE
    "Passes.td"
  OUTS
    --gen-pass-doc CodegenTransformsPasses.md
)
