# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

# ─── Tablegen ────────────────────────────────────────────────────────────────
# Generate op declarations/definitions AND dialect declarations/definitions
# from a single MIPSOps.td source (same pattern as LinalgExt).

iree_tablegen_library(
  NAME
    MIPSOpsIncGen
  TD_FILE
    "MIPSOps.td"
  OUTS
    --gen-op-decls MIPSOps.h.inc
    --gen-op-defs  MIPSOps.cpp.inc
    --dialect=mips --gen-dialect-decls MIPSDialect.h.inc
    --dialect=mips --gen-dialect-defs  MIPSDialect.cpp.inc
)

# ─── C++ library ─────────────────────────────────────────────────────────────

iree_cc_library(
  NAME
    IR
  HDRS
    "MIPSDialect.h"
    "MIPSOps.h"
    "MIPSDialect.h.inc"
  TEXTUAL_HDRS
    "MIPSOps.h.inc"
    "MIPSOps.cpp.inc"
  SRCS
    "MIPSDialect.cpp"
    "MIPSDialect.cpp.inc"
    "MIPSOps.cpp"
    "MIPSBufferizableOpInterface.cpp"
  DEPS
    ::MIPSOpsIncGen
    LLVMSupport
    MLIRIR
    MLIRSupport
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRTensorDialect
    MLIRBufferizationDialect
    MLIRBufferizationTransforms
    MLIRDestinationStyleOpInterface
    MLIRInferTypeOpInterface
    MLIRSideEffectInterfaces
    MLIRTensorUtils
    MLIRTransforms
  PUBLIC
)

# ─── Documentation ───────────────────────────────────────────────────────────

iree_tablegen_doc(
  NAME
    MIPSDialectDocGen
  CATEGORY "Dialects"
  TD_FILE
    "MIPSOps.td"
  OUTS
    --gen-dialect-doc -dialect=mips MIPSDialect.md
)
