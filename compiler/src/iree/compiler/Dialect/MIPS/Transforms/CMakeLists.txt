# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

# ─── Tablegen: generate Passes.h.inc from Passes.td ──────────────────────────

iree_tablegen_library(
  NAME
    PassesIncGen
  TD_FILE
    "Passes.td"
  OUTS
    --gen-pass-decls Passes.h.inc
)

# ─── Header-only pass declarations ───────────────────────────────────────────

iree_cc_library(
  NAME
    PassHeaders
  HDRS
    "Passes.h"
    "Passes.h.inc"
  DEPS
    ::PassesIncGen
    MLIRPass
    MLIRTransforms
  PUBLIC
)

# ─── Full transforms library ─────────────────────────────────────────────────

iree_cc_library(
  NAME
    Transforms
  HDRS
    "Passes.h"
  SRCS
    "LowerMIPSToFuncCall.cpp"
    "Passes.cpp"
  DEPS
    ::PassHeaders
    ::PassesIncGen
    iree::compiler::Dialect::MIPS::IR
    MLIRFuncDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRPass
    MLIRTransformUtils
    MLIRTransforms
  PUBLIC
)
