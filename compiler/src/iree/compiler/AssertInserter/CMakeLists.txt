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

iree_cc_library(
  NAME
    AssertInserter
  HDRS
    "Passes.h"
  SRCS
    "AssertInBounds.cpp"
    "Passes.cpp"
  DEPS
    ::PassHeaders
    ::PassesIncGen
    LLVMSupport
    MLIRArithDialect
    MLIRArithTransforms
    MLIRIR
    MLIRPass
    MLIRAffineToStandard
  PUBLIC
)

iree_tablegen_doc(
  NAME
  AssertInserterPassesDocGen
  CATEGORY
    "Passes"
  TD_FILE
    "Passes.td"
  OUTS
    --gen-pass-doc AssertInserterPasses.md
)
