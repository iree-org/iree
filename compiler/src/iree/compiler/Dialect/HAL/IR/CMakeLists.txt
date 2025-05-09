################################################################################
# Autogenerated by build_tools/bazel_to_cmake/bazel_to_cmake.py from           #
# compiler/src/iree/compiler/Dialect/HAL/IR/BUILD.bazel                        #
#                                                                              #
# Use iree_cmake_extra_content from iree/build_defs.oss.bzl to add arbitrary   #
# CMake-only content.                                                          #
#                                                                              #
# To disable autogeneration for this file entirely, delete this header.        #
################################################################################

iree_add_all_subdirs()

iree_cc_library(
  NAME
    IR
  HDRS
    "HALDialect.h"
    "HALOps.h"
    "HALTypes.h"
  TEXTUAL_HDRS
    "HALAttrInterfaces.cpp.inc"
    "HALAttrInterfaces.h.inc"
    "HALAttrs.cpp.inc"
    "HALAttrs.h.inc"
    "HALEnums.cpp.inc"
    "HALEnums.h.inc"
    "HALOpInterfaces.cpp.inc"
    "HALOpInterfaces.h.inc"
    "HALOps.cpp.inc"
    "HALOps.h.inc"
    "HALTypeInterfaces.cpp.inc"
    "HALTypeInterfaces.h.inc"
  SRCS
    "HALAttrs.cpp"
    "HALOpFolders.cpp"
    "HALOps.cpp"
    "HALTypes.cpp"
  DEPS
    ::HALAttrsGen
    ::HALInterfacesGen
    ::HALOpsGen
    LLVMSupport
    MLIRArithDialect
    MLIRControlFlowDialect
    MLIRFuncDialect
    MLIRFunctionInterfaces
    MLIRIR
    MLIRInferIntRangeInterface
    MLIRInferTypeOpInterface
    MLIRMemRefDialect
    MLIRParser
    MLIRSCFDialect
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRTransformUtils
    MLIRViewLikeInterface
    iree::compiler::Dialect::Stream::IR
    iree::compiler::Dialect::Util::IR
    iree::compiler::Utils
  PUBLIC
)

iree_cc_library(
  NAME
    HALDialect
  HDRS
    "HALDialect.h"
  SRCS
    "HALDialect.cpp"
  DEPS
    ::IR
    LLVMSupport
    MLIRArithDialect
    MLIRControlFlowDialect
    MLIRFuncDialect
    MLIRFunctionInterfaces
    MLIRIR
    MLIRMemRefDialect
    MLIRParser
    MLIRSCFDialect
    MLIRTransformUtils
    iree::compiler::Dialect::Encoding::IR
    iree::compiler::Dialect::HAL::Analysis
    iree::compiler::Dialect::HAL::Conversion::HALToVM
    iree::compiler::Dialect::HAL::hal_imports
    iree::compiler::Dialect::Stream::IR
    iree::compiler::Dialect::Util::IR
    iree::compiler::Dialect::VM::Conversion
  PUBLIC
)

iree_tablegen_library(
  NAME
    HALAttrsGen
  TD_FILE
    "HALAttrs.td"
  OUTS
    --gen-attrdef-decls --attrdefs-dialect=hal HALAttrs.h.inc
    --gen-attrdef-defs --attrdefs-dialect=hal HALAttrs.cpp.inc
    --gen-enum-decls HALEnums.h.inc
    --gen-enum-defs HALEnums.cpp.inc
)

iree_tablegen_library(
  NAME
    HALInterfacesGen
  TD_FILE
    "HALInterfaces.td"
  OUTS
    --gen-attr-interface-decls HALAttrInterfaces.h.inc
    --gen-attr-interface-defs HALAttrInterfaces.cpp.inc
    --gen-op-interface-decls HALOpInterfaces.h.inc
    --gen-op-interface-defs HALOpInterfaces.cpp.inc
    --gen-type-interface-decls HALTypeInterfaces.h.inc
    --gen-type-interface-defs HALTypeInterfaces.cpp.inc
)

iree_tablegen_library(
  NAME
    HALOpsGen
  TD_FILE
    "HALOps.td"
  OUTS
    --gen-op-decls HALOps.h.inc
    --gen-op-defs HALOps.cpp.inc
)

iree_tablegen_doc(
  NAME
    HALDialectDocGen
  CATEGORY
    "Dialects"
  TD_FILE
    "HALOps.td"
  OUTS
    --gen-dialect-doc --dialect=hal HALDialect.md
)

### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###
