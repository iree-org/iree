# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

iree_add_all_subdirs()

file(GLOB _GLOB_X_TD LIST_DIRECTORIES false RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS *.td)
iree_cc_library(
  NAME
    IR
  HDRS
    "HALOps.h"
    "HALTypes.h"
  TEXTUAL_HDRS
    "HALEnums.cpp.inc"
    "HALEnums.h.inc"
    "HALOpInterface.cpp.inc"
    "HALOpInterface.h.inc"
    "HALOps.cpp.inc"
    "HALOps.h.inc"
    "HALStructs.cpp.inc"
    "HALStructs.h.inc"
  SRCS
    "HALOpFolders.cpp"
    "HALOps.cpp"
    "HALTypes.cpp"
  DEPS
    LLVMSupport
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRStandardOps
    MLIRSupport
    MLIRTransformUtils
    iree::compiler::Dialect::IREE::IR
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
    MLIRIR
    MLIRParser
    MLIRStandardOps
    MLIRTransformUtils
    iree::compiler::Dialect::HAL::Conversion::HALToVM
    iree::compiler::Dialect::HAL::hal_imports
    iree::compiler::Dialect::VM::Conversion
  PUBLIC
)

iree_tablegen_library(
  NAME
    HALOpsGen
  TD_FILE
    "HALOps.td"
  OUTS
    -gen-op-decls HALOps.h.inc
    -gen-op-defs HALOps.cpp.inc
)

iree_tablegen_library(
  NAME
    HALStructsGen
  TD_FILE
    "HALBase.td"
  OUTS
    -gen-iree-struct-attr-decls HALStructs.h.inc
    -gen-iree-struct-attr-defs HALStructs.cpp.inc
  TBLGEN
    IREE
)

iree_tablegen_library(
  NAME
    HALTypesGen
  TD_FILE
    "HALBase.td"
  OUTS
    -gen-enum-decls HALEnums.h.inc
    -gen-enum-defs HALEnums.cpp.inc
    -gen-op-interface-decls HALOpInterface.h.inc
    -gen-op-interface-defs HALOpInterface.cpp.inc
)

iree_tablegen_doc(
  NAME
    HALDialectDocGen
  TD_FILE
    "HALOps.td"
  OUTS
    -gen-dialect-doc HALDialect.md
)
