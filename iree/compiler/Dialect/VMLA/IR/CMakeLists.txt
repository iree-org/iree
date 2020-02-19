# Copyright 2020 Google LLC
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

add_subdirectory(test)

iree_cc_library(
  NAME
    IR
  HDRS
    "VMLAEnums.h.inc"
    "VMLAOpInterface.h.inc"
    "VMLAOps.h"
    "VMLAOps.h.inc"
    "VMLATraits.h"
    "VMLATypes.h"
  TEXTUAL_HDRS
    "VMLAOps.cpp.inc"
  SRCS
    "VMLAEnums.cpp.inc"
    "VMLAOpInterface.cpp.inc"
    "VMLAOps.cpp"
    "VMLATypes.cpp"
  DEPS
    LLVMSupport
    MLIRIR
    MLIRStandardOps
    MLIRSupport
    MLIRTransformUtils
    MLIRTranslation
    iree::compiler::Dialect::IREE::IR
    iree::compiler::Dialect::Shape::IR
    iree::compiler::Dialect::VM::IR
  PUBLIC
)

iree_cc_library(
  NAME
    VMLADialect
  HDRS
    "VMLADialect.h"
  SRCS
    "VMLADialect.cpp"
  DEPS
    ::IR
    LLVMSupport
    MLIRIR
    MLIRParser
    MLIRStandardOps
    MLIRSupport
    MLIRTransformUtils
    iree::compiler::Dialect::VM::Conversion
    iree::compiler::Dialect::VMLA::Conversion::VMLAToVM
    iree::compiler::Dialect::VMLA::vmla_imports
  ALWAYSLINK
  PUBLIC
)

iree_tablegen_library(
  NAME
    VMLAEnumsGen
  TD_FILE
    "VMLABase.td"
  OUTS
    -gen-enum-decls VMLAEnums.h.inc
    -gen-enum-defs VMLAEnums.cpp.inc
)

iree_tablegen_library(
  NAME
    VMLAOpsGen
  TD_FILE
    "VMLAOps.td"
  OUTS
    -gen-op-decls VMLAOps.h.inc
    -gen-op-defs VMLAOps.cpp.inc
)

iree_tablegen_library(
  NAME
    VMLAOpInterfaceGen
  TD_FILE
    "VMLABase.td"
  OUTS
    -gen-op-interface-decls VMLAOpInterface.h.inc
    -gen-op-interface-defs VMLAOpInterface.cpp.inc
)
