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

add_subdirectory(test)

iree_cc_library(
  NAME
    IR
  HDRS
    "VMDialect.h"
    "VMEnums.h.inc"
    "VMFuncEncoder.h"
    "VMOpInterface.h.inc"
    "VMOps.h"
    "VMOps.h.inc"
    "VMTraits.h"
    "VMTypes.h"
  SRCS
    "VMDialect.cpp"
    "VMEnums.cpp.inc"
    "VMOpEncoder.cpp.inc"
    "VMOpFolders.cpp"
    "VMOpInterface.cpp.inc"
    "VMOps.cpp"
    "VMOps.cpp.inc"
    "VMTypes.cpp"
  DEPS
    LLVMSupport
    MLIRIR
    MLIRParser
    MLIRStandardOps
    MLIRSupport
    MLIRTransformUtils
    iree::compiler::Dialect::IREE::IR
  ALWAYSLINK
  PUBLIC
)

iree_tablegen_library(
  NAME
    VMEnumsGen
  TD_FILE
    "VMBase.td"
  OUTS
    -gen-enum-decls VMEnums.h.inc
    -gen-enum-defs VMEnums.cpp.inc
)

iree_tablegen_library(
  NAME
    VMOpsGen
  TD_FILE
    "VMOps.td"
  OUTS
    -gen-op-decls VMOps.h.inc
    -gen-op-defs VMOps.cpp.inc
)

iree_tablegen_library(
  NAME
    VMOpEncoderGen
  TD_FILE
    "VMOps.td"
  OUTS
    -gen-iree-vm-op-encoder-defs VMOpEncoder.cpp.inc
  TBLGEN
    IREE
)

iree_tablegen_library(
  NAME
    VMOpInterfaceGen
  TD_FILE
    "VMBase.td"
  OUTS
    -gen-op-interface-decls VMOpInterface.h.inc
    -gen-op-interface-defs VMOpInterface.cpp.inc
)
