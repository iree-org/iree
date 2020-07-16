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

iree_add_all_subdirs()

file(GLOB _GLOB_X_TD LIST_DIRECTORIES false RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS *.td)
iree_cc_library(
  NAME
    IR
  HDRS
    "SequenceDialect.h"
    "SequenceEnums.h.inc"
    "SequenceOpInterface.h.inc"
    "SequenceOps.h"
    "SequenceOps.h.inc"
    "SequenceTypes.h"
  SRCS
    "SequenceDialect.cpp"
    "SequenceEnums.cpp.inc"
    "SequenceOpInterface.cpp.inc"
    "SequenceOps.cpp"
    "SequenceOps.cpp.inc"
    "SequenceTypes.cpp"
  DEPS
    LLVMSupport
    MLIRIR
    MLIRParser
    MLIRSupport
    iree::compiler::Dialect::IREE::IR
  PUBLIC
)

iree_tablegen_library(
  NAME
    SequenceEnumsGen
  TD_FILE
    "SequenceBase.td"
  OUTS
    -gen-enum-decls SequenceEnums.h.inc
    -gen-enum-defs SequenceEnums.cpp.inc
)

iree_tablegen_library(
  NAME
    SequenceOpInterfaceGen
  TD_FILE
    "SequenceBase.td"
  OUTS
    -gen-op-interface-decls SequenceOpInterface.h.inc
    -gen-op-interface-defs SequenceOpInterface.cpp.inc
)

iree_tablegen_library(
  NAME
    SequenceOpsGen
  TD_FILE
    "SequenceOps.td"
  OUTS
    -gen-op-decls SequenceOps.h.inc
    -gen-op-defs SequenceOps.cpp.inc
)

iree_tablegen_doc(
  NAME
    SequenceDialectDocGen
  TD_FILE
    "SequenceOps.td"
  OUTS
    -gen-dialect-doc SequenceDialect.md
)
