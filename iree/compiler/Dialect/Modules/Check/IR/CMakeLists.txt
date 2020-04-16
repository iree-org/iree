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
    "CheckOps.h"
    "CheckOps.h.inc"
  SRCS
    "CheckOps.cpp"
    "CheckOps.cpp.inc"
  DEPS
    MLIRIR
    MLIRStandardOps
    iree::compiler::Dialect::HAL::IR
  PUBLIC
)

iree_cc_library(
  NAME
    CheckDialect
  HDRS
    "CheckDialect.h"
  SRCS
    "CheckDialect.cpp"
  DEPS
    ::IR
    MLIRIR
    MLIRParser
    MLIRTransforms
    iree::compiler::Dialect::HAL::Conversion
    iree::compiler::Dialect::Modules::Check::Conversion
    iree::compiler::Dialect::Modules::Check::check_imports
    iree::compiler::Dialect::VM::Conversion
  PUBLIC
)

iree_tablegen_library(
  NAME
    check_ops_gen
  TD_FILE
    "CheckOps.td"
  OUTS
    -gen-op-decls CheckOps.h.inc
    -gen-op-defs CheckOps.cpp.inc
)

iree_tablegen_doc(
  NAME
    CheckDialectDocGen
  TD_FILE
    "CheckOps.td"
  OUTS
    -gen-dialect-doc CheckDialect.md
)
