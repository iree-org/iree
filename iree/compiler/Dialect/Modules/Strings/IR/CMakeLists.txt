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

file(GLOB _GLOB_X_TD CONFIGURE_DEPENDS *.td)
iree_cc_library(
  NAME
    IR
  HDRS
    "Ops.h"
    "Ops.h.inc"
    "Types.h"
  SRCS
    "Ops.cc"
    "Ops.cc.inc"
  DEPS
    LLVMSupport
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSideEffects
    MLIRSupport
    MLIRTransforms
    iree::compiler::Dialect::HAL::IR
    iree::compiler::Dialect::IREE::IR
    iree::compiler::Dialect::VM::Conversion
  ALWAYSLINK
  PUBLIC
)

iree_cc_library(
  NAME
    Dialect
  HDRS
    "Dialect.h"
  SRCS
    "Dialect.cc"
  DEPS
    ::IR
    LLVMSupport
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRTransforms
    iree::compiler::Dialect::IREE::IR
    iree::compiler::Dialect::Modules::Strings::Conversion
    iree::compiler::Dialect::Modules::Strings::strings_imports
    iree::compiler::Dialect::VM::Conversion
  ALWAYSLINK
  PUBLIC
)

iree_tablegen_library(
  NAME
    Ops_gen
  TD_FILE
    "Ops.td"
  OUTS
    -gen-op-decls Ops.h.inc
    -gen-op-defs Ops.cc.inc
)

iree_tablegen_doc(
  NAME
    DialectDocGen
  TD_FILE
    "Ops.td"
  OUTS
    -gen-dialect-doc StringsDialect.md
)
