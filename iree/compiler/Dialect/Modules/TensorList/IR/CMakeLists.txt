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
    "TensorListOps.h"
    "TensorListOps.h.inc"
    "TensorListTypes.h"
  TEXTUAL_HDRS
    "TensorListOps.cpp.inc"
  SRCS
    "TensorListOps.cpp"
    "TensorListTypes.cpp"
  DEPS
    LLVMSupport
    MLIRIR
    MLIRParser
    MLIRStandardOps
    MLIRSupport
    MLIRTransformUtils
    iree::compiler::Dialect::HAL::IR
    iree::compiler::Dialect::IREE::IR
  PUBLIC
)

iree_cc_library(
  NAME
    TensorListDialect
  HDRS
    "TensorListDialect.h"
  SRCS
    "TensorListDialect.cpp"
  DEPS
    ::IR
    LLVMSupport
    MLIRIR
    MLIRParser
    MLIRTransformUtils
    iree::compiler::Dialect::HAL::Conversion
    iree::compiler::Dialect::HAL::IR
    iree::compiler::Dialect::IREE::IR
    iree::compiler::Dialect::Modules::TensorList::Conversion
    iree::compiler::Dialect::Modules::TensorList::tensorlist_imports
    iree::compiler::Dialect::VM::Conversion
  ALWAYSLINK
  PUBLIC
)

iree_tablegen_library(
  NAME
    TensorListOpsGen
  TD_FILE
    "TensorListOps.td"
  OUTS
    -gen-op-decls TensorListOps.h.inc
    -gen-op-defs TensorListOps.cpp.inc
)
