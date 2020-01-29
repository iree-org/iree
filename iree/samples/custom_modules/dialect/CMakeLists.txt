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

iree_cc_library(
  NAME
    dialect
  HDRS
    "conversion_patterns.h"
    "custom_dialect.h"
    "custom_ops.h.inc"
  SRCS
    "conversion_patterns.cc"
    "custom_dialect.cc"
    "custom_ops.cc.inc"
  DEPS
    iree::samples::custom_modules::dialect::custom_imports
    iree::compiler::Dialect::HAL::Conversion
    iree::compiler::Dialect::HAL::IR
    iree::compiler::Dialect::IREE::IR
    iree::compiler::Dialect::VM::Conversion
    LLVMSupport
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRTransforms
  ALWAYSLINK
  PUBLIC
)

iree_tablegen_library(
  NAME
    custom_ops_gen
  TD_FILE
    custom_ops.td
  OUTS
    -gen-op-decls custom_ops.h.inc
    -gen-op-defs custom_ops.cc.inc
)

iree_cc_embed_data(
  NAME
    custom_imports
  SRCS
    "custom.imports.mlir"
  CC_FILE_OUTPUT
    "custom.imports.cc"
  H_FILE_OUTPUT
    "custom.imports.h"
  CPP_NAMESPACE
    "mlir::iree_compiler::IREE::Custom"
  FLATTEN
)

iree_cc_binary(
  NAME
    custom-opt
  OUT
    custom-opt
  DEPS
    iree::samples::custom_modules::dialect
    iree::tools::iree_opt_library
    MLIROptMain
)

iree_cc_binary(
  NAME
    custom-translate
  OUT
    custom-translate
  DEPS
    iree::samples::custom_modules::dialect
    iree::tools::iree_translate_library
)
