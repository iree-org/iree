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
    dialect
  HDRS
    "conversion_patterns.h"
    "check_dialect.h"
    "check_ops.h"
    "check_ops.h.inc"
  SRCS
    "conversion_patterns.cc"
    "check_dialect.cc"
    "check_ops.cc"
    "check_ops.cc.inc"
  DEPS
    LLVMSupport
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRTransforms
    iree::compiler::Dialect::VM::Conversion
    iree::modules::Check::dialect::Check_imports
    iree::samples::custom_modules::dialect::custom_imports
  ALWAYSLINK
  PUBLIC
)

iree_tablegen_library(
  NAME
    check_ops_gen
  SRCS
    check_ops.td
  OUTS
    -gen-op-decls check_ops.h.inc
    -gen-op-defs check_ops.cc.inc
)

iree_cc_embed_data(
  NAME
    check_imports
  SRCS
    "check.imports.mlir"
  CC_FILE_OUTPUT
    "check.imports.cc"
  H_FILE_OUTPUT
    "check.imports.h"
  CPP_NAMESPACE
    "mlir::iree_compiler::IREE::Check"
  FLATTEN
)

iree_cc_binary(
  NAME
    check-opt
  DEPS
    MLIROptMain
    iree::modules::Check::dialect
    iree::tools::iree_opt_library
)

iree_cc_binary(
  NAME
    check-translate
  DEPS
    iree::modules::Check::dialect
    iree::tools::iree_translate_library
)
