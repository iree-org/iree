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
    "strings_dialect.h"
    "strings_ops.h"
    "strings_ops.h.inc"
  SRCS
    "conversion_patterns.cc"
    "strings_dialect.cc"
    "strings_ops.cc"
    "strings_ops.cc.inc"
  DEPS
    LLVMSupport
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRTransforms
    iree::compiler::Dialect::VM::Conversion
    iree::modules::Check::dialect::Check_imports
  ALWAYSLINK
  PUBLIC
)

iree_tablegen_library(
  NAME
    strings_ops_gen
  SRCS
    strings_ops.td
  OUTS
    -gen-op-decls strings_ops.h.inc
    -gen-op-defs strings_ops.cc.inc
)

iree_cc_embed_data(
  NAME
    strings_imports
  SRCS
    "strings.imports.mlir"
  CC_FILE_OUTPUT
    "strings.imports.cc"
  H_FILE_OUTPUT
    "strings.imports.h"
  CPP_NAMESPACE
    "mlir::iree_compiler::IREE::Strings"
  FLATTEN
)

iree_cc_binary(
  NAME
    strings-opt
  DEPS
    MLIROptMain
    iree::modules::Check::dialect
    iree::tools::iree_opt_library
)

iree_cc_binary(
  NAME
    strings-translate
  DEPS
    iree::modules::Check::dialect
    iree::tools::iree_translate_library
)
