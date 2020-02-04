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

# bazel_to_cmake: DO NOT EDIT
add_subdirectory(test)

iree_cc_library(
  NAME
    dialect
  HDRS
    "check_dialect.h"
    "check_ops.h"
    "check_ops.h.inc"
    "conversion_patterns.h"
  SRCS
    "check_dialect.cc"
    "check_ops.cc"
    "check_ops.cc.inc"
    "conversion_patterns.cc"
  DEPS
    ::check_imports
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
    check_ops_gen
  TD_FILE
    "check_ops.td"
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
  PUBLIC
)

iree_cc_binary(
  NAME
    check-opt
  OUT
    check-opt
  DEPS
    ::dialect
    iree::tools::iree_opt_library
    # MLIRMlirOptLib: includes main(); The equivalent Bazel library is MLIROptMain.
    # TODO(marbre): Replace with appropriate target after upstreams CMake is revised.
    MLIRMlirOptLib
)
add_executable(check-opt ALIAS iree_modules_check_dialect_check-opt)

iree_cc_binary(
  NAME
    check-translate
  DEPS
    ::dialect
    iree::tools::iree_translate_library
)
add_executable(check-translate ALIAS iree_modules_check_dialect_check-translate)
