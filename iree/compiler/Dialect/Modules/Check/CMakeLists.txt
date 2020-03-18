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
    iree::compiler::Dialect::Modules::Check::IR::CheckDialect
    iree::tools::iree_opt_main
)

iree_cc_binary(
  NAME
    check-translate
  OUT
    check-translate
  DEPS
    iree::compiler::Dialect::Modules::Check::IR::CheckDialect
    iree::tools::iree_translate_main
)
