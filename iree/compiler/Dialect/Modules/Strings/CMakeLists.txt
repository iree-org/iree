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

add_subdirectory(Conversion)
add_subdirectory(IR)

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
  PUBLIC
)

iree_cc_binary(
  NAME
    strings-opt
  OUT
    strings-opt
  DEPS
    iree::compiler::Dialect::Modules::Strings::IR::Dialect
    iree::tools::iree_opt_library
)

iree_cc_binary(
  NAME
    strings-translate
  OUT
    strings-translate
  DEPS
    iree::compiler::Dialect::Modules::Strings::IR::Dialect
    iree::tools::iree_translate_library
)
