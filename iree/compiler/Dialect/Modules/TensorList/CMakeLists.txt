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

file(GLOB _GLOB_X_TD CONFIGURE_DEPENDS *.td)
iree_cc_embed_data(
  NAME
    tensorlist_imports
  SRCS
    "tensorlist.imports.mlir"
  CC_FILE_OUTPUT
    "tensorlist.imports.cc"
  H_FILE_OUTPUT
    "tensorlist.imports.h"
  CPP_NAMESPACE
    "mlir::iree_compiler::IREE::TensorList"
  FLATTEN
  PUBLIC
)

iree_cc_binary(
  NAME
    tensorlist-opt
  OUT
    tensorlist-opt
  DEPS
    iree::compiler::Dialect::Modules::TensorList::IR::TensorListDialect
    iree::tools::iree_opt_library
)

iree_cc_binary(
  NAME
    tensorlist-translate
  OUT
    tensorlist-translate
  DEPS
    iree::compiler::Dialect::Modules::TensorList::IR::TensorListDialect
    iree::tools::iree_translate_library
)
