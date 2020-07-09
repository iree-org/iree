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
    all_bytecode_modules_cc
  GENERATED_SRCS
    "arithmetic_ops.module"
    "control_flow_ops.module"
    "list_ops.module"
  CC_FILE_OUTPUT
    "all_bytecode_modules.cc"
  H_FILE_OUTPUT
    "all_bytecode_modules.h"
  CPP_NAMESPACE
    "iree::vm::test"
  FLATTEN
  PUBLIC
)

iree_bytecode_module(
  NAME
    arithmetic_ops
  SRC
    "arithmetic_ops.mlir"
  FLAGS
    "-iree-vm-ir-to-bytecode-module"
  PUBLIC
)

iree_bytecode_module(
  NAME
    control_flow_ops
  SRC
    "control_flow_ops.mlir"
  FLAGS
    "-iree-vm-ir-to-bytecode-module"
  PUBLIC
)

iree_bytecode_module(
  NAME
    list_ops
  SRC
    "list_ops.mlir"
  CC_NAMESPACE
    "iree::vm::test"
  FLAGS
    "-iree-vm-ir-to-bytecode-module"
  PUBLIC
)
