# Copyright 2021 Google LLC
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

if(${IREE_ENABLE_EMITC})

iree_cc_test(
  NAME
    module_test
  SRCS
    "module_test.cc"
  DEPS
    absl::strings
    iree::base::logging
    iree::base::status
    iree::testing::gtest
    iree::testing::gtest_main
    iree::vm::cc
    iree::vm::ops
    iree::vm::shims
    ::arithmetic_ops_cc
)

iree_bytecode_module(
  NAME
    arithmetic_ops
  SRC
    "../arithmetic_ops.mlir"
  CC_NAMESPACE
    "iree::vm::test::emitc"
  FLAGS
    "-iree-vm-ir-to-c-module"
  PUBLIC
)

iree_bytecode_module(
  NAME
    shift_ops
  SRC
    "../shift_ops.mlir"
  CC_NAMESPACE
    "iree::vm::test::emitc"
  FLAGS
    "-iree-vm-ir-to-c-module"
  PUBLIC
)

endif()
