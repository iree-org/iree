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

iree_cc_library(
  NAME
    strings_module
  HDRS
    "api.h"
    "api_detail.h"
    "strings_module.h"
  SRCS
    "api.cc"
    "strings_module.cc"
  DEPS
    absl::inlined_vector
    absl::span
    absl::strings
    benchmark
    iree::base::api
    iree::base::logging
    iree::hal::api
    iree::modules::hal
    iree::vm
    iree::vm::bytecode_module
    iree::vm::module
    iree::vm::module_abi_cc
    iree::vm::ref
    iree::vm::stack
  PUBLIC
)

iree_cc_test(
  NAME
    strings_module_test
  SRCS
    "strings_module_test.cc"
  DEPS
    ::strings_module
    ::strings_module_test_module_cc
    absl::inlined_vector
    absl::strings
    benchmark
    iree::base::api
    iree::base::logging
    iree::hal::api
    iree::hal::vmla::vmla_driver_module
    iree::modules::hal
    iree::testing::gtest_main
    iree::vm
    iree::vm::bytecode_module
    iree::vm::module_abi_cc
    iree::vm::ref_cc
)

iree_bytecode_module(
  NAME
    strings_module_test_module
  SRC
    "strings_module_test.mlir"
  CC_NAMESPACE
    "iree::strings_module_test"
  FLAGS
    "-iree-mlir-to-vm-bytecode-module"
  PUBLIC
)
