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

iree_bytecode_module(
  NAME
    tensorlist_test_module
  SRC
    "tensorlist_test.mlir"
  CC_NAMESPACE
    "iree::modules::tensorlist"
  FLAGS
    "-iree-mlir-to-vm-bytecode-module"
  PUBLIC
)

iree_cc_test(
  NAME
    tensorlist_test
  SRCS
    "tensorlist_test.cc"
  DEPS
    ::native_module
    ::tensorlist_test_module_cc
    absl::core_headers
    absl::inlined_vector
    absl::span
    absl::strings
    iree::base::api
    iree::base::logging
    iree::hal::api
    iree::hal::vmla::vmla_driver_module
    iree::modules::hal
    iree::testing::gtest_main
    iree::vm
    iree::vm::bytecode_module
    iree::vm::ref_cc
)

iree_cc_library(
  NAME
    native_module
  HDRS
    "native_module.h"
  SRCS
    "native_module.cc"
  DEPS
    absl::inlined_vector
    absl::span
    iree::base::api
    iree::base::ref_ptr
    iree::base::status
    iree::hal::api
    iree::modules::hal
    iree::vm
    iree::vm::module_abi_cc
  PUBLIC
)
