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

add_subdirectory(dialect)

iree_bytecode_module(
  NAME
    check_test_module
  SRC
    "check_test.mlir"
  CC_NAMESPACE
    "iree::check"
  TRANSLATE_TOOL
    iree_modules_check_dialect_check-translate
  TRANSLATION
    "-iree-mlir-to-vm-bytecode-module"
  PUBLIC
)

iree_cc_test(
  NAME
    check_test
  SRCS
    "check_test.cc"
  DEPS
    ::check_test_module_cc
    ::native_module
    absl::core_headers
    absl::strings
    iree::base::api
    iree::base::logging
    iree::hal::api
    iree::modules::hal
    iree::testing::gtest_main
    iree::vm
    iree::vm::bytecode_module
    iree::vm::ref
)

iree_cc_library(
  NAME
    native_module
  HDRS
    "native_module.h"
  SRCS
    "native_module.cc"
  DEPS
    absl::strings
    iree::base::api
    iree::base::api_util
    iree::base::buffer_string_util
    iree::hal::api
    iree::modules::hal
    iree::vm
    iree::vm::module_abi_cc
  PUBLIC
)
