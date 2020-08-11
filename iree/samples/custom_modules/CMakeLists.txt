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

if(NOT ${IREE_TARGET_BACKEND_VMLA} OR NOT ${IREE_HAL_DRIVER_VMLA})
  return()
endif()

iree_add_all_subdirs()

iree_bytecode_module(
  NAME
    custom_modules_test_module
  SRC
    "custom_modules_test.mlir"
  CC_NAMESPACE
    "iree::samples::custom_modules"
  TRANSLATE_TOOL
    iree_samples_custom_modules_dialect_custom-translate
  FLAGS
    "-iree-mlir-to-vm-bytecode-module"
  PUBLIC
)

iree_cc_test(
  NAME
    custom_modules_test
  SRCS
    "custom_modules_test.cc"
  DEPS
    ::custom_modules_test_module_cc
    ::native_module
    absl::core_headers
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
    iree::base::api
    iree::hal::api
    iree::modules::hal
    iree::vm
    iree::vm::module_abi_cc
  PUBLIC
)
