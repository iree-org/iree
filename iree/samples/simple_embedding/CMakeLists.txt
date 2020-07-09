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
    simple_embedding_test_bytecode_module
  SRC
    "simple_embedding_test.mlir"
  CC_NAMESPACE
    "iree::samples"
  FLAGS
    "-iree-mlir-to-vm-bytecode-module"
    "-iree-hal-target-backends=vmla"
    "-iree-hal-target-backends=vulkan-spirv"
  PUBLIC
)

iree_cc_test(
  NAME
    simple_embedding_test
  SRCS
    "simple_embedding_test.cc"
  DATA
    iree::tools::sanitizer_suppressions.txt
  DEPS
    ::simple_embedding_test_bytecode_module_cc
    absl::core_headers
    absl::strings
    iree::base::api
    iree::base::api_util
    iree::base::logging
    iree::hal::api
    iree::hal::vmla::vmla_driver_module
    iree::hal::vulkan::vulkan_driver_module
    iree::modules::hal
    iree::testing::gtest_main
    iree::vm
    iree::vm::bytecode_module
  LABELS
    "driver=vulkan"
)
