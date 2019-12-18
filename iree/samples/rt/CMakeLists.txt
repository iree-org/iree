# Copyright 2019 Google LLC
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

add_subdirectory(vulkan)

iree_bytecode_module(
  NAME
    simple_module_test_bytecode_module
  SRC
    "simple_module_test.mlir"
  CC_NAMESPACE
    "iree::rt::samples"
)

iree_cc_test(
  NAME
    bytecode_module_test
  SRCS
    "bytecode_module_test.cc"
  DEPS
    absl::strings
    gtest_main
    iree::base::flatbuffer_util
    iree::base::status
    iree::base::status_matchers
    iree::hal::buffer_view
    iree::hal::driver_registry
    iree::samples::rt::simple_module_test_bytecode_module_cc
    iree::schemas
    iree::rt
    iree::vm::bytecode_module
    iree::vm::sequencer_module
    # Enabled drivers:
    iree::hal::interpreter::interpreter_driver_module
    iree::hal::vulkan::vulkan_driver_module
)
