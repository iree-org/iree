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

add_custom_command(
  OUTPUT simple_module_test_bytecode_module.emod
  COMMAND iree-translate
          --mlir-to-iree-module "${CMAKE_CURRENT_SOURCE_DIR}/simple_module_test.mlir"
          -o simple_module_test_bytecode_module.emod
  # TODO(marbre): Get iree_translate target working.
  #DEPENDS iree_translate
)

add_custom_command(
  OUTPUT simple_module_test_bytecode_module.h simple_module_test_bytecode_module.cc
  COMMAND generate_cc_embed_data simple_module_test_bytecode_module.emod
          --output_header=simple_module_test_bytecode_module.h
          --output_impl=simple_module_test_bytecode_module.cc
          --identifier=simple_module_test_bytecode_module
          --cpp_namespace=iree::rt::samples
  DEPENDS generate_cc_embed_data
          # TODO(marbre): Enable dep as soon as target is working.
          #simple_module_test_bytecode_module.emod
)

add_library(
  simple_module_test_bytecode_module STATIC
  simple_module_test_bytecode_module.cc
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
    iree::schemas
    iree::rt
    iree::vm::bytecode_module
    iree::vm::sequencer_module
    # Enabled drivers:
    iree::hal::interpreter::interpreter_driver_module
    iree::hal::vulkan::vulkan_driver_module
    simple_module_test_bytecode_module
)
