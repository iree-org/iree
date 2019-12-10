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

add_custom_command(
  OUTPUT simple_compute_test_module.emod
  COMMAND iree-translate
          --mlir-to-iree-module "${CMAKE_CURRENT_SOURCE_DIR}/simple_compute_test.mlir"
          -o simple_compute_test_module.emod
  # TODO(marbre): Get iree_translate target working.
  #DEPENDS iree_translate
)

add_custom_command(
  OUTPUT simple_compute_test_module.h simple_compute_test_module.cc
  COMMAND generate_cc_embed_data simple_compute_test_module.emod
          --output_header=simple_compute_test_module.h
          --output_impl=simple_compute_test_module.cc
          --identifier=simple_compute_test_module
          --cpp_namespace=iree::hal::samples
  DEPENDS generate_cc_embed_data
          # TODO(marbre): Enable dep as soon as target is working.
          #simple_compute_test_module.emod
)

add_library(
  simple_compute_test_module STATIC
  simple_compute_test_module.cc
)

iree_cc_test(
  NAME
    simple_compute_test
  SRCS
    "simple_compute_test.cc"
  DEPS
    absl::inlined_vector
    absl::strings
    absl::time
    gtest_main
    iree::base::flatbuffer_util
    iree::base::status
    iree::base::status_matchers
    iree::hal::command_buffer
    iree::hal::command_queue
    iree::hal::driver_registry
    iree::schemas
    # Enabled drivers:
    iree::hal::interpreter::interpreter_driver_module
    iree::hal::vulkan::vulkan_driver_module
    simple_compute_test_module
)
