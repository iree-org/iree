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
    # Enabled drivers:
    iree::hal::interpreter::interpreter_driver_module
    iree::hal::vulkan::vulkan_driver_module
)

iree_cc_binary(
    NAME
      vk_triangle
    SRCS
      "vk_triangle.cc"
    DEPS
      iree::base::api
      iree::base::logging
      iree::hal::api
      iree::hal::vulkan::vulkan_driver_module
      iree::rt::api
      iree::vm::api
      SDL2-static
)
