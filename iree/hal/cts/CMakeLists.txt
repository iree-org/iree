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
    cts_test_base
  HDRS
    "cts_test_base.h"
  DATA
    iree::tools::sanitizer_suppressions.txt
  DEPS
    iree::base::status
    iree::base::status_matchers
    iree::hal::driver_registry
    iree::hal::dylib::dylib_driver_module
    iree::hal::llvmjit::llvmjit_driver_module
    iree::hal::vmla::vmla_driver_module
    iree::hal::vulkan::vulkan_driver_module
    iree::testing::gtest_main
  TESTONLY
  PUBLIC
)

iree_cc_test(
  NAME
    allocator_test
  SRCS
    "allocator_test.cc"
  DEPS
    ::cts_test_base
    iree::base::status
    iree::base::status_matchers
    iree::hal::driver_registry
    iree::testing::gtest
)

iree_cc_test(
  NAME
    command_buffer_test
  SRCS
    "command_buffer_test.cc"
  DEPS
    ::cts_test_base
    iree::base::status
    iree::base::status_matchers
    iree::hal::driver_registry
    iree::testing::gtest
)

iree_cc_test(
  NAME
    command_queue_test
  SRCS
    "command_queue_test.cc"
  DEPS
    ::cts_test_base
    iree::base::status
    iree::base::status_matchers
    iree::hal::driver_registry
    iree::testing::gtest
)

iree_cc_test(
  NAME
    device_creation_test
  SRCS
    "device_creation_test.cc"
  DEPS
    ::cts_test_base
    iree::hal::driver_registry
    iree::testing::gtest
)

iree_cc_test(
  NAME
    semaphore_test
  SRCS
    "semaphore_test.cc"
  DEPS
    ::cts_test_base
    iree::hal::driver_registry
    iree::testing::gtest
)
