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
    cc
  HDRS
    "allocator.h"
    "buffer.h"
    "command_buffer.h"
    "command_queue.h"
    "debug_capture_manager.h"
    "descriptor_set.h"
    "descriptor_set_layout.h"
    "device.h"
    "device_info.h"
    "driver.h"
    "event.h"
    "executable.h"
    "executable_cache.h"
    "executable_format.h"
    "executable_layout.h"
    "resource.h"
    "semaphore.h"
  SRCS
    "buffer.cc"
  DEPS
    absl::span
    absl::strings
    iree::base::core_headers
    iree::base::logging
    iree::base::ref_ptr
    iree::base::status
    iree::base::time
    iree::base::tracing
  PUBLIC
)

iree_cc_test(
  NAME
    buffer_test
  SRCS
    "buffer_mapping_test.cc"
    "buffer_test.cc"
  DEPS
    ::cc
    absl::span
    iree::base::status
    iree::testing::gtest
    iree::testing::gtest_main
)
