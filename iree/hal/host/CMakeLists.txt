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

iree_add_all_subdirs()

iree_cc_library(
  NAME
    async_command_queue
  HDRS
    "async_command_queue.h"
  SRCS
    "async_command_queue.cc"
  DEPS
    ::serial_submission_queue
    absl::core_headers
    absl::synchronization
    iree::base::status
    iree::base::tracing
    iree::hal::command_queue
    iree::hal::semaphore
  PUBLIC
)

iree_cc_test(
  NAME
    async_command_queue_test
  SRCS
    "async_command_queue_test.cc"
  DEPS
    ::async_command_queue
    ::serial_submission_queue
    absl::memory
    absl::time
    iree::base::status
    iree::base::status_matchers
    iree::base::time
    iree::hal::command_queue
    iree::hal::testing::mock_command_buffer
    iree::hal::testing::mock_command_queue
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    host_buffer
  HDRS
    "host_buffer.h"
  SRCS
    "host_buffer.cc"
  DEPS
    iree::base::logging
    iree::base::source_location
    iree::base::status
    iree::hal::buffer
  PUBLIC
)

iree_cc_library(
  NAME
    host_descriptor_set
  HDRS
    "host_descriptor_set.h"
  SRCS
    "host_descriptor_set.cc"
  DEPS
    absl::inlined_vector
    iree::hal::descriptor_set
    iree::hal::descriptor_set_layout
  PUBLIC
)

iree_cc_library(
  NAME
    host_event
  HDRS
    "host_event.h"
  SRCS
    "host_event.cc"
  DEPS
    iree::hal::event
  PUBLIC
)

iree_cc_library(
  NAME
    host_executable_layout
  HDRS
    "host_executable_layout.h"
  SRCS
    "host_executable_layout.cc"
  DEPS
    absl::inlined_vector
    iree::base::memory
    iree::hal::descriptor_set_layout
    iree::hal::executable_layout
  PUBLIC
)

iree_cc_library(
  NAME
    host_local_allocator
  HDRS
    "host_local_allocator.h"
  SRCS
    "host_local_allocator.cc"
  DEPS
    ::host_buffer
    iree::base::source_location
    iree::base::status
    iree::base::tracing
    iree::hal::allocator
    iree::hal::buffer
  PUBLIC
)

iree_cc_library(
  NAME
    host_local_command_processor
  HDRS
    "host_local_command_processor.h"
  SRCS
    "host_local_command_processor.cc"
  DEPS
    ::host_descriptor_set
    ::host_executable_layout
    absl::inlined_vector
    iree::base::source_location
    iree::base::status
    iree::base::tracing
    iree::hal::command_buffer
  PUBLIC
)

iree_cc_library(
  NAME
    host_semaphore
  HDRS
    "host_semaphore.h"
  SRCS
    "host_semaphore.cc"
  DEPS
    absl::core_headers
    absl::inlined_vector
    absl::span
    absl::synchronization
    iree::base::status
    iree::base::tracing
    iree::hal::semaphore
  PUBLIC
)

iree_cc_test(
  NAME
    host_semaphore_test
  SRCS
    "host_semaphore_test.cc"
  DEPS
    ::host_semaphore
    absl::time
    iree::base::status
    iree::base::status_matchers
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    inproc_command_buffer
  HDRS
    "inproc_command_buffer.h"
  SRCS
    "inproc_command_buffer.cc"
  DEPS
    iree::base::arena
    iree::base::intrusive_list
    iree::base::status
    iree::base::tracing
    iree::hal::command_buffer
  PUBLIC
)

iree_cc_library(
  NAME
    serial_submission_queue
  HDRS
    "serial_submission_queue.h"
  SRCS
    "serial_submission_queue.cc"
  DEPS
    ::host_semaphore
    absl::core_headers
    absl::inlined_vector
    absl::synchronization
    iree::base::intrusive_list
    iree::base::status
    iree::base::tracing
    iree::hal::command_queue
  PUBLIC
)
