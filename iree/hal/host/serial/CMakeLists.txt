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
    async_command_queue
  HDRS
    "async_command_queue.h"
  SRCS
    "async_command_queue.cc"
  DEPS
    absl::core_headers
    absl::synchronization
    iree::base::status
    iree::base::tracing
    iree::hal::command_queue
    iree::hal::host::serial::serial_submission_queue
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
    absl::memory
    iree::base::status
    iree::base::time
    iree::hal::command_queue
    iree::hal::host::serial::serial_submission_queue
    iree::hal::testing::mock_command_buffer
    iree::hal::testing::mock_command_queue
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    serial_command_processor
  HDRS
    "serial_command_processor.h"
  SRCS
    "serial_command_processor.cc"
  DEPS
    absl::inlined_vector
    iree::base::status
    iree::base::tracing
    iree::hal::command_buffer
    iree::hal::host::host_descriptor_set
    iree::hal::host::host_executable
    iree::hal::host::host_executable_layout
  PUBLIC
)

iree_cc_library(
  NAME
    serial_scheduling_model
  HDRS
    "serial_scheduling_model.h"
  SRCS
    "serial_scheduling_model.cc"
  DEPS
    ::async_command_queue
    ::serial_command_processor
    ::serial_submission_queue
    absl::inlined_vector
    iree::base::memory
    iree::base::status
    iree::base::tracing
    iree::hal::host::condvar_semaphore
    iree::hal::host::inproc_command_buffer
    iree::hal::host::nop_event
    iree::hal::host::scheduling_model
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
    absl::core_headers
    absl::inlined_vector
    absl::synchronization
    iree::base::intrusive_list
    iree::base::status
    iree::base::tracing
    iree::hal::command_queue
    iree::hal::host::condvar_semaphore
  PUBLIC
)
