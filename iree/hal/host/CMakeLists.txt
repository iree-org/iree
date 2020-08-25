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
    condvar_semaphore
  HDRS
    "condvar_semaphore.h"
  SRCS
    "condvar_semaphore.cc"
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
    condvar_semaphore_test
  SRCS
    "condvar_semaphore_test.cc"
  DEPS
    ::condvar_semaphore
    iree::base::api
    iree::base::status
    iree::testing::gtest
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
    host_executable
  HDRS
    "host_executable.h"
  DEPS
    iree::base::status
    iree::hal::descriptor_set
    iree::hal::executable
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
    iree::base::status
    iree::base::tracing
    iree::hal::allocator
    iree::hal::buffer
  PUBLIC
)

iree_cc_library(
  NAME
    host_local_device
  HDRS
    "host_local_device.h"
  SRCS
    "host_local_device.cc"
  DEPS
    ::host_descriptor_set
    ::host_executable_layout
    ::host_local_allocator
    ::scheduling_model
    absl::core_headers
    absl::memory
    absl::span
    iree::base::memory
    iree::base::status
    iree::base::tracing
    iree::hal::allocator
    iree::hal::command_buffer_validation
    iree::hal::device
  PUBLIC
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
    nop_event
  HDRS
    "nop_event.h"
  SRCS
    "nop_event.cc"
  DEPS
    iree::hal::event
  PUBLIC
)

iree_cc_library(
  NAME
    scheduling_model
  HDRS
    "scheduling_model.h"
  DEPS
    iree::hal::command_queue
  PUBLIC
)
