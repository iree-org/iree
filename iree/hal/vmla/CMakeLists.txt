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
    op_kernels
  HDRS
    "op_kernels.h"
  TEXTUAL_HDRS
    "op_kernels_generic.h"
    "op_kernels_ruy.h"
  DEPS
    absl::algorithm
    absl::core_headers
    absl::flat_hash_set
    absl::inlined_vector
    absl::memory
    absl::span
    iree::base::status
    iree::base::tracing
    ruy
  PUBLIC
)

iree_cc_test(
  NAME
    op_kernels_test
  SRCS
    "op_kernels_test.cc"
  DEPS
    ::op_kernels
    absl::inlined_vector
    iree::base::memory
    iree::base::status_matchers
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    vmla_cache
  HDRS
    "vmla_cache.h"
  SRCS
    "vmla_cache.cc"
  DEPS
    ::vmla_executable
    iree::base::source_location
    iree::base::status
    iree::base::tracing
    iree::hal::allocator
    iree::hal::executable
    iree::hal::executable_cache
    iree::hal::executable_format
    iree::vm::instance
    iree::vm::module
  PUBLIC
)

iree_cc_library(
  NAME
    vmla_command_processor
  HDRS
    "vmla_command_processor.h"
  SRCS
    "vmla_command_processor.cc"
  DEPS
    ::vmla_executable
    ::vmla_module
    iree::base::api_util
    iree::base::status
    iree::base::tracing
    iree::hal::host::host_buffer
    iree::hal::host::serial_command_processor
    iree::vm::invocation
    iree::vm::stack
    iree::vm::variant_list
  PUBLIC
)

iree_cc_library(
  NAME
    vmla_device
  HDRS
    "vmla_device.h"
  SRCS
    "vmla_device.cc"
  DEPS
    ::vmla_cache
    ::vmla_command_processor
    absl::inlined_vector
    absl::memory
    absl::span
    absl::strings
    iree::base::memory
    iree::base::status
    iree::base::tracing
    iree::hal::command_buffer_validation
    iree::hal::command_queue
    iree::hal::device
    iree::hal::host::async_command_queue
    iree::hal::host::host_descriptor_set
    iree::hal::host::host_event
    iree::hal::host::host_executable_layout
    iree::hal::host::host_local_allocator
    iree::hal::host::host_semaphore
    iree::hal::host::inproc_command_buffer
    iree::hal::host::serial_submission_queue
    iree::vm::instance
    iree::vm::module
  PUBLIC
)

iree_cc_library(
  NAME
    vmla_driver
  HDRS
    "vmla_driver.h"
  SRCS
    "vmla_driver.cc"
  DEPS
    ::vmla_device
    ::vmla_module
    iree::base::api_util
    iree::base::tracing
    iree::hal::device_info
    iree::hal::driver
    iree::vm::instance
    iree::vm::module
  PUBLIC
)

iree_cc_library(
  NAME
    vmla_driver_module
  SRCS
    "vmla_driver_module.cc"
  DEPS
    ::vmla_driver
    iree::base::init
    iree::base::status
    iree::hal::driver_registry
  ALWAYSLINK
  PUBLIC
)

iree_cc_library(
  NAME
    vmla_executable
  HDRS
    "vmla_executable.h"
  SRCS
    "vmla_executable.cc"
  DEPS
    ::vmla_module
    absl::inlined_vector
    absl::span
    iree::base::api_util
    iree::base::status
    iree::base::tracing
    iree::hal::allocator
    iree::hal::executable
    iree::hal::executable_spec
    iree::schemas::vmla_executable_def_cc_fbs
    iree::vm::bytecode_module
    iree::vm::context
    iree::vm::instance
    iree::vm::module
    iree::vm::variant_list
  PUBLIC
)

iree_cc_library(
  NAME
    vmla_module
  HDRS
    "vmla_module.h"
  SRCS
    "vmla_module.cc"
  DEPS
    ::op_kernels
    absl::span
    iree::base::api
    iree::base::memory
    iree::base::ref_ptr
    iree::base::status
    iree::base::tracing
    iree::vm
    iree::vm::module_abi_cc
    iree::vm::types
  PUBLIC
)
