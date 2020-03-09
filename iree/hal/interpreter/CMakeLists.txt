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
    bytecode_cache
  HDRS
    "bytecode_cache.h"
  SRCS
    "bytecode_cache.cc"
  DEPS
    ::bytecode_executable
    iree::base::source_location
    iree::base::status
    iree::base::tracing
    iree::hal::allocator
    iree::hal::executable
    iree::hal::executable_cache
    iree::hal::executable_format
  PUBLIC
)

iree_cc_library(
  NAME
    bytecode_executable
  HDRS
    "bytecode_dispatch.h"
    "bytecode_executable.h"
    "bytecode_reader.h"
    "bytecode_tables_interpreter.h"
    "interpreter_module.h"
    "stack.h"
    "type.h"
  SRCS
    "bytecode_dispatch.cc"
    "bytecode_dispatch_conversion.h"
    "bytecode_dispatch_util.cc"
    "bytecode_dispatch_util.h"
    "bytecode_executable.cc"
    "bytecode_reader.cc"
    "bytecode_tables_interpreter.cc"
    "interpreter_module.cc"
    "stack.cc"
    "type.cc"
  DEPS
    ::bytecode_kernels
    absl::core_headers
    absl::inlined_vector
    absl::optional
    absl::span
    absl::strings
    iree::base::flatbuffer_util
    iree::base::logging
    iree::base::memory
    iree::base::ref_ptr
    iree::base::shape
    iree::base::status
    iree::base::tracing
    iree::hal::allocator
    iree::hal::buffer_view
    iree::hal::executable
    iree::hal::executable_spec
    iree::hal::heap_buffer
    iree::schemas::bytecode::interpreter_bytecode_v0
    iree::schemas::interpreter_module_def_cc_fbs
  PUBLIC
)

iree_cc_library(
  NAME
    bytecode_kernels
  HDRS
    "bytecode_kernels.h"
  TEXTUAL_HDRS
    "bytecode_kernels_generic.h"
    "bytecode_kernels_ruy.h"
  DEPS
    absl::algorithm
    absl::core_headers
    absl::flat_hash_set
    absl::inlined_vector
    absl::memory
    absl::span
    iree::base::shape
    iree::base::status
    iree::base::tracing
    iree::hal::buffer_view
    ruy
  PUBLIC
)

iree_cc_test(
  NAME
    bytecode_kernels_test
  SRCS
    "bytecode_kernels_test.cc"
  DEPS
    ::bytecode_kernels
    iree::base::memory
    iree::base::status_matchers
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    interpreter_command_processor
  HDRS
    "interpreter_command_processor.h"
  SRCS
    "interpreter_command_processor.cc"
  DEPS
    ::bytecode_executable
    absl::inlined_vector
    absl::span
    iree::base::source_location
    iree::base::status
    iree::base::tracing
    iree::hal::buffer_view
    iree::hal::host::host_local_command_processor
  PUBLIC
)

iree_cc_library(
  NAME
    interpreter_device
  HDRS
    "interpreter_device.h"
  SRCS
    "interpreter_device.cc"
  DEPS
    ::bytecode_cache
    ::bytecode_kernels
    ::interpreter_command_processor
    absl::inlined_vector
    absl::memory
    absl::span
    iree::base::memory
    iree::base::status
    iree::base::tracing
    iree::hal::command_buffer_validation
    iree::hal::command_queue
    iree::hal::device
    iree::hal::fence
    iree::hal::host::async_command_queue
    iree::hal::host::host_event
    iree::hal::host::host_local_allocator
    iree::hal::host::host_submission_queue
    iree::hal::host::inproc_command_buffer
  PUBLIC
)

iree_cc_library(
  NAME
    interpreter_driver
  HDRS
    "interpreter_driver.h"
  SRCS
    "interpreter_driver.cc"
  DEPS
    ::interpreter_device
    iree::hal::device_info
    iree::hal::driver
  PUBLIC
)

iree_cc_library(
  NAME
    interpreter_driver_module
  SRCS
    "interpreter_driver_module.cc"
  DEPS
    ::interpreter_driver
    iree::base::init
    iree::base::status
    iree::hal::driver_registry
  ALWAYSLINK
  PUBLIC
)
