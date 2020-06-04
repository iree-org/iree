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
    dylib_command_processor
  HDRS
    "dylib_command_processor.h"
  SRCS
    "dylib_command_processor.cc"
  DEPS
    ::dylib_executable
    ::memref_runtime
    absl::inlined_vector
    iree::base::tracing
    iree::hal::buffer
    iree::hal::host::host_local_command_processor
  PUBLIC
)

iree_cc_library(
  NAME
    dylib_device
  HDRS
    "dylib_device.h"
  SRCS
    "dylib_device.cc"
  DEPS
    ::dylib_command_processor
    ::dylib_executable_cache
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
    iree::hal::host::host_submission_queue
    iree::hal::host::inproc_command_buffer
    iree::hal::semaphore
  PUBLIC
)

iree_cc_library(
  NAME
    dylib_driver
  HDRS
    "dylib_driver.h"
  SRCS
    "dylib_driver.cc"
  DEPS
    ::dylib_device
    iree::hal::device_info
    iree::hal::driver
  PUBLIC
)

iree_cc_library(
  NAME
    dylib_driver_module
  SRCS
    "dylib_driver_module.cc"
  DEPS
    ::dylib_driver
    iree::base::init
    iree::base::status
    iree::hal::driver_registry
  ALWAYSLINK
  PUBLIC
)

iree_cc_library(
  NAME
    dylib_executable
  HDRS
    "dylib_executable.h"
  SRCS
    "dylib_executable.cc"
  DEPS
    absl::inlined_vector
    absl::span
    flatbuffers
    iree::base::dynamic_library
    iree::base::file_io
    iree::base::status
    iree::hal::executable
    iree::hal::executable_spec
    iree::schemas::dylib_executable_def_cc_fbs
  PUBLIC
)

iree_cc_library(
  NAME
    dylib_executable_cache
  HDRS
    "dylib_executable_cache.h"
  SRCS
    "dylib_executable_cache.cc"
  DEPS
    ::dylib_executable
    iree::base::source_location
    iree::base::status
    iree::base::tracing
    iree::hal::executable
    iree::hal::executable_cache
    iree::hal::executable_format
  PUBLIC
)

iree_cc_library(
  NAME
    memref_runtime
  HDRS
    "memref_runtime.h"
  PUBLIC
)
