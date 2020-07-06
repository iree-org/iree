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
    llvmjit_device
  HDRS
    "llvmjit_device.h"
  SRCS
    "llvmjit_device.cc"
  DEPS
    ::llvmjit_executable_cache
    iree::base::tracing
    iree::hal::host::host_local_device
  PUBLIC
)

iree_cc_library(
  NAME
    llvmjit_driver
  HDRS
    "llvmjit_driver.h"
  SRCS
    "llvmjit_driver.cc"
  DEPS
    ::llvmjit_device
    LLVMExecutionEngine
    iree::hal::device_info
    iree::hal::driver
    iree::hal::host::serial::serial_scheduling_model
  PUBLIC
)

iree_cc_library(
  NAME
    llvmjit_driver_module
  SRCS
    "llvmjit_driver_module.cc"
  DEPS
    ::llvmjit_driver
    LLVMSupport
    LLVMX86CodeGen
    iree::base::init
    iree::base::status
    iree::hal::driver_registry
  ALWAYSLINK
  PUBLIC
)

iree_cc_library(
  NAME
    llvmjit_executable
  HDRS
    "llvmjit_executable.h"
  SRCS
    "llvmjit_executable.cc"
  DEPS
    ::memref_runtime
    LLVMAsmParser
    LLVMCore
    LLVMOrcJIT
    LLVMSupport
    absl::span
    flatbuffers
    iree::base::status
    iree::base::tracing
    iree::hal::buffer
    iree::hal::executable
    iree::hal::executable_spec
    iree::hal::host::host_executable
    iree::schemas::llvmir_executable_def_cc_fbs
  PUBLIC
)

iree_cc_library(
  NAME
    llvmjit_executable_cache
  HDRS
    "llvmjit_executable_cache.h"
  SRCS
    "llvmjit_executable_cache.cc"
  DEPS
    ::llvmjit_executable
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
