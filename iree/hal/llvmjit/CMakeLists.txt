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

if(NOT ${IREE_HAL_DRIVER_LLVM})
  return()
endif()

iree_add_all_subdirs()

iree_cc_library(
  NAME
    llvmjit
  HDRS
    "llvmjit_device.h"
    "llvmjit_driver.h"
    "llvmjit_executable.h"
    "llvmjit_executable_cache.h"
  SRCS
    "llvmjit_device.cc"
    "llvmjit_driver.cc"
    "llvmjit_executable.cc"
    "llvmjit_executable_cache.cc"
  DEPS
    LLVMAsmParser
    LLVMCore
    LLVMExecutionEngine
    LLVMOrcJIT
    LLVMSupport
    absl::span
    iree::base::flatcc
    iree::base::status
    iree::base::tracing
    iree::hal
    iree::hal::host::host_executable
    iree::hal::host::host_local_device
    iree::hal::host::serial::serial_scheduling_model
    iree::schemas::llvmir_executable_def_c_fbs
  PUBLIC
)

iree_cc_library(
  NAME
    llvmjit_driver_module
  SRCS
    "llvmjit_driver_module.cc"
  DEPS
    ::llvmjit
    LLVMSupport
    LLVMX86CodeGen
    iree::base::init
    iree::base::status
    iree::hal::driver_registry
  ALWAYSLINK
  PUBLIC
)
