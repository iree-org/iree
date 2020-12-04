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

if(NOT ${IREE_HAL_DRIVER_VMLA})
  return()
endif()

iree_add_all_subdirs()

iree_cc_library(
  NAME
    op_kernels
  HDRS
    "op_kernels.h"
  TEXTUAL_HDRS
    "op_kernels_fft.h"
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
    pffft
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
    iree::base::core_headers
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    op_module
  HDRS
    "op_module.h"
  SRCS
    "op_module.cc"
  DEPS
    ::op_kernels
    absl::span
    iree::base::api
    iree::base::core_headers
    iree::base::ref_ptr
    iree::base::status
    iree::base::tracing
    iree::vm
    iree::vm::cc
  PUBLIC
)

iree_cc_library(
  NAME
    vmla
  HDRS
    "vmla_cache.h"
    "vmla_device.h"
    "vmla_driver.h"
    "vmla_executable.h"
  SRCS
    "vmla_cache.cc"
    "vmla_device.cc"
    "vmla_driver.cc"
    "vmla_executable.cc"
  DEPS
    ::op_module
    absl::inlined_vector
    absl::memory
    absl::span
    absl::strings
    iree::base::api
    iree::base::core_headers
    iree::base::flatcc
    iree::base::ref_ptr
    iree::base::status
    iree::base::tracing
    iree::hal
    iree::hal::host::host_buffer
    iree::hal::host::host_executable
    iree::hal::host::host_local_device
    iree::hal::host::serial::serial_scheduling_model
    iree::schemas::vmla_executable_def_c_fbs
    iree::vm
    iree::vm::bytecode_module
  PUBLIC
)
