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

# bazel_to_cmake: DO NOT EDIT, IREE_HAL_DRIVER_VMLA filtering is custom logic
if(${IREE_HAL_DRIVER_VMLA})
  iree_cc_test(
    NAME
      check_test
    SRCS
      "check_test.cc"
    DEPS
      ::native_module
      absl::core_headers
      absl::strings
      iree::base::api
      iree::base::logging
      iree::base::status
      iree::hal::api
      iree::hal::vmla::vmla_driver_module
      iree::modules::hal
      iree::testing::gtest
      iree::testing::gtest_main
      iree::vm
      iree::vm::bytecode_module
      iree::vm::ref_cc
  )
endif()

iree_cc_binary(
  NAME
    iree-check-module
  OUT
    iree-check-module
  SRCS
    "check_module_main.cc"
  DEPS
    ::native_module
    absl::flags
    absl::strings
    iree::base::api
    iree::base::file_io
    iree::base::init
    iree::base::localfile
    iree::base::status
    iree::base::target_platform
    iree::base::tracing
    iree::modules::hal
    iree::testing::gtest
    iree::tools::vm_util
    iree::vm::bytecode_module
    ${IREE_HAL_DRIVER_MODULES}
  TESTONLY
)

iree_cc_library(
  NAME
    native_module
  HDRS
    "native_module.h"
  SRCS
    "native_module.cc"
  DEPS
    absl::inlined_vector
    absl::strings
    iree::base::api
    iree::base::status
    iree::hal::api
    iree::modules::hal
    iree::testing::gtest
    iree::vm
    iree::vm::module_abi_cc
  TESTONLY
  PUBLIC
)
