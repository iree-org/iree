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

# Doesn't use bazel_to_cmake because IREE_HAL_DRIVER_VMLA filtering is custom logic
if(${IREE_HAL_DRIVER_VMLA})
  iree_cc_test(
    NAME
      check_test
    SRCS
      "check_test.cc"
    DEPS
      ::native_module
      absl::span
      iree::base
      iree::base::internal
      iree::base::logging
      iree::base::status
      iree::hal
      iree::hal::vmvx::registration
      iree::modules::hal
      iree::testing::gtest
      iree::testing::gtest_main
      iree::vm
      iree::vm::bytecode_module
      iree::vm::cc
  )
endif()


iree_cc_library(
  NAME
    native_module
  HDRS
    "native_module.h"
  SRCS
    "native_module.cc"
  DEPS
    iree::base
    iree::base::internal
    iree::base::status
    iree::hal
    iree::modules::hal
    iree::testing::gtest
    iree::vm
    iree::vm::cc
  TESTONLY
  PUBLIC
)
