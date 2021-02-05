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
    vm_util
  HDRS
    "vm_util.h"
  SRCS
    "vm_util.cc"
  DEPS
    absl::span
    absl::strings
    iree::base::internal::file_io
    iree::base::signature_mangle
    iree::base::status
    iree::hal::api
    iree::modules::hal
    iree::vm
    iree::vm::bytecode_module
    iree::vm::cc
  PUBLIC
)

iree_cc_test(
  NAME
    vm_util_test
  SRCS
    "vm_util_test.cc"
  DEPS
    ::vm_util
    absl::strings
    iree::base::api
    iree::hal::api
    iree::hal::vmla::registration
    iree::modules::hal
    iree::testing::gtest
    iree::testing::gtest_main
    iree::vm
)
