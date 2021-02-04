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
    arena
  HDRS
    "arena.h"
  SRCS
    "arena.cc"
  DEPS
    absl::core_headers
    absl::span
    iree::base::core_headers
    iree::base::logging
  PUBLIC
)

iree_cc_test(
  NAME
    arena_test
  SRCS
    "arena_test.cc"
  DEPS
    ::arena
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    intrusive_list
  HDRS
    "intrusive_list.h"
    "intrusive_list_unique_ptr.inc"
  DEPS
    iree::base::logging
  PUBLIC
)

iree_cc_test(
  NAME
    intrusive_list_test
  SRCS
    "intrusive_list_test.cc"
    "intrusive_list_unique_ptr_test.cc"
  DEPS
    ::intrusive_list
    absl::memory
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    ref_ptr
  HDRS
    "ref_ptr.h"
  DEPS
    absl::core_headers
    iree::base::internal
    iree::base::logging
  PUBLIC
)

iree_cc_test(
  NAME
    ref_ptr_test
  SRCS
    "ref_ptr_test.cc"
  DEPS
    ::ref_ptr
    iree::testing::gtest
    iree::testing::gtest_main
)
