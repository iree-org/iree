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
    atomic_slist
  HDRS
    "atomic_slist.h"
  SRCS
    "atomic_slist.c"
  DEPS
    iree::base::core_headers
    iree::base::synchronization
  PUBLIC
)

iree_cc_test(
  NAME
    atomic_slist_test
  SRCS
    "atomic_slist_test.cc"
  DEPS
    ::atomic_slist
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    atomics
  HDRS
    "atomics_clang.h"
    "atomics_gcc.h"
    "atomics_msvc.h"
  DEPS
    iree::base::target_platform
  PUBLIC
)

iree_cc_library(
  NAME
    file_handle_win32
  HDRS
    "file_handle_win32.h"
  SRCS
    "file_handle_win32.cc"
  DEPS
    absl::memory
    absl::strings
    iree::base::core_headers
    iree::base::status
  PUBLIC
)

iree_cc_library(
  NAME
    file_io_internal
  SRCS
    "file_io_posix.cc"
    "file_io_win32.cc"
  DEPS
    ::file_handle_win32
    absl::memory
    absl::strings
    iree::base::core_headers
    iree::base::file_io_hdrs
    iree::base::file_path
    iree::base::status
    iree::base::tracing
  PUBLIC
)

iree_cc_library(
  NAME
    main
  HDRS
    "main.h"
  SRCS
    "main_posix.cc"
    "main_win32.cc"
  DEPS
    iree::base::core_headers
    iree::base::logging
  PUBLIC
)

iree_cc_library(
  NAME
    ostringstream
  HDRS
    "ostringstream.h"
  SRCS
    "ostringstream.cc"
  PUBLIC
)

iree_cc_library(
  NAME
    prng
  HDRS
    "prng.h"
  DEPS
    iree::base::core_headers
  PUBLIC
)

iree_cc_test(
  NAME
    prng_test
  SRCS
    "prng_test.cc"
  DEPS
    ::prng
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    status_internal
  HDRS
    "status.h"
    "status_builder.h"
    "statusor.h"
  SRCS
    "status.cc"
    "status_builder.cc"
    "statusor.cc"
  DEPS
    ::ostringstream
    absl::core_headers
    absl::strings
    absl::utility
    iree::base::api
    iree::base::core_headers
    iree::base::logging
  PUBLIC
)
