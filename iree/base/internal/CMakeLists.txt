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
    internal
  HDRS
    "atomics.h"
    "debugging.h"
    "math.h"
  SRCS
    "atomics_clang.h"
    "atomics_gcc.h"
    "atomics_msvc.h"
  DEPS
    iree::base::core_headers
  PUBLIC
)

iree_cc_test(
  NAME
    atomics_test
  SRCS
    "atomics_test.cc"
  DEPS
    ::internal
    iree::base::core_headers
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_test(
  NAME
    math_test
  SRCS
    "math_test.cc"
  DEPS
    ::internal
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    atomic_slist
  HDRS
    "atomic_slist.h"
  SRCS
    "atomic_slist.c"
  DEPS
    ::internal
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
    file_io
  HDRS
    "file_io.h"
  DEPS
    absl::memory
    absl::span
    absl::strings
    iree::base::core_headers
    iree::base::internal::file_io_internal
    iree::base::status
  PUBLIC
)

iree_cc_library(
  NAME
    file_io_hdrs
  HDRS
    "file_io.h"
  DEPS
    absl::strings
    iree::base::status
  PUBLIC
)

iree_cc_test(
  NAME
    file_io_test
  SRCS
    "file_io_test.cc"
  DEPS
    ::file_io
    ::file_path
    absl::strings
    iree::base::logging
    iree::base::status
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    file_io_internal
  SRCS
    "file_io_posix.cc"
    "file_io_win32.cc"
  DEPS
    ::file_handle_win32
    ::file_io_hdrs
    ::file_path
    absl::memory
    absl::strings
    iree::base::core_headers
    iree::base::status
    iree::base::tracing
  PUBLIC
)

iree_cc_library(
  NAME
    file_path
  HDRS
    "file_path.h"
  SRCS
    "file_path.cc"
  DEPS
    absl::strings
  PUBLIC
)

iree_cc_test(
  NAME
    file_path_test
  SRCS
    "file_path_test.cc"
  DEPS
    ::file_path
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    flags
  HDRS
    "flags.h"
  SRCS
    "flags.cc"
  DEPS
    absl::flags_parse
    iree::base::api
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
    prng
  HDRS
    "prng.h"
  DEPS
    ::internal
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
    "statusor.h"
  SRCS
    "status.cc"
    "statusor.cc"
  DEPS
    absl::core_headers
    absl::strings
    absl::utility
    iree::base::api
    iree::base::core_headers
    iree::base::logging
  PUBLIC
)

iree_cc_library(
  NAME
    wait_handle
  HDRS
    "wait_handle.h"
  SRCS
    "wait_handle.c"
    "wait_handle_epoll.c"
    "wait_handle_impl.h"
    "wait_handle_kqueue.c"
    "wait_handle_poll.c"
    "wait_handle_posix.c"
    "wait_handle_posix.h"
    "wait_handle_win32.c"
  DEPS
    iree::base::api
    iree::base::core_headers
    iree::base::tracing
  PUBLIC
)

iree_cc_test(
  NAME
    wait_handle_test
  SRCS
    "wait_handle_test.cc"
  DEPS
    ::wait_handle
    iree::testing::gtest
    iree::testing::gtest_main
)
