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

iree_cc_library(
  NAME
    api
  HDRS
    "api.h"
  SRCS
    "api.cc"
  DEPS
    iree::base::api_util
    iree::base::file_mapping
    iree::base::tracing
  PUBLIC
)

iree_cc_library(
  NAME
    api_util
  HDRS
    "api.h"
    "api_util.h"
  DEPS
    iree::base::status
  PUBLIC
)

iree_cc_library(
  NAME
    arena
  HDRS
    "arena.h"
  SRCS
    "arena.cc"
  DEPS
    absl::span
    iree::base::logging
  PUBLIC
)

iree_cc_test(
  NAME
    arena_test
  SRCS
    "arena_test.cc"
  DEPS
    gtest_main
    iree::base::arena
)

iree_cc_library(
  NAME
    bitfield
  HDRS
    "bitfield.h"
  DEPS
    absl::span
    iree::base::bitfield
  PUBLIC
)

iree_cc_test(
  NAME
    bitfield_test
  SRCS
    "bitfield_test.cc"
  DEPS
    absl::base
    gtest_main
    iree::base::bitfield
)

iree_cc_library(
  NAME
    file_handle
  SRCS
    "internal/file_handle_win32.cc"
  HDRS
    "internal/file_handle_win32.h"
  DEPS
    absl::memory
    absl::strings
    iree::base::status
    iree::base::tracing
    iree::base::target_platform
  PUBLIC
)

iree_cc_library(
  NAME
    file_io
  SRCS
    "internal/file_io_posix.cc"
    "internal/file_io_win32.cc"
  HDRS
    "file_io.h"
  DEPS
    absl::memory
    absl::strings
    absl::span
    iree::base::file_handle
    iree::base::ref_ptr
    iree::base::status
    iree::base::target_platform
  PUBLIC
)

iree_cc_library(
  NAME
    file_mapping
  SRCS
    "internal/file_mapping_posix.cc"
    "internal/file_mapping_win32.cc"
  HDRS
    "file_mapping.h"
  DEPS
    absl::memory
    absl::strings
    absl::span
    iree::base::file_handle
    iree::base::ref_ptr
    iree::base::status
    iree::base::tracing
    iree::base::target_platform
  PUBLIC
)

iree_cc_library(
  NAME
    file_path
  SRCS
    "file_path.cc"
  HDRS
    "file_path.h"
  DEPS
    absl::strings
  PUBLIC
)

iree_cc_library(
  NAME
    flatbuffer_util
  HDRS
    "flatbuffer_util.h"
  SRCS
    "flatbuffer_util.cc"
  DEPS
    absl::memory
    absl::optional
    absl::span
    absl::strings
    flatbuffers
    iree::base::memory
    iree::base::status
    iree::base::tracing
  PUBLIC
)

iree_cc_library(
  NAME
    init
  HDRS
    "init.h"
    "internal/init_internal.h"
  SRCS
    "internal/init_internal.cc"
  DEPS
    absl::flags
    iree::base::target_platform
  PUBLIC
)

iree_cc_library(
  NAME
    intrusive_list
  HDRS
    "intrusive_list.h"
    "intrusive_list_ref_ptr.inc"
    "intrusive_list_unique_ptr.inc"
  DEPS
    iree::base::logging
    iree::base::ref_ptr
  PUBLIC
)

iree_cc_test(
  NAME
    intrusive_list_test
  SRCS
    "intrusive_list_test.cc"
  DEPS
    gtest_main
    iree::base::intrusive_list
)

iree_cc_library(
  NAME
    logging
  HDRS
    "logging.h"
    "internal/logging.h"
  SRCS
    "internal/logging.cc"
  DEPS
    absl::base
    absl::flags
  PUBLIC
)

iree_cc_library(
  NAME
    math
  HDRS
    "math.h"
  DEPS
    absl::core_headers
  PUBLIC
)

iree_cc_library(
  NAME
    memory
  HDRS
    "memory.h"
  DEPS
    absl::span
  PUBLIC
)

iree_cc_library(
  NAME
    ref_ptr
  HDRS
    "ref_ptr.h"
  DEPS
    absl::base
    iree::base::logging
  PUBLIC
)

iree_cc_test(
  NAME
    ref_ptr_test
  SRCS
    "ref_ptr_test.cc"
  DEPS
    gtest_main
    iree::base::ref_ptr
)

iree_cc_library(
  NAME
    shape
  HDRS
    "shape.h"
  SRCS
    "shape.cc"
  DEPS
    absl::base
    absl::span
    absl::strings
    absl::type_traits
    iree::base::logging
    iree::base::status
  PUBLIC
)

iree_cc_test(
  NAME
    shape_test
  SRCS
    "shape_test.cc"
  DEPS
    absl::span
    absl::strings
    gtest_main
    iree::base::logging
    iree::base::shape
    iree::base::status
    iree::base::status_matchers
)

iree_cc_library(
  NAME
    source_location
  HDRS
    "source_location.h"
    "internal/source_location.h"
  PUBLIC
)

iree_cc_library(
  NAME
    status
  HDRS
    "status.h"
    "internal/status.h"
    "internal/status_builder.h"
    "internal/status_errno.h"
    "internal/status_errors.h"
    "internal/status_macros.h"
    "internal/status_win32_errors.h"
    "internal/statusor.h"
  SRCS
    "internal/status.cc"
    "internal/status_builder.cc"
    "internal/status_errno.cc"
    "internal/status_errors.cc"
    "internal/status_win32_errors.cc"
    "internal/statusor.cc"
  DEPS
    absl::base
    absl::debugging
    absl::flags
    absl::memory
    absl::strings
    iree::base::logging
    iree::base::source_location
  PUBLIC
)

iree_cc_library(
  NAME
    status_matchers
  HDRS
    "status_matchers.h"
    "internal/status_matchers.h"
  DEPS
    absl::strings
    absl::optional
    gmock
    iree::base::status
  TESTONLY
  PUBLIC
)

iree_cc_library(
  NAME
    target_platform
  HDRS
    "target_platform.h"
  PUBLIC
)

iree_cc_library(
  NAME
    time
  HDRS
    "time.h"
  DEPS
    absl::time
  PUBLIC
)

if(${IREE_ENABLE_TRACING})
  iree_cc_library(
    NAME
      tracing
    HDRS
      "tracing.h"
    SRCS
      "tracing.cc"
    DEPS
      absl::core_headers
      absl::flags
      absl::strings
      absl::synchronization
      absl::time
      iree::base::file_io
      iree::base::init
      iree::base::logging
      iree::base::status
    PUBLIC
  )
else()
  iree_cc_library(
    NAME
      tracing
    HDRS
      "tracing.h"
    SRCS
      "tracing_disabled.cc"
    DEPS
      absl::flags
    PUBLIC
  )
endif()

# TODO(benvanik): get wait_handle ported to win32.
# iree_cc_library(
#   NAME
#     wait_handle
#   HDRS
#     "wait_handle.h"
#   SRCS
#     "wait_handle.cc"
#   DEPS
#     absl::base
#     absl::fixed_array
#     absl::span
#     absl::strings
#     absl::time
#     iree::base::logging
#     iree::base::ref_ptr
#     iree::base::status
#     iree::base::time
#   PUBLIC
# )
#
# iree_cc_test(
#   NAME
#     wait_handle_test
#   SRCS
#     "wait_handle_test.cc"
#   DEPS
#     absl::time
#     gtest_main
#     iree::base::status
#     iree::base::status_matchers
#     iree::base::wait_handle
# )

