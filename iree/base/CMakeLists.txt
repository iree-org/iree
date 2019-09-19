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
    base_arena
  HDRS
    "arena.h"
  SRCS
    "arena.cc"
  DEPS
    absl::span
    iree::base_logging
  PUBLIC
)

iree_cc_test(
  NAME
    base_arena_test
  SRCS
    "arena_test.cc"
  DEPS
    iree::base_arena
    gtest_main
)

iree_cc_library(
  NAME
    base_bitfield
  HDRS
    "bitfield.h"
  DEPS
    absl::span
    iree::base_bitfield
  PUBLIC
)

iree_cc_test(
  NAME
    base_bitfield_test
  SRCS
    "bitfield_test.cc"
  DEPS
    absl::base
    iree::base_bitfield
    gtest_main
)

iree_cc_library(
  NAME
    base_file_io
  HDRS
    "file_io.h"
  DEPS
    absl::strings
    iree::base_status
  PUBLIC
)

iree_cc_library(
  NAME
    base_flatbuffer_util
  HDRS
    "flatbuffer_util.h"
  SRCS
    "flatbuffer_util.cc"
  DEPS
    absl::memory
    absl::optional
    absl::span
    absl::strings
    iree::base_memory
    iree::base_status
    iree::base_tracing
  PUBLIC
)

iree_cc_library(
  NAME
    base_init
  HDRS
    "init.h"
    "internal/init_internal.h"
  SRCS
    "internal/init_internal.cc"
  DEPS
    absl::flags
  PUBLIC
)

iree_cc_library(
  NAME
    base_intrusive_list
  HDRS
    "intrusive_list.h"
    "intrusive_list_ref_ptr.inc"
    "intrusive_list_unique_ptr.inc"
  DEPS
    iree::base_logging
    iree::base_ref_ptr
  PUBLIC
)

iree_cc_test(
  NAME
    base_intrusive_list_test
  SRCS
    "intrusive_list_test.cc"
  DEPS
    iree::base_intrusive_list
    gtest_main
)

iree_cc_library(
  NAME
    base_logging
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
    base_memory
  HDRS
    "memory.h"
  DEPS
    absl::span
  PUBLIC
)

iree_cc_library(
  NAME
    base_ref_ptr
  HDRS
    "ref_ptr.h"
  DEPS
    absl::base
    iree::base_logging
  PUBLIC
)

iree_cc_test(
  NAME
    base_ref_ptr_test
  SRCS
    "ref_ptr_test.cc"
  DEPS
    iree::base_ref_ptr
    gtest_main
)

iree_cc_library(
  NAME
    base_shape
  HDRS
    "shape.h"
  SRCS
    "shape.cc"
  DEPS
    absl::base
    absl::span
    absl::strings
    absl::type_traits
    iree::base_logging
    iree::base_status
  PUBLIC
)

iree_cc_test(
  NAME
    base_shape_test
  SRCS
    "shape_test.cc"
  DEPS
    absl::span
    absl::strings
    iree::base_logging
    iree::base_shape
    iree::base_status
    gtest_main
)

iree_cc_library(
  NAME
    base_status
  HDRS
    "status.h"
    "internal/status.h"
    "internal/status_builder.h"
    "internal/status_errno.h"
    "internal/status_errors.h"
    "internal/status_macros.h"
    "internal/statusor.h"
  SRCS
    "internal/status.cc"
    "internal/status_builder.cc"
    "internal/status_errno.cc"
    "internal/status_errors.cc"
    "internal/statusor.cc"
  DEPS
    absl::base
    absl::debugging
    absl::flags
    absl::memory
    absl::strings
    iree::base_logging
  PUBLIC
)

iree_cc_library(
  NAME
    base_time
  HDRS
    "time.h"
  DEPS
    absl::time
  PUBLIC
)

iree_cc_library(
  NAME
    base_tracing
  HDRS
    "tracing.h"
  SRCS
    # TODO(benvanik): implement tracing.
    "tracing_disabled.cc"
  DEPS
    absl::flags
    iree::base_logging
  PUBLIC
)

iree_cc_library(
  NAME
    base_wait_handle
  HDRS
    "wait_handle.h"
  SRCS
    "wait_handle.cc"
  DEPS
    absl::base
    absl::fixed_array
    absl::span
    absl::strings
    absl::time
    iree::base_logging
    iree::base_ref_ptr
    iree::base_status
    iree::base_time
  PUBLIC
)

iree_cc_test(
  NAME
    base_wait_handle_test
  SRCS
    "wait_handle_test.cc"
  DEPS
    absl::time
    iree::base_wait_handle
    iree::base_status
    gtest_main
)
