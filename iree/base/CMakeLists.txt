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

add_subdirectory(internal)

iree_cc_library(
  NAME
    api
  HDRS
    "api.h"
  SRCS
    "api.cc"
  DEPS
    iree::base::api_hdrs
    iree::base::api_util
    iree::base::file_mapping
    iree::base::init
    iree::base::tracing
  PUBLIC
)

iree_cc_library(
  NAME
    api_hdrs
  HDRS
    "api.h"
  PUBLIC
)

iree_cc_library(
  NAME
    api_util
  HDRS
    "api_util.h"
  DEPS
    absl::base
    absl::container
    absl::time
    iree::base::api_hdrs
    iree::base::logging
    iree::base::shape
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
    absl::base
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
    iree::testing::gtest_main
    iree::base::arena
)

iree_cc_library(
  NAME
    bitfield
  HDRS
    "bitfield.h"
  DEPS
    absl::span
  PUBLIC
)

iree_cc_test(
  NAME
    bitfield_test
  SRCS
    "bitfield_test.cc"
  DEPS
    absl::core_headers
    iree::testing::gtest_main
    iree::base::bitfield
)

iree_cc_library(
  NAME
    "buffer_string_util"
  SRCS
    "buffer_string_util.cc"
  HDRS
    "buffer_string_util.h"
  DEPS
    absl::flat_hash_map
    absl::strings
    absl::optional
    absl::span
    iree::base::memory
    iree::base::shape
    iree::base::source_location
    iree::base::status
)

iree_cc_test(
  NAME
    "buffer_string_util_test"
  SRCS
    "buffer_string_util_test.cc"
  DEPS
    iree::testing::gtest_main
    iree::base::buffer_string_util
    iree::base::memory
    iree::base::status
    iree::base::status_matchers
    absl::strings
)

iree_cc_library(
  NAME
    file_io
  HDRS
    "file_io.h"
  DEPS
    absl::memory
    absl::strings
    absl::span
    iree::base::status
    iree::base::target_platform
    iree::base::internal::file_io_internal
  PUBLIC
)

iree_cc_library(
  NAME
    file_io_hdrs
  HDRS
    "file_io.h"
  DEPS
    iree::base::status
)

iree_cc_test(
  NAME
    file_io_test
  SRCS
    "file_io_test.cc"
  DEPS
    ::file_io
    ::file_path
    ::status
    ::status_matchers
    absl::strings
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    file_mapping
  HDRS
    "file_mapping.h"
  DEPS
    absl::memory
    absl::strings
    absl::span
    iree::base::ref_ptr
    iree::base::status
    iree::base::internal::file_mapping_internal
  PUBLIC
)

iree_cc_library(
  NAME
    file_mapping_hdrs
  HDRS
    "file_mapping.h"
  DEPS
    absl::span
    iree::base::ref_ptr
    iree::base::status
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

iree_cc_test(
  NAME
    file_path_test
  SRCS
    "file_path_test.cc"
  DEPS
    iree::base::file_path
    iree::testing::gtest_main
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
    iree::base::file_mapping
    iree::base::memory
    iree::base::ref_ptr
    iree::base::source_location
    iree::base::status
    iree::base::tracing
  PUBLIC
)

iree_cc_library(
  NAME
    init
  HDRS
    "init.h"
  DEPS
    iree::base::internal::init_internal
  PUBLIC
)

iree_cc_library(
  NAME
    initializer
  HDRS
    "initializer.h"
  SRCS
    "initializer.cc"
  DEPS
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
    "intrusive_list_ref_ptr_test.cc"
    "intrusive_list_test.cc"
    "intrusive_list_unique_ptr_test.cc"
  DEPS
    absl::memory
    iree::testing::gtest_main
    iree::base::intrusive_list
)

iree_cc_library(
  NAME
    logging
  HDRS
    "logging.h"
  DEPS
    iree::base::internal::logging_internal
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
    platform_headers
  HDRS
    "platform_headers.h"
  DEPS
    iree::base::target_platform
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
    iree::testing::gtest_main
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
    absl::span
    absl::strings
    absl::type_traits
    iree::base::logging
    iree::base::source_location
    iree::base::status
  PUBLIC
)

iree_cc_test(
  NAME
    shape_test
  SRCS
    "shape_test.cc"
  DEPS
    iree::base::shape
    iree::base::status
    iree::base::status_matchers
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    shaped_buffer
  HDRS
    "shaped_buffer.h"
  DEPS
    absl::fixed_array
    absl::span
    iree::base::logging
    iree::base::memory
    iree::base::shape
)

iree_cc_library(
  NAME
    shaped_buffer_string_util
  SRCS
    "shaped_buffer_string_util.cc"
  HDRS
    "shaped_buffer_string_util.h"
  DEPS
    absl::fixed_array
    absl::strings
    absl::optional
    absl::span
    iree::base::buffer_string_util
    iree::base::memory
    iree::base::shape
    iree::base::shaped_buffer
    iree::base::source_location
    iree::base::status
)

iree_cc_test(
  NAME
    shaped_buffer_string_util_test
  SRCS
    "shaped_buffer_string_util_test.cc"
  DEPS
    absl::strings
    iree::base::buffer_string_util
    iree::base::memory
    iree::base::shaped_buffer
    iree::base::shaped_buffer_string_util
    iree::base::status
    iree::base::status_matchers
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    signature_mangle
  HDRS
    "signature_mangle.h"
  SRCS
    "signature_mangle.cc"
  DEPS
    absl::container
    absl::memory
    absl::optional
    absl::span
    absl::strings
    iree::base::status
  PUBLIC
)

iree_cc_test(
  NAME
    signature_mangle_test
  SRCS
    "signature_mangle_test.cc"
  DEPS
    iree::testing::gtest_main
    iree::base::signature_mangle
)

iree_cc_library(
  NAME
    source_location
  HDRS
    "source_location.h"
  DEPS
    iree::base::internal::source_location_internal
  PUBLIC
)

iree_cc_library(
  NAME
    status
  HDRS
    "status.h"
  DEPS
    iree::base::source_location
    iree::base::internal::status_internal
  PUBLIC
)

iree_cc_library(
  NAME
    status_matchers
  HDRS
    "status_matchers.h"
  DEPS
    iree::base::internal::status_matchers_internal
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
    DEPS
      absl::strings
      absl::optional
      absl::time
      iree::base::tracing_enabled
      # TODO(marbre): Add dependencies
      # "@com_google_tracing_framework_cpp//:tracing_framework_bindings_cpp"
      # "@com_google_tracing_framework_cpp//:wtf_enable": [":tracing_enabled"]
    PUBLIC
  )
else()
  iree_cc_library(
    NAME
      tracing
    HDRS
      "tracing.h"
    DEPS
      absl::strings
      absl::time
      absl::optional
      iree::base::tracing_disabled
    PUBLIC
  )
endif()

iree_cc_library(
  NAME
    tracing_disabled
  HDRS
    "tracing.h"
  SRCS
    "tracing_disabled.cc"
  DEPS
    absl::flags
    absl::strings
    absl::time
    absl::optional
    # TODO(marbre): Add dependency
    # "@com_google_tracing_framework_cpp//:tracing_framework_bindings_cpp"
  ALWAYSLINK
)

if(${IREE_ENABLE_TRACING})
  iree_cc_library(
    NAME
      tracing_enabled
    HDRS
      "tracing.h"
    SRCS
      "tracing_enabled.cc"
    DEPS
      absl::core_headers
      absl::flags
      absl::strings
      absl::synchronization
      absl::time
      absl::optional
      iree::base::file_io
      iree::base::file_path
      iree::base::initializer
      iree::base::logging
      iree::base::status
      # TODO(marbre): Add dependency
      # "@com_google_tracing_framework_cpp//:tracing_framework_bindings_cpp"
    ALWAYSLINK
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
#     iree::testing::gtest_main
#     iree::base::status
#     iree::base::status_matchers
#     iree::base::wait_handle
# )
