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
    alignment
  HDRS
    "alignment.h"
  DEPS
    ::platform_headers
    ::target_platform
  PUBLIC
)

iree_cc_library(
  NAME
    api
  HDRS
    "api.h"
  SRCS
    "api.cc"
  DEPS
    ::api_hdrs
    ::api_util
    ::file_mapping
    ::init
    ::tracing
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
    ::api_hdrs
    ::logging
    ::shape
    ::status
    absl::core_headers
    absl::inlined_vector
    absl::time
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
    ::logging
    absl::core_headers
    absl::span
  PUBLIC
)

iree_cc_test(
  NAME
    arena_test
  SRCS
    "arena_test.cc"
  DEPS
    ::arena
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    atomics
  HDRS
    "atomics.h"
  DEPS
    ::platform_headers
    ::target_platform
  PUBLIC
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
    ::bitfield
    absl::core_headers
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    buffer_string_util
  HDRS
    "buffer_string_util.h"
  SRCS
    "buffer_string_util.cc"
  DEPS
    ::memory
    ::shape
    ::source_location
    ::status
    absl::flat_hash_map
    absl::optional
    absl::span
    absl::strings
  PUBLIC
)

iree_cc_test(
  NAME
    buffer_string_util_test
  SRCS
    "buffer_string_util_test.cc"
  DEPS
    ::buffer_string_util
    ::memory
    ::status
    ::status_matchers
    absl::strings
    iree::testing::gtest_main
)

iree_select_compiler_opts(_DYNAMIC_LIBRARY_LINKOPTS
  CLANG_OR_GCC
    "-ldl"
)

iree_cc_library(
  NAME
    dynamic_library
  HDRS
    "dynamic_library.h"
  SRCS
    "dynamic_library_posix.cc"
    "dynamic_library_win32.cc"
  LINKOPTS
    ${_DYNAMIC_LIBRARY_LINKOPTS}
DEPS
    ::logging
    ::platform_headers
    ::status
    ::target_platform
    ::tracing
    absl::memory
    absl::span
  PUBLIC
)

iree_cc_library(
  NAME
    file_io
  HDRS
    "file_io.h"
  DEPS
    ::status
    ::target_platform
    absl::memory
    absl::span
    absl::strings
    iree::base::internal::file_io_internal
  PUBLIC
)

iree_cc_library(
  NAME
    file_io_hdrs
  HDRS
    "file_io.h"
  DEPS
    ::status
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
    ::logging
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
    ::ref_ptr
    ::status
    absl::memory
    absl::span
    absl::strings
    iree::base::internal::file_mapping_internal
  PUBLIC
)

iree_cc_library(
  NAME
    file_mapping_hdrs
  HDRS
    "file_mapping.h"
  DEPS
    ::ref_ptr
    ::status
    absl::span
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
    ::file_mapping
    ::memory
    ::ref_ptr
    ::source_location
    ::status
    ::tracing
    absl::memory
    absl::optional
    absl::span
    absl::strings
    flatbuffers
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
    ::target_platform
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
    ::logging
    ::ref_ptr
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
    ::intrusive_list
    absl::memory
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    localfile
  DEPS
    iree::base::internal::localfile_internal
  ALWAYSLINK
  PUBLIC
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
    ::target_platform
  PUBLIC
)

iree_cc_library(
  NAME
    ref_ptr
  HDRS
    "ref_ptr.h"
  DEPS
    ::logging
    absl::core_headers
  PUBLIC
)

iree_cc_test(
  NAME
    ref_ptr_test
  SRCS
    "ref_ptr_test.cc"
  DEPS
    ::ref_ptr
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    shape
  HDRS
    "shape.h"
  SRCS
    "shape.cc"
  DEPS
    ::logging
    ::source_location
    ::status
    absl::span
    absl::strings
    absl::type_traits
  PUBLIC
)

iree_cc_test(
  NAME
    shape_test
  SRCS
    "shape_test.cc"
  DEPS
    ::shape
    ::status
    ::status_matchers
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    shaped_buffer
  HDRS
    "shaped_buffer.h"
  DEPS
    ::logging
    ::memory
    ::shape
    absl::fixed_array
    absl::span
  PUBLIC
)

iree_cc_library(
  NAME
    shaped_buffer_string_util
  HDRS
    "shaped_buffer_string_util.h"
  SRCS
    "shaped_buffer_string_util.cc"
  DEPS
    ::buffer_string_util
    ::memory
    ::shape
    ::shaped_buffer
    ::source_location
    ::status
    absl::fixed_array
    absl::optional
    absl::span
    absl::strings
  PUBLIC
)

iree_cc_test(
  NAME
    shaped_buffer_string_util_test
  SRCS
    "shaped_buffer_string_util_test.cc"
  DEPS
    ::buffer_string_util
    ::memory
    ::shaped_buffer
    ::shaped_buffer_string_util
    ::status
    ::status_matchers
    absl::strings
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
    ::status
    absl::inlined_vector
    absl::memory
    absl::optional
    absl::span
    absl::strings
  PUBLIC
)

iree_cc_test(
  NAME
    signature_mangle_test
  SRCS
    "signature_mangle_test.cc"
  DEPS
    ::signature_mangle
    iree::testing::gtest_main
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
    ::source_location
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

iree_cc_test(
  NAME
    status_test
  SRCS
    "status_test.cc"
  DEPS
    ::status
    ::status_matchers
    iree::testing::gtest_main
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
