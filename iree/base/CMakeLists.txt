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
    base
  HDRS
    "api.h"
  SRCS
    "allocator.c"
    "allocator.h"
    "api.c"
    "status.c"
    "string_view.c"
    "string_view.h"
    "time.c"
    "time.h"
  DEPS
    ::core_headers
    ::tracing
  PUBLIC
)

iree_cc_library(
  NAME
    core_headers
  HDRS
    "alignment.h"
    "attributes.h"
    "config.h"
    "target_platform.h"
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
    logging
  HDRS
    "logging.h"
  SRCS
    "logging.cc"
  DEPS
    ::core_headers
    ::tracing
    iree::base::internal::flags
  PUBLIC
)

iree_cc_library(
  NAME
    signature_parser
  HDRS
    "signature_parser.h"
  SRCS
    "signature_parser.cc"
  DEPS
    absl::optional
    absl::span
    absl::strings
  PUBLIC
)

iree_cc_test(
  NAME
    signature_parser_test
  SRCS
    "signature_parser_test.cc"
  DEPS
    ::signature_parser
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    status
  HDRS
    "status.h"
  DEPS
    iree::base::internal::status_internal
  PUBLIC
)

iree_cc_test(
  NAME
    status_test
  SRCS
    "status_test.cc"
  DEPS
    ::status
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_test(
  NAME
    string_view_test
  SRCS
    "string_view_test.cc"
  DEPS
    ::base
    iree::testing::gtest
    iree::testing::gtest_main
)

# TODO(benvanik): evaluate if we want this as part of the API. Could restrict it
# to excusively static linkage scenarios and note that it's unstable. It's just
# really really useful and the only way for applications to interleave with our
# tracing (today).
if(${IREE_ENABLE_RUNTIME_TRACING})
  iree_cc_library(
    NAME
      tracing
    HDRS
      "tracing.h"
      "${IREE_ROOT_DIR}/third_party/tracy/Tracy.hpp"
      "${IREE_ROOT_DIR}/third_party/tracy/TracyC.h"
    SRCS
      "tracing.cc"
    DEPS
      ::core_headers
    DEFINES
      "IREE_TRACING_MODE=2"
    PUBLIC
  )
else()
  iree_cc_library(
    NAME
      tracing
    HDRS
      "tracing.h"
    DEPS
      ::core_headers
    PUBLIC
  )
endif()
