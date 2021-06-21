# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
    cc
  HDRS
    "status_cc.h"
  SRCS
    "status_cc.cc"
  DEPS
    ::base
    ::core_headers
    ::logging
  PUBLIC
)

iree_cc_test(
  NAME
    status_test
  SRCS
    "status_test.cc"
  DEPS
    ::base
    ::cc
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
