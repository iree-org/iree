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
    "assert.h"
    "bitfield.c"
    "bitfield.h"
    "loop.c"
    "loop.h"
    "loop_inline.c"
    "loop_inline.h"
    "status.c"
    "status.h"
    "status_cc.h"
    "string_builder.c"
    "string_builder.h"
    "string_view.c"
    "string_view.h"
    "time.c"
    "time.h"
    "wait_source.c"
    "wait_source.h"
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

iree_cc_test(
  NAME
    bitfield_test
  SRCS
    "bitfield_test.cc"
  DEPS
    ::base
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_test(
  NAME
    loop_inline_test
  SRCS
    "loop_inline_test.cc"
  DEPS
    ::base
    ::loop_test_hdrs
    ::tracing
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_library(
  NAME
    loop_test_hdrs
  HDRS
    "loop_test.h"
  DEPS
    ::base
    ::tracing
    iree::base::internal::wait_handle
    iree::testing::gtest
  TESTONLY
  PUBLIC
)

iree_cc_test(
  NAME
    status_test
  SRCS
    "status_test.cc"
  DEPS
    ::base
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_test(
  NAME
    string_builder_test
  SRCS
    "string_builder_test.cc"
  DEPS
    ::base
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
    target_platform
  HDRS
    "target_platform.h"
  PUBLIC
)

iree_cc_library(
  NAME
    loop_sync
  HDRS
    "loop_sync.h"
  SRCS
    "loop_sync.c"
  DEPS
    ::base
    ::tracing
    iree::base::internal
    iree::base::internal::wait_handle
  PUBLIC
)

iree_cc_test(
  NAME
    loop_sync_test
  SRCS
    "loop_sync_test.cc"
  DEPS
    ::base
    ::loop_sync
    ::loop_test_hdrs
    ::tracing
    iree::testing::gtest
    iree::testing::gtest_main
)

# TODO(benvanik): evaluate if we want this as part of the API. Could restrict it
# to excusively static linkage scenarios and note that it's unstable. It's just
# really really useful and the only way for applications to interleave with our
# tracing (today).
iree_cc_library(
  NAME
    tracing
  HDRS
    "tracing.h"
  SRCS
    "tracing.cc"
  DEPS
    ::core_headers
    tracy_client::runtime_impl
  PUBLIC
)

if(EMSCRIPTEN)
  iree_cc_library(
    NAME
      loop_emscripten
    HDRS
      "loop_emscripten.h"
    SRCS
      "loop_emscripten.c"
    DEPS
      ::base
      iree::base::internal::wait_handle
    PUBLIC
  )

  iree_link_js_library(
    TARGET
      ::loop_emscripten
    SRCS
      "loop_emscripten.js"
  )

  # TODO(scotttodd): rewrite as a JS/C file and test with Promises
  #   The C++ test uses IREE_LOOP_COMMAND_DRAIN, which is not implemented here
  #   This test is just useful for the initial bring-up.
  iree_cc_test(
    NAME
      loop_emscripten_test
    SRCS
      "loop_emscripten_test.cc"
    DEPS
      ::base
      ::loop_emscripten
      ::loop_test_hdrs
      iree::testing::gtest
      iree::testing::gtest_main
  )
endif()
