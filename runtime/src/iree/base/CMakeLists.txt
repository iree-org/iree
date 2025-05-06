# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

iree_cc_library(
  NAME
    core_headers
  HDRS
    "alignment.h"
    "attributes.h"
    "config.h"
    "target_platform.h"
    "time.h"
    "tracing.h"
  PUBLIC
)

# Users providing their own allocators can have their sources live out-of-tree.
# Note that with out of tree sources they must be absolute paths in order to be
# found in this directory.
set(IREE_ALLOCATOR_LIBC_SRCS "allocator_libc.c")
set(IREE_ALLOCATOR_MIMALLOC_SRCS "allocator_mimalloc.c")

# IREE_ALLOCATOR_SYSTEM -> variable expansion.
string(TOUPPER ${IREE_ALLOCATOR_SYSTEM} _IREE_ALLOCATOR_SYSTEM)
string(REGEX REPLACE "-" "_" _IREE_ALLOCATOR_SYSTEM ${_IREE_ALLOCATOR_SYSTEM})
message(STATUS "iree_allocator_system() using `IREE_ALLOCATOR_${_IREE_ALLOCATOR_SYSTEM}_*`")
get_property(_IREE_ALLOCATOR_SYSTEM_SRCS VARIABLE PROPERTY IREE_ALLOCATOR_${_IREE_ALLOCATOR_SYSTEM}_SRCS)
get_property(_IREE_ALLOCATOR_SYSTEM_COPTS VARIABLE PROPERTY IREE_ALLOCATOR_${_IREE_ALLOCATOR_SYSTEM}_COPTS)
get_property(_IREE_ALLOCATOR_SYSTEM_DEPS VARIABLE PROPERTY IREE_ALLOCATOR_${_IREE_ALLOCATOR_SYSTEM}_DEPS)
get_property(_IREE_ALLOCATOR_SYSTEM_CTL VARIABLE PROPERTY IREE_ALLOCATOR_${_IREE_ALLOCATOR_SYSTEM}_CTL)
get_property(_IREE_ALLOCATOR_SYSTEM_SELF VARIABLE PROPERTY IREE_ALLOCATOR_${_IREE_ALLOCATOR_SYSTEM}_SELF)
if(IREE_ALLOCATOR_SYSTEM_CTL)
  list(APPEND _IREE_ALLOCATOR_SYSTEM_DEFINES
    "-DIREE_ALLOCATOR_SYSTEM_CTL=${_IREE_ALLOCATOR_SYSTEM_CTL}"
  )
else()
  list(APPEND _IREE_ALLOCATOR_SYSTEM_DEFINES
    "-DIREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_${IREE_ALLOCATOR_SYSTEM}_ctl"
  )
endif()
if(IREE_ALLOCATOR_SYSTEM_SELF)
  list(APPEND _IREE_ALLOCATOR_SYSTEM_DEFINES
    "-DIREE_ALLOCATOR_SYSTEM_SELF=${_IREE_ALLOCATOR_SYSTEM_SELF}"
  )
endif()

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
    "status_payload.h"
    "status_stack_trace.c"
    "string_builder.c"
    "string_builder.h"
    "string_view.c"
    "string_view.h"
    "time.c"
    "wait_source.c"
    "wait_source.h"
    ${_IREE_ALLOCATOR_SYSTEM_SRCS}
  COPTS
    ${_IREE_ALLOCATOR_SYSTEM_COPTS}
  DEFINES
    ${_IREE_ALLOCATOR_SYSTEM_DEFINES}
  DEPS
    ::core_headers
    iree::base::internal::time
    iree::base::tracing::provider
    ${_IREE_ALLOCATOR_SYSTEM_DEPS}
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
    loop_sync
  HDRS
    "loop_sync.h"
  SRCS
    "loop_sync.c"
  DEPS
    ::base
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
    iree::testing::gtest
    iree::testing::gtest_main
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
    LINKOPTS
      "-sEXPORTED_RUNTIME_METHODS=['dynCall']"
  )
endif()
