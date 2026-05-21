# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# CTS test suite infrastructure for HAL drivers.
#
# Two functions:
#
# 1. iree_hal_cts_testdata() - Compiles CTS test executables for one format
#    and creates a testdata registration library. MLIR sources are discovered
#    via file(GLOB) from the testdata directory.
#
# 2. iree_hal_cts_test_suite() - Creates CTS test binaries for a driver,
#    linking against pre-built testdata libraries from iree_hal_cts_testdata().

# Template for generated testdata registration .cc files.
set(_IREE_CTS_TESTDATA_TEMPLATE [=[
// Auto-generated executable format registration for CTS.
#include "iree/hal/cts/util/registry.h"
#include "@HEADER_PATH@"

namespace iree::hal::cts {

static iree_const_byte_span_t Get@FORMAT_FUNC_NAME@ExecutableData(
    iree_string_view_t file_name) {
  const iree_file_toc_t* toc = @IDENTIFIER@_create();
  for (size_t i = 0; toc[i].name != nullptr; ++i) {
    if (iree_string_view_equal(file_name,
                               iree_make_cstring_view(toc[i].name))) {
      return iree_make_const_byte_span(
          reinterpret_cast<const uint8_t*>(toc[i].data), toc[i].size);
    }
  }
  return iree_const_byte_span_empty();
}

static bool @FORMAT_VAR_NAME@_registered_ =
    (CtsRegistry::RegisterExecutableFormat(
         "@BACKEND_NAME@",
         {"@FORMAT_NAME@", @FORMAT_STRING@, Get@FORMAT_FUNC_NAME@ExecutableData}),
     true);

}  // namespace iree::hal::cts
]=])

function(_iree_cts_camel_case INPUT OUTPUT_VAR)
  string(REPLACE "_" ";" _PARTS "${INPUT}")
  set(_RESULT "")
  foreach(_PART ${_PARTS})
    string(SUBSTRING "${_PART}" 0 1 _FIRST)
    string(TOUPPER "${_FIRST}" _FIRST)
    string(SUBSTRING "${_PART}" 1 -1 _REST)
    string(APPEND _RESULT "${_FIRST}${_REST}")
  endforeach()
  set(${OUTPUT_VAR} "${_RESULT}" PARENT_SCOPE)
endfunction()

# iree_hal_cts_testdata()
#
# Compiles CTS test executables for one format and creates a testdata
# registration library. MLIR sources are discovered via file(GLOB) from
# the provided TESTDATA_DIR.
#
# Creates two targets:
#   testdata_${FORMAT_NAME}      - iree_hal_executables embedded data
#   testdata_${FORMAT_NAME}_lib  - registration library (link into tests)
#
# Parameters:
#   FORMAT_NAME: Short name (e.g., "vmvx", "llvm_cpu", "cuda").
#   TARGET_DEVICE: Target device for iree-compile (e.g., "local", "cuda").
#   IDENTIFIER: C identifier for the embedded data (e.g., "iree_cts_testdata_vmvx").
#   BACKEND_NAME: Backend name for CtsRegistry (e.g., "local_task").
#   FORMAT_STRING: C expression for the format (e.g., "vmvx-bytecode-fb").
#   TESTDATA_DIR: Directory containing MLIR test sources.
#   FLAGS: Additional compiler flags.
function(iree_hal_cts_testdata)
  cmake_parse_arguments(
    _RULE
    ""
    "FORMAT_NAME;TARGET_DEVICE;IDENTIFIER;BACKEND_NAME;FORMAT_STRING;TESTDATA_DIR"
    "FLAGS"
    ${ARGN}
  )

  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  # Compiling testdata requires iree-compile (built or imported).
  if(NOT IREE_BUILD_COMPILER AND NOT IREE_HOST_BIN_DIR)
    return()
  endif()

  if(NOT DEFINED _RULE_TESTDATA_DIR)
    message(SEND_ERROR "iree_hal_cts_testdata requires TESTDATA_DIR")
  endif()

  # Discover MLIR sources from the testdata directory.
  file(GLOB _SRCS LIST_DIRECTORIES false CONFIGURE_DEPENDS
    "${_RULE_TESTDATA_DIR}/*.mlir")
  if(NOT _SRCS)
    message(FATAL_ERROR "No MLIR files found in ${_RULE_TESTDATA_DIR}")
  endif()

  set(_TESTDATA_NAME "testdata_${_RULE_FORMAT_NAME}")

  # Compile all MLIR sources to HAL executables and bundle into embedded data.
  iree_hal_executables(
    NAME "${_TESTDATA_NAME}"
    SRCS ${_SRCS}
    TARGET_DEVICE "${_RULE_TARGET_DEVICE}"
    FLAGS ${_RULE_FLAGS}
    IDENTIFIER "${_RULE_IDENTIFIER}"
    TESTONLY
    PUBLIC
  )

  # Generate the registration .cc from template.
  _iree_cts_camel_case("${_RULE_FORMAT_NAME}" _FUNC_NAME)

  set(HEADER_PATH "${_TESTDATA_NAME}.h")
  set(FORMAT_FUNC_NAME "${_FUNC_NAME}")
  set(IDENTIFIER "${_RULE_IDENTIFIER}")
  set(FORMAT_VAR_NAME "${_RULE_FORMAT_NAME}_format")
  set(BACKEND_NAME "${_RULE_BACKEND_NAME}")
  set(FORMAT_NAME "${_RULE_FORMAT_NAME}")
  set(FORMAT_STRING "${_RULE_FORMAT_STRING}")

  string(CONFIGURE "${_IREE_CTS_TESTDATA_TEMPLATE}" _GENERATED_CC @ONLY)
  set(_GEN_CC_FILE "${CMAKE_CURRENT_BINARY_DIR}/${_TESTDATA_NAME}.cc")
  file(GENERATE OUTPUT "${_GEN_CC_FILE}" CONTENT "${_GENERATED_CC}")

  iree_cc_library(
    NAME "${_TESTDATA_NAME}_lib"
    SRCS "${_GEN_CC_FILE}"
    DEPS
      ::${_TESTDATA_NAME}
      iree::hal::cts::util::registry
    TESTONLY
    ALWAYSLINK
    PUBLIC
  )
endfunction()

# iree_hal_cts_test_suite()
#
# Creates CTS test binaries for a HAL driver. Non-executable tests (buffer,
# command_buffer, core, file, queue) are always created. Executable-dependent
# tests (dispatch, executable) are created only if TESTDATA_LIBS are provided.
#
# Parameters:
#   BACKENDS_LIB: CMake target for the backends registration library.
#   TESTDATA_LIBS: Testdata library targets from iree_hal_cts_testdata().
#   NAME: Optional prefix for test binary names (e.g., "graph", "stream").
#   ARGS: Runtime arguments passed to all test binaries.
#   LABELS: Test labels for filtering.
#   RESOURCE_GROUP: Optional shared resource group. Tests sharing the same
#     resource group do not run concurrently under CTest.
function(iree_hal_cts_test_suite)
  cmake_parse_arguments(
    _RULE
    ""
    "BACKENDS_LIB;NAME;RESOURCE_GROUP"
    "TESTDATA_LIBS;ARGS;LABELS"
    ${ARGN}
  )

  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  if(NOT DEFINED _RULE_BACKENDS_LIB)
    message(SEND_ERROR "iree_hal_cts_test_suite requires BACKENDS_LIB")
  endif()

  # Build the name prefix: "name_" if set, "" otherwise.
  set(_PREFIX "")
  if(_RULE_NAME)
    set(_PREFIX "${_RULE_NAME}_")
  endif()

  set(_COMMON_DEPS
    ${_RULE_BACKENDS_LIB}
    iree::hal::cts::util::registry
    iree::hal::cts::util::test_base
    iree::testing::gtest
  )

  set(_TEST_MAIN
    "${PROJECT_SOURCE_DIR}/runtime/src/iree/hal/cts/util/test_main.cc"
  )

  # Build ARGS block for iree_cc_test.
  set(_ARGS_BLOCK "")
  if(_RULE_ARGS)
    set(_ARGS_BLOCK ARGS ${_RULE_ARGS})
  endif()

  set(_LABELS_BLOCK "")
  if(_RULE_LABELS)
    set(_LABELS_BLOCK LABELS ${_RULE_LABELS})
  endif()

  set(_RESOURCE_GROUP_BLOCK "")
  if(_RULE_RESOURCE_GROUP)
    set(_RESOURCE_GROUP_BLOCK RESOURCE_GROUP "${_RULE_RESOURCE_GROUP}")
  endif()

  # Non-executable test categories.
  foreach(_CATEGORY buffer command_buffer core file queue)
    iree_cc_test(
      NAME "${_PREFIX}${_CATEGORY}_tests"
      SRCS "${_TEST_MAIN}"
      DEPS
        ${_COMMON_DEPS}
        "iree::hal::cts::${_CATEGORY}::all_tests"
      ${_ARGS_BLOCK}
      ${_LABELS_BLOCK}
      ${_RESOURCE_GROUP_BLOCK}
    )
  endforeach()

  # Executable-dependent test categories (require testdata compiled by
  # iree-compile, which may be unavailable in runtime-only builds).
  if(_RULE_TESTDATA_LIBS)
    # Verify all testdata targets exist (they won't if the compiler is absent).
    set(_TESTDATA_AVAILABLE TRUE)
    iree_package_ns(_TESTDATA_PACKAGE_NS)
    foreach(_LIB ${_RULE_TESTDATA_LIBS})
      string(REGEX REPLACE "^::" "${_TESTDATA_PACKAGE_NS}::" _FULL_LIB "${_LIB}")
      string(REPLACE "::" "_" _TARGET_NAME "${_FULL_LIB}")
      if(NOT TARGET "${_TARGET_NAME}")
        set(_TESTDATA_AVAILABLE FALSE)
        break()
      endif()
    endforeach()
  endif()

  if(_RULE_TESTDATA_LIBS AND _TESTDATA_AVAILABLE)
    set(_EXECUTABLE_SUITES
      "dispatch_tests\;iree::hal::cts::command_buffer::all_dispatch_tests"
      "executable_tests\;iree::hal::cts::core::all_executable_tests"
      "queue_dispatch_tests\;iree::hal::cts::queue::queue_dispatch_test"
    )
    foreach(_PAIR ${_EXECUTABLE_SUITES})
      list(GET _PAIR 0 _SUFFIX)
      list(GET _PAIR 1 _TEST_LIB)
      iree_cc_test(
        NAME "${_PREFIX}${_SUFFIX}"
        SRCS "${_TEST_MAIN}"
        DEPS
          ${_COMMON_DEPS}
          ${_RULE_TESTDATA_LIBS}
          ${_TEST_LIB}
        ${_ARGS_BLOCK}
        ${_LABELS_BLOCK}
        ${_RESOURCE_GROUP_BLOCK}
      )
    endforeach()
  endif()
endfunction()
