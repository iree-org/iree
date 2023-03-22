# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

# iree_cc_test()
#
# CMake function to imitate Bazel's cc_test rule.
#
# Parameters:
# NAME: name of target. This name is used for the generated executable and
#     CTest target.
# ARGS: List of command line arguments to pass to the test binary.
#     Note: flag passing is only enforced through CTest, so manually running
#     the test binaries (such as under a debugger) will _not_ pass any
#     arguments without extra setup.
# SRCS: List of source files for the binary
# DATA: List of other targets and files required for this binary
# DEPS: List of other libraries to be linked in to the binary targets
# COPTS: List of private compile options
# DEFINES: List of public defines
# LINKOPTS: List of link options
# LABELS: Additional labels to apply to the test. The package path is added
#     automatically.
#
# Note:
# iree_cc_test will create a binary called ${PACKAGE_NAME}_${NAME}, e.g.
# iree_base_foo_test.
#
#
# Usage:
# iree_cc_library(
#   NAME
#     awesome
#   HDRS
#     "a.h"
#   SRCS
#     "a.cc"
#   PUBLIC
# )
#
# iree_cc_test(
#   NAME
#     awesome_test
#   SRCS
#     "awesome_test.cc"
#   DEPS
#     gtest_main
#     iree::awesome
# )
function(iree_cc_test)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "ARGS;SRCS;COPTS;DEFINES;LINKOPTS;DATA;DEPS;LABELS;TIMEOUT"
    ${ARGN}
  )

  # Prefix the library with the package name, so we get: iree_package_name
  iree_package_name(_PACKAGE_NAME)
  iree_package_ns(_PACKAGE_NS)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  add_executable(${_NAME} "")
  # Alias the iree_package_name test binary to iree::package::name.
  # This lets us more clearly map to Bazel and makes it possible to
  # disambiguate the underscores in paths vs. the separators.
  add_executable(${_PACKAGE_NS}::${_RULE_NAME} ALIAS ${_NAME})

  # If the test binary name matches the package then treat it as a default.
  # For example, foo/bar/ library 'bar' would end up as 'foo::bar'. This isn't
  # likely to be common for tests, but is consistent with the behavior for
  # libraries.
  iree_package_dir(_PACKAGE_DIR)
  if(${_RULE_NAME} STREQUAL ${_PACKAGE_DIR})
    add_executable(${_PACKAGE_NS} ALIAS ${_NAME})
  endif()

  set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "${_RULE_NAME}")
  target_sources(${_NAME}
    PRIVATE
      ${_RULE_SRCS}
  )
  target_include_directories(${_NAME} SYSTEM
    PUBLIC
      "$<BUILD_INTERFACE:${IREE_SOURCE_DIR}>"
      "$<BUILD_INTERFACE:${IREE_BINARY_DIR}>"
  )
  target_compile_definitions(${_NAME}
    PUBLIC
      ${_RULE_DEFINES}
  )
  target_compile_options(${_NAME}
    PRIVATE
      ${IREE_DEFAULT_COPTS}
      ${_RULE_COPTS}
  )
  target_link_options(${_NAME}
    PRIVATE
      ${IREE_DEFAULT_LINKOPTS}
      ${_RULE_LINKOPTS}
  )

  # Replace dependencies passed by ::name with iree::package::name

  list(TRANSFORM _RULE_DEPS REPLACE "^::" "${_PACKAGE_NS}::")

  target_link_libraries(${_NAME}
    PUBLIC
      ${_RULE_DEPS}
  )
  iree_add_data_dependencies(NAME ${_NAME} DATA ${_RULE_DATA})

  # Add all IREE targets to a folder in the IDE for organization.
  set_property(TARGET ${_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER}/test)

  set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD ${IREE_CXX_STANDARD})
  set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

  list(APPEND _RULE_DEPS "gmock")

  # Implicit deps.
  if(IREE_IMPLICIT_DEFS_CC_DEPS)
    list(APPEND _RULE_DEPS ${IREE_IMPLICIT_DEFS_CC_DEPS})
  endif()

  string(REPLACE "::" "/" _PACKAGE_PATH ${_PACKAGE_NS})
  set(_NAME_PATH "${_PACKAGE_PATH}/${_RULE_NAME}")

  # Case for cross-compiling towards Android.
  if(ANDROID)
    set(_ANDROID_REL_DIR "${_PACKAGE_PATH}/${_RULE_NAME}")
    set(_ANDROID_ABS_DIR "/data/local/tmp/${_ANDROID_REL_DIR}")

    # Define a custom target for pushing and running the test on Android device.
    set(_NAME_PATH ${_NAME_PATH}_on_android_device)
    add_test(
      NAME
        ${_NAME_PATH}
      COMMAND
        "${CMAKE_SOURCE_DIR}/build_tools/cmake/run_android_test.${IREE_HOST_SCRIPT_EXT}"
        "${_ANDROID_REL_DIR}/$<TARGET_FILE_NAME:${_NAME}>"
        ${_RULE_ARGS}
    )
    # Use environment variables to instruct the script to push artifacts
    # onto the Android device before running the test. This needs to match
    # with the expectation of the run_android_test.{sh|bat|ps1} script.
    set(
      _ENVIRONMENT_VARS
        TEST_ANDROID_ABS_DIR=${_ANDROID_ABS_DIR}
        TEST_EXECUTABLE=$<TARGET_FILE:${_NAME}>
        TEST_TMPDIR=${_ANDROID_ABS_DIR}/test_tmpdir
    )
    set_property(TEST ${_NAME_PATH} PROPERTY ENVIRONMENT ${_ENVIRONMENT_VARS})
  elseif((IREE_ARCH STREQUAL "riscv_64" OR
          IREE_ARCH STREQUAL "riscv_32") AND
         CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # The test target needs to run within the QEMU emulator for RV64 Linux
    # crosscompile build or on-device.
    add_test(
      NAME
        ${_NAME_PATH}
      COMMAND
       "${IREE_ROOT_DIR}/build_tools/cmake/run_riscv_test.sh"
        "$<TARGET_FILE:${_NAME}>"
        ${_RULE_ARGS}
    )
    iree_configure_test(${_NAME_PATH})
  else(ANDROID)
    add_test(
      NAME
        ${_NAME_PATH}
      COMMAND
        "$<TARGET_FILE:${_NAME}>"
        ${_RULE_ARGS}
      )

    iree_configure_test(${_NAME_PATH})
  endif(ANDROID)

  if (NOT DEFINED _RULE_TIMEOUT)
    set(_RULE_TIMEOUT 60)
  endif()

  list(APPEND _RULE_LABELS "${_PACKAGE_PATH}")
  set_property(TEST ${_NAME_PATH} PROPERTY LABELS "${_RULE_LABELS}")
  set_property(TEST ${_NAME_PATH} PROPERTY TIMEOUT ${_RULE_TIMEOUT})
endfunction()
