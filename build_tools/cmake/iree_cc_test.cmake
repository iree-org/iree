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

include(CMakeParseArguments)
include(iree_installed_test)

# iree_cc_test()
#
# CMake function to imitate Bazel's cc_test rule.
#
# Parameters:
# NAME: name of target (see Usage below)
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
# By default, iree_cc_test will always create a binary named iree_${NAME}.
# This will also add it to ctest list as iree_${NAME}.
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
    "SRCS;COPTS;DEFINES;LINKOPTS;DATA;DEPS;LABELS"
    ${ARGN}
  )

  # Prefix the library with the package name, so we get: iree_package_name
  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  add_executable(${_NAME} "")
  add_executable(${_RULE_NAME} ALIAS ${_NAME})
  if(_RULE_OUT)
    set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "${_RULE_OUT}")
  else()
    set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "${_RULE_NAME}")
  endif()
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
  iree_package_ns(_PACKAGE_NS)
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

  string(REPLACE "::" "/" _PACKAGE_PATH ${_PACKAGE_NS})
  set(_TEST_NAME "${_PACKAGE_PATH}/${_RULE_NAME}")

  list(APPEND _RULE_LABELS "${_PACKAGE_PATH}")

  # Case for cross-compiling towards Android.
  if(ANDROID)
    set(_ANDROID_REL_DIR "${_PACKAGE_PATH}/${_RULE_NAME}")
    set(_ANDROID_ABS_DIR "/data/local/tmp/${_ANDROID_REL_DIR}")

    # Define a custom target for pushing and running the test on Android device.
    set(_TEST_NAME ${_TEST_NAME}_on_android_device)
    add_test(
      NAME
        ${_TEST_NAME}
      COMMAND
        "${CMAKE_SOURCE_DIR}/build_tools/cmake/run_android_test.${IREE_HOST_SCRIPT_EXT}"
        "${_ANDROID_REL_DIR}/$<TARGET_FILE_NAME:${_NAME}>"
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
    set_property(TEST ${_TEST_NAME} PROPERTY ENVIRONMENT ${_ENVIRONMENT_VARS})
    set_property(TEST ${_TEST_NAME} PROPERTY LABELS "${_RULE_LABELS}")
  else(ANDROID)
    iree_add_installed_test(
      TEST_NAME "${_TEST_NAME}"
      LABELS "${_RULE_LABELS}"
      COMMAND
        # We run all our tests through a custom test runner to allow temp
        # directory cleanup upon test completion.
        "${CMAKE_SOURCE_DIR}/build_tools/cmake/run_test.${IREE_HOST_SCRIPT_EXT}"
        "$<TARGET_FILE:${_NAME}>"
      INSTALLED_COMMAND
        # Must match install destination below.
        "${_PACKAGE_PATH}/$<TARGET_FILE_NAME:${_NAME}>"
    )
  endif(ANDROID)

  install(TARGETS ${_NAME}
    DESTINATION "tests/${_PACKAGE_PATH}"
    COMPONENT Tests
  )
endfunction()
