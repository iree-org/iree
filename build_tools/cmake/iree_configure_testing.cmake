# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Functions for setting up testing in the IREE project. Configures some specific
# environment variables and sets up the creation of test-specific temporary
# directories.

enable_testing(iree)
# A property is apparently the only way to get an uncached global variable.
set_property(GLOBAL PROPERTY IREE_TEST_TMPDIRS "")
set(IREE_TEST_TMPDIR_ROOT "${IREE_BINARY_DIR}/test_tmpdir")

# iree_configure_test
#
# Registers test for temporary directory creation and adds properties common to
# all IREE tests. This should be invoked with each test added with `add_test`.
#
# Parameters:
#   TEST_NAME: the test name, e.g. iree/base/math_test
function(iree_configure_test TEST_NAME)
  set(_TEST_TMPDIR "${IREE_TEST_TMPDIR_ROOT}/${TEST_NAME}_test_tmpdir")
  set_property(GLOBAL APPEND PROPERTY IREE_TEST_TMPDIRS ${_TEST_TMPDIR})
  set_property(TEST ${TEST_NAME} APPEND PROPERTY ENVIRONMENT "TEST_TMPDIR=${_TEST_TMPDIR}")
  set_property(TEST ${TEST_NAME} APPEND PROPERTY ENVIRONMENT "IREE_BINARY_DIR=${IREE_BINARY_DIR}")

  # File extension cmake uses for the target platform.
  set_property(TEST ${TEST_NAME} APPEND PROPERTY ENVIRONMENT "IREE_DYLIB_EXT=${CMAKE_SHARED_LIBRARY_SUFFIX}")

  # IREE_*_DISABLE environment variables may used to skip test cases which
  # require both a compiler target backend and compatible runtime HAL driver.
  #
  # These variables may be set by the test environment, typically as a property
  # of some continuous execution test runner or by an individual developer, or
  # here by the build system.
  #
  # Tests which only depend on a compiler target backend or a runtime HAL
  # driver, but not both, should generally use a different method of filtering.
  if(NOT IREE_TARGET_BACKEND_VULKAN_SPIRV OR NOT IREE_HAL_DRIVER_VULKAN)
    set_property(TEST ${TEST_NAME} APPEND PROPERTY ENVIRONMENT "IREE_VULKAN_DISABLE=1")
  endif()

  if(NOT IREE_TARGET_BACKEND_METAL_SPIRV OR NOT IREE_HAL_DRIVER_METAL)
    set_property(TEST ${TEST_NAME} APPEND PROPERTY ENVIRONMENT "IREE_METAL_DISABLE=1")
  endif()

endfunction()

# iree_create_ctest_customization
#
# Constructs a CTestCustom.cmake file with custom commands run before ctest
# runs all tests. These commands create new temporary directories for each test
# that was properly configured with `iree_configure_test`.
#
# Note that this must be called after all tests are registered as it depends on
# a global variable (gross, I know).
#
# Takes no arguments
function(iree_create_ctest_customization)
  get_property(IREE_TEST_TMPDIRS GLOBAL PROPERTY IREE_TEST_TMPDIRS)
  set(IREE_CREATE_TEST_TMPDIRS_COMMANDS "")
  set(_CMD_PREFIX "\"cmake -E make_directory")
  set(_CUR_CMD "${_CMD_PREFIX}")
  set(_CMD_LEN_LIMIT 8191)
  foreach(_DIR IN LISTS IREE_TEST_TMPDIRS)
    string(LENGTH "${_CUR_CMD}" _CUR_CMD_LEN)
    if(_CUR_CMD_LEN GREATER _CMD_LEN_LIMIT)
      message(SEND_ERROR
          "Make directory command for single test directory is longer than"
          " maximum command length ${_CMD_LEN_LIMIT}: '${_CUR_CMD}'")
    endif()
    string(LENGTH "${_DIR}" _DIR_LEN)
    math(EXPR _NEW_CMD_LEN "${_CUR_CMD_LEN} + ${_DIR_LEN} + 1")
    if(_NEW_CMD_LEN GREATER _CMD_LEN_LIMIT)
      string(APPEND _CUR_CMD "\"\n")
      string(APPEND IREE_CREATE_TEST_TMPDIRS_COMMANDS "${_CUR_CMD}")
      set(_CUR_CMD "${_CMD_PREFIX} ${_DIR}")
    else()
      string(APPEND _CUR_CMD " ${_DIR}")
    endif()
  endforeach()
  if(NOT _CUR_CMD STREQUAL _CMD_PREFIX)
    string(APPEND _CUR_CMD "\"\n")
    string(APPEND IREE_CREATE_TEST_TMPDIRS_COMMANDS "${_CUR_CMD}")
  endif()

  configure_file("build_tools/cmake/CTestCustom.cmake.in" "${IREE_BINARY_DIR}/CTestCustom.cmake" @ONLY)
endfunction()
