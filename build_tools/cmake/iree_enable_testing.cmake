# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#

enable_testing(iree)
# This is apparently the only way to get an uncached global variable.
set_property(GLOBAL PROPERTY IREE_TEST_TMPDIRS_REQUIRED "")
set(IREE_TEST_TMPDIR_ROOT "${IREE_BINARY_DIR}/test_tmpdir")

function(register_test_tmpdir)
  set_property(GLOBAL APPEND PROPERTY IREE_TEST_TMPDIRS_REQUIRED ${_TEST_TMPDIR})
endfunction()

function(ctest_create_test_tmpdir_pretest)
  get_property(IREE_TEST_TMPDIRS_REQUIRED GLOBAL PROPERTY IREE_TEST_TMPDIRS_REQUIRED)
  set(IREE_CREATE_TEST_TMPDIRS_COMMANDS "")
  set(_CMD_PREFIX "\"cmake -E make_directory")
  set(_CUR_CMD "${_CMD_PREFIX}")
  set(_CMD_LEN_LIMIT 8191)
  foreach(_DIR IN LISTS IREE_TEST_TMPDIRS_REQUIRED)
    string(LENGTH "${_CUR_CMD}" _CUR_CMD_LEN)
    if(_CUR_CMD_LEN GREATER _CMD_LEN_LIMIT)
      message(SEND_ERROR "Make directory command for single test directory is longer than maximum command length ${_CMD_LEN_LIMIT}: '${_CUR_CMD}'")
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
  string(APPEND _CUR_CMD "\"\n")
  string(APPEND IREE_CREATE_TEST_TMPDIRS_COMMANDS "${_CUR_CMD}")

  configure_file("build_tools/cmake/CTestCustom.cmake.in" "${IREE_BINARY_DIR}/CTestCustom.cmake" @ONLY)
endfunction()
