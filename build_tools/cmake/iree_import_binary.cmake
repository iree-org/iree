# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

# iree_import_binary()
#
# CMake function to import an executable/binary file into a CMake target.
# This imports from the directory specified by IREE_HOST_BIN_DIR, and
# that variable _must_ be set for calls to the function to be valid.
#
# Parameters:
# NAME: name of target/binary (see Usage below)
# OPTIONAL: Don't fail if not found (but will issue a warning)
#
# Usage:
#   if(BUILD_AWESOME_TOOL)
#     iree_cc_binary(
#       NAME awesome-tool
#       SRCS "awesome-tool-main.cc"
#     )
#   elseif(IREE_HOST_BIN_DIR)
#     # Import '${IREE_HOST_BIN_DIR}/awesome-tool[.exe]' into the
#     # CMake target 'awesome-tool'.
#     iree_import_binary(NAME awesome-tool)
#   else()
#     message(STATUS "Not building or importing awesome-tool")
#   endif()
function(iree_import_binary)
  cmake_parse_arguments(
    _RULE
    "OPTIONAL"
    "NAME"
    ""
    ${ARGN}
  )

  # TODO(scotttodd): optional 'TARGET' argument (that defaults to NAME)
  # TODO(scotttodd): SHARED_LIBRARY_DEPS argument?

  if(NOT IREE_HOST_BIN_DIR)
    message(FATAL_ERROR "IREE_HOST_BIN_DIR must be set to use iree_import_binary")
  endif()

  # We can't use CMAKE_EXECUTABLE_SUFFIX for host tools when cross-compiling for
  # platforms like Emscripten that set the suffix (e.g. to .js).
  # https://gitlab.kitware.com/cmake/cmake/-/issues/17553
  set(_HOST_EXECUTABLE_SUFFIX "")
  if(CMAKE_HOST_WIN32)
    set(_HOST_EXECUTABLE_SUFFIX ".exe")
  endif()

  set(_FULL_BINARY_NAME "${_RULE_NAME}${_HOST_EXECUTABLE_SUFFIX}")
  set(_BINARY_PATH "${IREE_HOST_BIN_DIR}/${_FULL_BINARY_NAME}")
  file(REAL_PATH "${_BINARY_PATH}" _BINARY_PATH
       BASE_DIRECTORY ${IREE_ROOT_DIR} EXPAND_TILDE)

  if(NOT EXISTS ${_BINARY_PATH})
    if(_RULE_OPTIONAL)
      message(WARNING "Could not find optional '${_FULL_BINARY_NAME}' under "
              "'${IREE_HOST_BIN_DIR}'. Features that depend on it may fail to "
              "build.")
    else()
      message(FATAL_ERROR "Could not find '${_FULL_BINARY_NAME}' under "
              "'${IREE_HOST_BIN_DIR}'\n(Expanded to '${_BINARY_PATH}').\n"
              "Ensure that IREE_HOST_BIN_DIR points to a complete binary directory.")
    endif()
  endif()

  add_executable(${_RULE_NAME} IMPORTED GLOBAL)
  set_property(TARGET "${_RULE_NAME}" PROPERTY IMPORTED_LOCATION "${_BINARY_PATH}")
endfunction()
