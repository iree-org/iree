# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

# iree_c_embed_data()
#
# CMake function to imitate Bazel's c_embed_data rule.
#
# Parameters:
# PACKAGE: Name of the package (overrides actual path)
# NAME: Name of target (see Note).
# SRCS: List of files to embed (in source or build directory).
# INCLUDES: Include directories to add to dependencies
# C_FILE_OUTPUT: The C implementation file to output.
# H_FILE_OUTPUT: The H header file to output.
# STRIP_PREFIX: Strips this verbatim prefix from filenames (in the TOC).
# FLATTEN: Removes all directory components from filenames (in the TOC).
# IDENTIFIER: The identifier to use in generated names (defaults to name).
# PUBLIC: Add this so that this library will be exported under ${PACKAGE}::
# Also in IDE, target will appear in ${PACKAGE} folder while non PUBLIC will be
# in ${PACKAGE}/internal.
# TESTONLY: When added, this target will only be built if user passes
#    -DIREE_BUILD_TESTS=ON to CMake.
# TODO(scotttodd): Support passing KWARGS down into iree_cc_library?
#
function(iree_c_embed_data)
  cmake_parse_arguments(
    _RULE
    "PUBLIC;TESTONLY;FLATTEN"
    "PACKAGE;NAME;IDENTIFIER;STRIP_PREFIX;C_FILE_OUTPUT;H_FILE_OUTPUT"
    "DEPS;SRCS;INCLUDES"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()

  if(DEFINED _RULE_IDENTIFIER)
    set(_IDENTIFIER ${_RULE_IDENTIFIER})
  else()
    set(_IDENTIFIER ${_RULE_NAME})
  endif()

  set(_ARGS)
  list(APPEND _ARGS "--output_header=${_RULE_H_FILE_OUTPUT}")
  list(APPEND _ARGS "--output_impl=${_RULE_C_FILE_OUTPUT}")
  list(APPEND _ARGS "--identifier=${_IDENTIFIER}")
  if(DEFINED _RULE_STRIP_PREFIX)
    list(APPEND _ARGS "--strip_prefix=${_RULE_STRIP_PREFIX}")
  endif()
  if(_RULE_FLATTEN)
    list(APPEND _ARGS "--flatten")
  endif()

  set(_RELATIVE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  cmake_path(RELATIVE_PATH _RELATIVE_BINARY_DIR BASE_DIRECTORY "${IREE_BINARY_DIR}")

  foreach(_SRC ${_RULE_SRCS})
    if(_SRC MATCHES "^/")
      # _SRC is an absolute path (starts with `/`).
      list(APPEND _RESOLVED_SRCS "${_SRC}")
    elseif(_SRC MATCHES "^[$]<")
      # _SRC is a CMake generator expression (starts with `$<`).
      list(APPEND _RESOLVED_SRCS "${_SRC}")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_SRC}")
      # _SRC path exists relatively to current source dir.
      list(APPEND _RESOLVED_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/${_SRC}")
    elseif(EXISTS "${IREE_SOURCE_DIR}/${_SRC}")
      # _SRC path exists relatively to root source dir.
      list(APPEND _RESOLVED_SRCS "${IREE_SOURCE_DIR}/${_SRC}")
    else()
      # All else having failed, interpret _SRC as a path in the binary dir,
      # that is, a generated file. As the present logic executes before
      # that file would be generated, we can't perform a if(EXISTS) test to
      # validate that hypothesis.
      #
      # Additional difficulty: _SRC could be a path relative to either the
      # current binary dir, or the root binary dir. Again, it's too early here
      # to use a if(EXISTS) test to determine that. Instead, the following
      # regex replacement strips the current binary dir as a prefix from _SRC.
      # So if _SRC was relative to the root binary dir, now it is relative to
      # the current binary dir. And if it was already relative to the current
      # binary dir, then the regex should fail to match (unless we're very
      # unlucky and we have paths of the from root/a/b/a/b, but that's not a
      # problem that we have today. And if that ever happens, people will still
      # be able to get the correct behavior by passing a path relative to the
      # root dir).
      string(REGEX REPLACE "^${_RELATIVE_BINARY_DIR}/" "" _SRC_RELATIVE "${_SRC}")
      list(APPEND _RESOLVED_SRCS "${CMAKE_CURRENT_BINARY_DIR}/${_SRC_RELATIVE}")
    endif()
  endforeach(_SRC)

  add_custom_command(
    OUTPUT "${_RULE_H_FILE_OUTPUT}" "${_RULE_C_FILE_OUTPUT}"
    COMMAND generate_embed_data ${_ARGS} ${_RESOLVED_SRCS}
    DEPENDS generate_embed_data ${_RESOLVED_SRCS}
  )

  if(_RULE_TESTONLY)
    set(_TESTONLY_ARG "TESTONLY")
  endif()
  if(_RULE_PUBLIC)
    set(_PUBLIC_ARG "PUBLIC")
  endif()

  iree_cc_library(
    PACKAGE ${_RULE_PACKAGE}
    NAME ${_RULE_NAME}
    HDRS "${_RULE_H_FILE_OUTPUT}"
    SRCS "${_RULE_C_FILE_OUTPUT}"
    DEPS "${_RULE_DEPS}"
    INCLUDES ${_RULE_INCLUDES}
    "${_PUBLIC_ARG}"
    "${_TESTONLY_ARG}"
  )
endfunction()
