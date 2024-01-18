# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

# flatbuffer_c_library()
#
# CMake function to invoke the flatcc compiler.
#
# Parameters:
# NAME: name of target (see Note)
# SRCS: List of source files for the library
# FLATCC_ARGS: List of flattbuffers arguments. Default:
#             "--common"
#             "--reader"
# PUBLIC: Add this so that this library will be exported under iree::
# Also in IDE, target will appear in IREE folder while non PUBLIC will be in IREE/internal.
# TESTONLY: When added, this target will only be built if user passes -DIREE_BUILD_TESTS=ON to CMake.
#
# Note:
# By default, flatbuffer_c_library will always create a library named ${NAME},
# and alias target iree::${NAME}. The iree:: form should always be used.
# This is to reduce namespace pollution.
#
# flatbuffer_c_library(
#   NAME
#     some_def
#   SRCS
#     "some_def.fbs"
#   FLATCC_ARGS
#     "--reader"
#     "--builder"
#     "--verifier"
#     "--json"
#   PUBLIC
# )
# iree_cc_binary(
#   NAME
#     main_lib
#   ...
#   DEPS
#     iree::schemas::some_def
# )
function(flatbuffer_c_library)
  cmake_parse_arguments(_RULE
    "PUBLIC;TESTONLY"
    "NAME"
    "SRCS;FLATCC_ARGS"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()

  # Prefix the library with the package name, so we get: iree_package_name
  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  if(NOT DEFINED _RULE_FLATCC_ARGS)
    set(_RULE_FLATCC_ARGS
      "--common"
      "--reader"
    )
  else()
    set(_RULE_FLATCC_ARGS ${_RULE_FLATCC_ARGS})
  endif()

  set(_OUTS "")
  foreach(_SRC ${_RULE_SRCS})
    get_filename_component(_SRC_FILENAME ${_SRC} NAME_WE)
    foreach(_ARG ${_RULE_FLATCC_ARGS})
      if(_ARG STREQUAL "--reader")
        list(APPEND _OUTS "${_SRC_FILENAME}_reader.h")
      elseif(_ARG STREQUAL "--builder")
        list(APPEND _OUTS "${_SRC_FILENAME}_builder.h")
      elseif(_ARG STREQUAL "--verifier")
        list(APPEND _OUTS "${_SRC_FILENAME}_verifier.h")
      elseif(_ARG STREQUAL "--json")
        list(APPEND _OUTS "${_SRC_FILENAME}_json_printer.h")
        list(APPEND _OUTS "${_SRC_FILENAME}_json_parser.h")
      endif()
    endforeach()
  endforeach()
  list(TRANSFORM _OUTS PREPEND "${CMAKE_CURRENT_BINARY_DIR}/")

  add_custom_command(
    OUTPUT
      ${_OUTS}
    COMMAND
      iree-flatcc-cli
          -o "${CMAKE_CURRENT_BINARY_DIR}"
          -I "${IREE_ROOT_DIR}"
          ${_RULE_FLATCC_ARGS}
          "${_RULE_SRCS}"
    WORKING_DIRECTORY
      "${CMAKE_CURRENT_SOURCE_DIR}"
    MAIN_DEPENDENCY
      ${_RULE_SRCS}
    DEPENDS
      ${_RULE_SRCS}
    COMMAND_EXPAND_LISTS
  )

  set(_GEN_TARGET "${_NAME}_gen")
  add_custom_target(
    ${_GEN_TARGET}
    DEPENDS
      ${_OUTS}
  )

  add_library(${_NAME} INTERFACE)
  add_dependencies(${_NAME} ${_GEN_TARGET})
  target_include_directories(${_NAME}
    INTERFACE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    )
  target_link_libraries(${_NAME}
    INTERFACE
      ${IREE_DEFAULT_LINKOPTS}
  )
  target_compile_options(${_NAME}
    INTERFACE
      "-I${IREE_ROOT_DIR}/third_party/flatcc/include/"
      "-I${IREE_ROOT_DIR}/third_party/flatcc/include/flatcc/reflection/"
  )
  iree_install_targets(
    TARGETS ${_NAME}
  )
  
  # Alias the iree_package_name library to iree::package::name.
  # This lets us more clearly map to Bazel and makes it possible to
  # disambiguate the underscores in paths vs. the separators.
  iree_package_ns(_PACKAGE_NS)
  iree_add_alias_library(${_PACKAGE_NS}::${_RULE_NAME} ${_NAME})
endfunction()
