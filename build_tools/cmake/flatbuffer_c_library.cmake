# Copyright 2020 Google LLC
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

# flatbuffer_c_library()
#
# CMake function to invoke the flatcc compiler.
#
# Parameters:
# NAME: name of target (see Note)
# SRCS: List of source files for the library
# DEPS: List of other libraries to be linked in to the binary targets
# COPTS: List of private compile options
# DEFINES: List of public defines
# LINKOPTS: List of link options
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
#     base_schema
#   SRCS
#     "a.fbs"
# )
# flatbuffer_c_library(
#   NAME
#     other_schemas
#   SRCS
#     "b.fbs"
#   DEPS
#     iree::schemas::base_schema
#   PUBLIC
# )
#
# iree_cc_binary(
#   NAME
#     main_lib
#   ...
#   DEPS
#     iree::schemas::other_schemas
# )
function(flatbuffer_c_library)
  cmake_parse_arguments(_RULE
    "PUBLIC;TESTONLY"
    "NAME"
    "SRCS;COPTS;DEFINES;LINKOPTS;DEPS;FLATCC_ARGS"
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
      endif()
    endforeach()
  endforeach()
  list(TRANSFORM _OUTS PREPEND "${CMAKE_CURRENT_BINARY_DIR}/")

  iree_get_executable_path(_FLATCC_BIN flatcc_cli)
  add_custom_command(
    OUTPUT
      ${_OUTS}
    COMMAND
      "${_FLATCC_BIN}"
          -o "${CMAKE_CURRENT_BINARY_DIR}"
          -I "${IREE_ROOT_DIR}"
          ${_RULE_FLATCC_ARGS}
          "${_RULE_SRCS}"
    WORKING_DIRECTORY
      "${CMAKE_CURRENT_SOURCE_DIR}"
    MAIN_DEPENDENCY
      ${_RULE_SRCS}
    DEPENDS
      iree_host_flatcc_cli
      ${_RULE_SRCS}
    COMMAND_EXPAND_LISTS
  )

  set(_GEN_TARGET "${_NAME}_gen")
  add_custom_target(
    ${_GEN_TARGET}
    DEPENDS
      ${_OUTS}
      ${_RULE_DEPS}
  )

  add_library(${_NAME} INTERFACE)
  add_dependencies(${_NAME} ${_GEN_TARGET})
  target_include_directories(${_NAME}
    INTERFACE
      "$<BUILD_INTERFACE:${IREE_COMMON_INCLUDE_DIRS}>"
      ${CMAKE_CURRENT_BINARY_DIR}
    )
  target_link_libraries(${_NAME}
    INTERFACE
      flatcc::runtime
      ${_RULE_LINKOPTS}
      ${IREE_DEFAULT_LINKOPTS}
  )
  target_compile_definitions(${_NAME}
    INTERFACE
      ${_RULE_DEFINES}
  )
  target_compile_options(${_NAME}
    INTERFACE
      "-I${IREE_ROOT_DIR}/third_party/flatcc/include/flatcc/reflection/"
  )

  # Alias the iree_package_name library to iree::package::name.
  # This lets us more clearly map to Bazel and makes it possible to
  # disambiguate the underscores in paths vs. the separators.
  iree_package_ns(_PACKAGE_NS)
  add_library(${_PACKAGE_NS}::${_RULE_NAME} ALIAS ${_NAME})
endfunction()
