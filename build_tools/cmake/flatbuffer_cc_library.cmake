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

include(BuildFlatBuffers)
include(CMakeParseArguments)

# flatbuffer_cc_library()
#
# CMake function to imitate Bazel's flatbuffer_cc_library rule.
#
# Parameters:
# NAME: name of target (see Note)
# SRCS: List of source files for the library
# DEPS: List of other libraries to be linked in to the binary targets
# COPTS: List of private compile options
# DEFINES: List of public defines
# LINKOPTS: List of link options
# FLATC_ARGS: List of flattbuffers arguments. Default:
#             "--keep-prefix"
#             "--scoped-enums"
#             "--reflect-names"
#             "--gen-object-api"
# PUBLIC: Add this so that this library will be exported under iree::
# Also in IDE, target will appear in IREE folder while non PUBLIC will be in IREE/internal.
# TESTONLY: When added, this target will only be built if user passes -DIREE_BUILD_TESTS=ON to CMake.
#
# Note:
# By default, flatbuffer_cc_library will always create a library named ${NAME},
# and alias target iree::${NAME}. The iree:: form should always be used.
# This is to reduce namespace pollution.
#
# flatbuffer_cc_library(
#   NAME
#     base_schema
#   SRCS
#     "a.cc"
# )
# flatbuffer_cc_library(
#   NAME
#     other_schemas
#   SRCS
#     "b.fbs"
#   DEPS
#     iree::schemas::base_schema # not "awesome" !
#   PUBLIC
# )
#
# flatbuffer_cc_library(
#   NAME
#     main_lib
#   ...
#   DEPS
#     iree::schemas::other_schemas
# )
function(flatbuffer_cc_library)
  cmake_parse_arguments(_RULE
    "PUBLIC;TESTONLY"
    "NAME"
    "SRCS;COPTS;DEFINES;LINKOPTS;DEPS;FLATC_ARGS"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()

  # Prefix the library with the package name, so we get: iree_package_name
  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  if(NOT DEFINED _RULE_FLATC_ARGS)
    set(FLATBUFFERS_FLATC_SCHEMA_EXTRA_ARGS
      # Preserve root-relative include paths in generated code.
      "--keep-prefix"
      # Use C++11 'enum class' for enums.
      "--scoped-enums"
      # Include reflection tables used for dumping debug representations.
      "--reflect-names"
      # Generate FooT types for unpack/pack support. Note that this should only
      # be used in tooling as the code size/runtime overhead is non-trivial.
      "--gen-object-api"
    )
  else()
    set(FLATBUFFERS_FLATC_SCHEMA_EXTRA_ARGS ${_RULE_FLATC_ARGS})
  endif()

  set(_GEN_TARGET "${_NAME}_gen")

  build_flatbuffers(
    "${_RULE_SRCS}"
    "${IREE_ROOT_DIR}"
    "${_GEN_TARGET}" # custom_target_name
    "${_RULE_DEPS}"  # additional_dependencies
    "${CMAKE_CURRENT_BINARY_DIR}" # generated_include_dir
    "${CMAKE_CURRENT_BINARY_DIR}" # binary_schemas_dir
    "" # copy_text_schemas_dir
  )

  # Add dependency on flatc explicitly. This is needed for cross-compiling
  # where flatc comes from another CMake invocation for host.
  iree_add_executable_dependencies(${_GEN_TARGET} flatc)

  add_library(${_NAME} INTERFACE)
  add_dependencies(${_NAME} ${_GEN_TARGET})
  target_include_directories(${_NAME}
    INTERFACE
      "$<BUILD_INTERFACE:${IREE_COMMON_INCLUDE_DIRS}>"
      ${CMAKE_CURRENT_BINARY_DIR}
    )
  target_link_libraries(${_NAME}
    INTERFACE
      flatbuffers
      ${_RULE_LINKOPTS}
      ${IREE_DEFAULT_LINKOPTS}
  )
  target_compile_definitions(${_NAME}
    INTERFACE
      ${_RULE_DEFINES}
  )

  # Alias the iree_package_name library to iree::package::name.
  # This lets us more clearly map to Bazel and makes it possible to
  # disambiguate the underscores in paths vs. the separators.
  iree_package_ns(_PACKAGE_NS)
  add_library(${_PACKAGE_NS}::${_RULE_NAME} ALIAS ${_NAME})
endfunction()
