# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(FetchContent)

# disable some targets we don't use
set(protobuf_INSTALL OFF)
set(protobuf_BUILD_TESTS OFF)

# to prevent protobuf itself from using `find_package`
set(protobuf_FORCE_FETCH_DEPENDENCIES ON)

# pin the version of protobuf and the commit its tag must point to
set(protobuf_VERSION 29.1)
set(protobuf_COMMIT 796e49f6ca82f763d1087d2ff7355d2f0d7b71b1)

# A shallow clone needs a tag rather than a commit, so only populate the
# source here and check the commit before adding it to the build.
FetchContent_Declare(
  protobuf
  GIT_REPOSITORY https://github.com/protocolbuffers/protobuf
  GIT_TAG v${protobuf_VERSION}
  GIT_SHALLOW ON
  SOURCE_SUBDIR do-not-add-subdirectory
)

FetchContent_MakeAvailable(protobuf)

# skip the check for a local source dir set with FETCHCONTENT_SOURCE_DIR_PROTOBUF
if(NOT FETCHCONTENT_SOURCE_DIR_PROTOBUF)
  find_package(Git REQUIRED)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
    WORKING_DIRECTORY ${protobuf_SOURCE_DIR}
    OUTPUT_VARIABLE _protobuf_commit
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  if(NOT _protobuf_commit STREQUAL protobuf_COMMIT)
    message(FATAL_ERROR
      "protobuf v${protobuf_VERSION} was fetched at commit ${_protobuf_commit}, "
      "expected ${protobuf_COMMIT}")
  endif()
endif()

add_subdirectory(${protobuf_SOURCE_DIR} ${protobuf_BINARY_DIR})

# make protobuf_generate() function available
include(${protobuf_SOURCE_DIR}/cmake/protobuf-generate.cmake)

# iree_pjrt_protobuf_cc_library()
#
# CMake function to invoke the protoc compiler.
#
# Parameters:
# NAME: name of target
# SRCS: List of source files for the library
# PROTOC_ARGS: List of protoc arguments.
# PUBLIC: Add this so that this library will be exported under iree::
# Also in IDE, target will appear in IREE folder while non PUBLIC will be in IREE/internal.
# TESTONLY: When added, this target will only be built if user passes -DIREE_BUILD_TESTS=ON to CMake.
#
# iree_pjrt_protobuf_cc_library(
#   NAME
#     some_def
#   SRC
#     some_def.proto
#   PUBLIC
# )
function(iree_pjrt_protobuf_cc_library)
  cmake_parse_arguments(_RULE
    "PUBLIC;TESTONLY"
    "NAME"
    "SRCS;PROTOC_ARGS"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()

  # Prefix the library with the package name, so we get: iree_package_name
  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  add_library(${_NAME} ${_RULE_SRCS})
  protobuf_generate(
    TARGET ${_NAME}
    LANGUAGE cpp
    PROTOC_OPTIONS ${_RULE_PROTOC_ARGS}
    IMPORT_DIRS
      ${protobuf_SOURCE_DIR}/src
      ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_include_directories(${_NAME}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )
  target_link_libraries(${_NAME}
    PUBLIC
    protobuf::libprotobuf
    ${IREE_DEFAULT_LINKOPTS}
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
