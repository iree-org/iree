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

cmake_minimum_required(VERSION 3.13)

project(IreeNative)

set(IREE_ROOT_DIR /usr/local/google/home/jennik/github/iree/)

function(iree_package_ns PACKAGE_NS)
  string(REPLACE ${IREE_ROOT_DIR} "" _PACKAGE ${CMAKE_CURRENT_LIST_DIR})
  string(SUBSTRING ${_PACKAGE} 1 -1 _PACKAGE)
  string(REPLACE "/" "::" _PACKAGE_NS ${_PACKAGE})
  set(${PACKAGE_NS} ${_PACKAGE_NS} PARENT_SCOPE)
endfunction()

function(iree_package_name PACKAGE_NAME)
  iree_package_ns(_PACKAGE_NS)
  string(REPLACE "::" "_" _PACKAGE_NAME ${_PACKAGE_NS})
  set(${PACKAGE_NAME} ${_PACKAGE_NAME} PARENT_SCOPE)
endfunction()

function(iree_package_dir PACKAGE_DIR)
  iree_package_ns(_PACKAGE_NS)
  string(FIND ${_PACKAGE_NS} "::" _END_OFFSET REVERSE)
  math(EXPR _END_OFFSET "${_END_OFFSET} + 2")
  string(SUBSTRING ${_PACKAGE_NS} ${_END_OFFSET} -1 _PACKAGE_DIR)
  set(${PACKAGE_DIR} ${_PACKAGE_DIR} PARENT_SCOPE)
endfunction()

function(iree_add_data_dependencies)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "DATA"
    ${ARGN}
  )

  if(NOT _RULE_DATA)
    return()
  endif()

  foreach(_DATA_LABEL ${_RULE_DATA})
    if(TARGET ${_DATA_LABEL})
      add_dependencies(${_RULE_NAME} ${_DATA_LABEL})
    else()
      # Not a target, assume to be a file instead.
      string(REPLACE "::" "/" _FILE_PATH ${_DATA_LABEL})

      # Create a target which copies the data file into the build directory.
      # If this file is included in multiple rules, only create the target once.
      string(REPLACE "::" "_" _DATA_TARGET ${_DATA_LABEL})
      if(NOT TARGET ${_DATA_TARGET})
        set(_INPUT_PATH "${CMAKE_SOURCE_DIR}/${_FILE_PATH}")
        set(_OUTPUT_PATH "${CMAKE_BINARY_DIR}/${_FILE_PATH}")
        add_custom_target(${_DATA_TARGET}
          COMMAND ${CMAKE_COMMAND} -E copy ${_INPUT_PATH} ${_OUTPUT_PATH}
        )
      endif()

      add_dependencies(${_RULE_NAME} ${_DATA_TARGET})
    endif()
  endforeach()
endfunction()

function(iree_cc_library)
  cmake_parse_arguments(
    _RULE
    "PUBLIC;ALWAYSLINK;TESTONLY;SHARED;WHOLEARCHIVE"
    "NAME"
    "HDRS;TEXTUAL_HDRS;SRCS;COPTS;DEFINES;LINKOPTS;DATA;DEPS;INCLUDES"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()

  iree_package_ns(_PACKAGE_NS)
  # Replace dependencies passed by ::name with ::iree::package::name
  list(TRANSFORM _RULE_DEPS REPLACE "^::" "${_PACKAGE_NS}::")
  list(TRANSFORM _RULE_DATA REPLACE "^::" "${_PACKAGE_NS}::")

  # Prefix the library with the package name, so we get: iree_package_name.
  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  # Check if this is a header-only library.
  # Note that as of February 2019, many popular OS's (for example, Ubuntu
  # 16.04 LTS) only come with cmake 3.5 by default.  For this reason, we can't
  # use list(FILTER...)
  set(_CC_SRCS "${_RULE_SRCS}")
  foreach(src_file IN LISTS _CC_SRCS)
    if(${src_file} MATCHES ".*\\.(h|inc)")
      list(REMOVE_ITEM _CC_SRCS "${src_file}")
    endif()
  endforeach()
  if("${_CC_SRCS}" STREQUAL "")
    set(_RULE_IS_INTERFACE 1)
  else()
    set(_RULE_IS_INTERFACE 0)
  endif()

  if(NOT _RULE_IS_INTERFACE)
    if (_RULE_SHARED)
      add_library(${_NAME} SHARED "")
    else()
      add_library(${_NAME} STATIC "")
      if (_RULE_WHOLEARCHIVE)
        message(FATAL_ERROR "WHOLEARCHIVE must be set together with SHARED")
      endif()
    endif()

    target_sources(${_NAME}
      PRIVATE
        ${_RULE_SRCS}
        ${_RULE_TEXTUAL_HDRS}
        ${_RULE_HDRS}
    )
    target_include_directories(${_NAME} SYSTEM
      PUBLIC
        "$<BUILD_INTERFACE:${IREE_COMMON_INCLUDE_DIRS}>"
    )
    target_include_directories(${_NAME}
      PUBLIC
        "$<BUILD_INTERFACE:${_RULE_INCLUDES}>"
    )
    target_compile_options(${_NAME}
      PRIVATE
        ${_RULE_COPTS}
        ${IREE_DEFAULT_COPTS}
    )

  if(_RULE_WHOLEARCHIVE)
      iree_whole_archive_link(${_NAME} ${_RULE_DEPS})
    else()
      target_link_libraries(${_NAME} PUBLIC ${_RULE_DEPS})
    endif()
    target_link_libraries(${_NAME}
      PRIVATE
        ${_RULE_LINKOPTS}
        ${IREE_DEFAULT_LINKOPTS}
    )

    iree_add_data_dependencies(NAME ${_NAME} DATA ${_RULE_DATA})
    target_compile_definitions(${_NAME}
      PUBLIC
        ${_RULE_DEFINES}
    )

    if(DEFINED _RULE_ALWAYSLINK)
      set_property(TARGET ${_NAME} PROPERTY ALWAYSLINK 1)
    endif()

    # Add all IREE targets to a folder in the IDE for organization.
    if(_RULE_PUBLIC)
      set_property(TARGET ${_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER})
    elseif(_RULE_TESTONLY)
      set_property(TARGET ${_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER}/test)
    else()
      set_property(TARGET ${_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER}/internal)
    endif()

    # INTERFACE libraries can't have the CXX_STANDARD property set.
    set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD ${IREE_CXX_STANDARD})
    set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
  else()
    # Generating header-only library.
    add_library(${_NAME} INTERFACE)
    target_include_directories(${_NAME} SYSTEM
      INTERFACE
        "$<BUILD_INTERFACE:${IREE_COMMON_INCLUDE_DIRS}>"
    )
    target_compile_options(${_NAME}
      INTERFACE
        ${_RULE_COPTS}
        ${IREE_DEFAULT_COPTS}
    )
    target_link_libraries(${_NAME}
      INTERFACE
        ${_RULE_DEPS}
        ${_RULE_LINKOPTS}
        ${IREE_DEFAULT_LINKOPTS}
    )
    iree_add_data_dependencies(NAME ${_NAME} DATA ${_RULE_DATA})
    target_compile_definitions(${_NAME}
      INTERFACE
        ${_RULE_DEFINES}
    )
  endif()

  # Alias the iree_package_name library to iree::package::name.
  # This lets us more clearly map to Bazel and makes it possible to
  # disambiguate the underscores in paths vs. the separators.
  add_library(${_PACKAGE_NS}::${_RULE_NAME} ALIAS ${_NAME})
  iree_package_dir(_PACKAGE_DIR)
  if(${_RULE_NAME} STREQUAL ${_PACKAGE_DIR})
    # If the library name matches the package then treat it as a default.
    # For example, foo/bar/ library 'bar' would end up as 'foo::bar'.
    add_library(${_PACKAGE_NS} ALIAS ${_NAME})
  endif()
endfunction()

iree_cc_library(
  NAME
    iree_cc_wrappers
  SRCS
     "context_wrapper.cc"
     "function_wrapper.cc"
     "instance_wrapper.cc"
     "module_wrapper.cc"
  HDRS
    "context_wrapper.h"
    "function_wrapper.h"
    "instance_wrapper.h"
    "module_wrapper.h"
  DEPS
    iree::base::status
    iree::base::init
    iree::base::logging
    iree::modules::hal::hal
    iree::modules::strings::strings_module
    iree::modules::tensorlist::native_module
    iree::vm::instance
    iree::vm::bytecode_module
    iree::base::api
    iree::hal::api
    iree::vm::context
    iree::vm::ref_cc
    iree::hal::vmla::vmla_driver_module
    iree::modules::hal
)