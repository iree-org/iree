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

# iree_cc_binary()
#
# CMake function to imitate Bazel's cc_binary rule.
#
# Parameters:
# NAME: name of target (see Usage below)
# SRCS: List of source files for the binary
# DEPS: List of other libraries to be linked in to the binary targets
# COPTS: List of private compile options
# DEFINES: List of public defines
# LINKOPTS: List of link options
#
# Note:
# By default, iree_cc_binary will always create a binary named iree_${NAME}.
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
# iree_cc_binary(
#   NAME
#     awesome_tool
#   OUT
#     awesome-tool
#   SRCS
#     "awesome_tool_main.cc"
#   DEPS
#     iree::awesome
# )
function(iree_cc_binary)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;OUT"
    "SRCS;COPTS;DEFINES;LINKOPTS;DEPS"
    ${ARGN}
  )

  # Prefix the library with the package name, so we get: iree_package_name
  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  add_executable(${_NAME} "")
  if(_RULE_SRCS)
    target_sources(${_NAME}
      PRIVATE
        ${_RULE_SRCS}
    )
  else()
    set(_DUMMY_SRC "${CMAKE_CURRENT_BINARY_DIR}/${_NAME}_dummy.cc")
    file(WRITE ${_DUMMY_SRC} "")
    target_sources(${_NAME}
      PRIVATE
        ${_DUMMY_SRC}
    )
  endif()
  if(_RULE_OUT)
    set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "${_RULE_OUT}")
  endif()
  target_include_directories(${_NAME}
    PUBLIC
      ${IREE_COMMON_INCLUDE_DIRS}
    PRIVATE
      ${GTEST_INCLUDE_DIRS}
  )
  target_compile_definitions(${_NAME}
    PUBLIC
      ${_RULE_DEFINES}
  )
  target_compile_options(${_NAME}
    PRIVATE
      ${_RULE_COPTS}
  )

  # Split DEPS into two lists - one for whole archive (alwayslink) and one for
  # standard linking (which only links in symbols that are directly used).
  #
  # TODO(scotttodd): traverse transitive dependencies also (all direct
  #                  dependencies of ALWAYSLINK libraries should be linked in)
  set(_ALWAYS_LINK_DEPS "")
  set(_STANDARD_DEPS "")
  foreach(_DEP ${_RULE_DEPS})
    # Check if _DEP is a library with the ALWAYSLINK property set.
    get_target_property(_DEP_TYPE ${_DEP} TYPE)
    if(${_DEP_TYPE} STREQUAL "INTERFACE_LIBRARY")
      # Can't be ALWAYSLINK since it's an INTERFACE library.
      # We also can't even query for the property, since it isn't whitelisted.
      set(_DEP_IS_ALWAYSLINK OFF)
    else()
      get_target_property(_DEP_IS_ALWAYSLINK ${_DEP} ALWAYSLINK)
    endif()

    # Append to the corresponding list of deps.
    if(_DEP_IS_ALWAYSLINK)
      list(APPEND _ALWAYS_LINK_DEPS ${_DEP})

      # For MSVC, also add a `-WHOLEARCHIVE:` version of the dep.
      # CMake treats -WHOLEARCHIVE[:lib] as a link flag and will not actually
      # try to link the library in, so we need the flag *and* the dependency.
      if(MSVC)
        get_target_property(_ALIASED_TARGET ${_DEP} ALIASED_TARGET)
        if (_ALIASED_TARGET)
          list(APPEND _ALWAYS_LINK_DEPS "-WHOLEARCHIVE:${_ALIASED_TARGET}")
        else()
          list(APPEND _ALWAYS_LINK_DEPS "-WHOLEARCHIVE:${_DEP}")
        endif()
      endif()
    else()
      list(APPEND _STANDARD_DEPS ${_DEP})
    endif()
  endforeach(_DEP)

  # Call into target_link_libraries with the lists of deps.
  # TODO(scotttodd): `-Wl,-force_load` version
  if(MSVC)
    target_link_libraries(${_NAME}
      PUBLIC
        ${_ALWAYS_LINK_DEPS}
        ${_STANDARD_DEPS}
      PRIVATE
        ${_RULE_LINKOPTS}
    )
  else()
    target_link_libraries(${_NAME}
      PUBLIC
        "-Wl,--whole-archive"
        ${_ALWAYS_LINK_DEPS}
        "-Wl,--no-whole-archive"
        ${_STANDARD_DEPS}
      PRIVATE
        ${_RULE_LINKOPTS}
    )
  endif()

  # Add all IREE targets to a folder in the IDE for organization.
  set_property(TARGET ${_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER}/binaries)

  set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD ${IREE_CXX_STANDARD})
  set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

endfunction()
