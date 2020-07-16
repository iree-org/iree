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

#-------------------------------------------------------------------------------
# Missing CMake Variables
#-------------------------------------------------------------------------------

if(${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Windows")
  set(IREE_HOST_SCRIPT_EXT "bat")
  # https://gitlab.kitware.com/cmake/cmake/-/issues/17553
  set(IREE_HOST_EXECUTABLE_SUFFIX ".exe")
else()
  set(IREE_HOST_SCRIPT_EXT "sh")
  set(IREE_HOST_EXECUTABLE_SUFFIX "")
endif()

#-------------------------------------------------------------------------------
# General utilities
#-------------------------------------------------------------------------------

# iree_to_bool
#
# Sets `variable` to `ON` if `value` is true and `OFF` otherwise.
function(iree_to_bool VARIABLE VALUE)
  if(VALUE)
    set(${VARIABLE} "ON" PARENT_SCOPE)
  else()
    set(${VARIABLE} "OFF" PARENT_SCOPE)
  endif()
endfunction()


#-------------------------------------------------------------------------------
# Packages and Paths
#-------------------------------------------------------------------------------

# Sets ${PACKAGE_NS} to the IREE-root relative package name in C++ namespace
# format (::).
#
# Example when called from iree/base/CMakeLists.txt:
#   iree::base
function(iree_package_ns PACKAGE_NS)
  string(REPLACE ${IREE_ROOT_DIR} "" _PACKAGE ${CMAKE_CURRENT_LIST_DIR})
  string(SUBSTRING ${_PACKAGE} 1 -1 _PACKAGE)
  string(REPLACE "/" "::" _PACKAGE_NS ${_PACKAGE})
  set(${PACKAGE_NS} ${_PACKAGE_NS} PARENT_SCOPE)
endfunction()

# Sets ${PACKAGE_NAME} to the IREE-root relative package name.
#
# Example when called from iree/base/CMakeLists.txt:
#   iree_base
function(iree_package_name PACKAGE_NAME)
  iree_package_ns(_PACKAGE_NS)
  string(REPLACE "::" "_" _PACKAGE_NAME ${_PACKAGE_NS})
  set(${PACKAGE_NAME} ${_PACKAGE_NAME} PARENT_SCOPE)
endfunction()

# Sets ${PACKAGE_PATH} to the IREE-root relative package path.
#
# Example when called from iree/base/CMakeLists.txt:
#   iree/base
function(iree_package_path PACKAGE_PATH)
  iree_package_ns(_PACKAGE_NS)
  string(REPLACE "::" "/" _PACKAGE_PATH ${_PACKAGE_NS})
  set(${PACKAGE_PATH} ${_PACKAGE_PATH} PARENT_SCOPE)
endfunction()

# Sets ${PACKAGE_DIR} to the directory name of the current package.
#
# Example when called from iree/base/CMakeLists.txt:
#   base
function(iree_package_dir PACKAGE_DIR)
  iree_package_ns(_PACKAGE_NS)
  string(FIND ${_PACKAGE_NS} "::" _END_OFFSET REVERSE)
  math(EXPR _END_OFFSET "${_END_OFFSET} + 2")
  string(SUBSTRING ${_PACKAGE_NS} ${_END_OFFSET} -1 _PACKAGE_DIR)
  set(${PACKAGE_DIR} ${_PACKAGE_DIR} PARENT_SCOPE)
endfunction()

# iree_get_executable_path
#
# Gets the path to an executable in a cross-compilation-aware way. This
# should be used when accessing binaries that are used as part of the build,
# such as for generating files used for later build steps. Those binaries
# can come from third-party projects or another CMake invocation.
#
# Paramters:
# - OUTPUT_PATH_VAR: variable name for receiving the path to the built target.
# - EXECUTABLE: the executable to get its path.
function(iree_get_executable_path OUTPUT_PATH_VAR EXECUTABLE)
  if(CMAKE_CROSSCOMPILING)
    # The target is defined in the CMake invocation for host. We don't have
    # access to the target; relying on the path here.
    set(_OUTPUT_PATH "${IREE_HOST_BINARY_ROOT}/bin/${EXECUTABLE}${IREE_HOST_EXECUTABLE_SUFFIX}")
    set(${OUTPUT_PATH_VAR} "${_OUTPUT_PATH}" PARENT_SCOPE)
  else()
    # The target is defined in this CMake invocation. We can query the location
    # directly from CMake.
    set(${OUTPUT_PATH_VAR} "$<TARGET_FILE:${EXECUTABLE}>" PARENT_SCOPE)
  endif()
endfunction()

# iree_get_target_path
#
# Gets the path to a target in a cross-compilation-aware way. This should be
# used when accessing targets that are used as part of the build, such as for
# generating files used for later build steps. Those targets should be defined
# inside IREE itself.
#
# Paramters:
# - OUTPUT_VAR: variable name for receiving the path to the built target.
# - TARGET: the target to get its path.
function(iree_get_target_path OUTPUT_VAR TARGET)
  # If this is a host target for cross-compilation, it should have a
  # `HOST_TARGET_FILE` property containing the artifact's path.
  # Otherwise it must be a target defined in the current CMake invocation
  # and we can just use `$<TARGET_FILE:${TARGET}>` on it.
  set(${OUTPUT_VAR}
      "$<IF:$<BOOL:$<TARGET_PROPERTY:${TARGET},HOST_TARGET_FILE>>,$<TARGET_PROPERTY:${TARGET},HOST_TARGET_FILE>,$<TARGET_FILE:${TARGET}>>"
      PARENT_SCOPE)
endfunction()

#-------------------------------------------------------------------------------
# select()-like Evaluation
#-------------------------------------------------------------------------------

# Appends ${OPTS} with a list of values based on the current compiler.
#
# Example:
#   iree_select_compiler_opts(COPTS
#     CLANG
#       "-Wno-foo"
#       "-Wno-bar"
#     CLANG_CL
#       "/W3"
#     GCC
#       "-Wsome-old-flag"
#     MSVC
#       "/W3"
#   )
#
# Note that variables are allowed, making it possible to share options between
# different compiler targets.
function(iree_select_compiler_opts OPTS)
  cmake_parse_arguments(
    PARSE_ARGV 1
    _IREE_SELECTS
    ""
    ""
    "ALL;CLANG;CLANG_CL;MSVC;GCC;CLANG_OR_GCC;MSVC_OR_CLANG_CL"
  )
  set(_OPTS)
  list(APPEND _OPTS "${_IREE_SELECTS_ALL}")
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    list(APPEND _OPTS "${_IREE_SELECTS_GCC}")
    list(APPEND _OPTS "${_IREE_SELECTS_CLANG_OR_GCC}")
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    if(MSVC)
      list(APPEND _OPTS ${_IREE_SELECTS_CLANG_CL})
      list(APPEND _OPTS ${_IREE_SELECTS_MSVC_OR_CLANG_CL})
    else()
      list(APPEND _OPTS ${_IREE_SELECTS_CLANG})
      list(APPEND _OPTS ${_IREE_SELECTS_CLANG_OR_GCC})
    endif()
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    list(APPEND _OPTS ${_IREE_SELECTS_MSVC})
    list(APPEND _OPTS ${_IREE_SELECTS_MSVC_OR_CLANG_CL})
  else()
    message(ERROR "Unknown compiler: ${CMAKE_CXX_COMPILER}")
    list(APPEND _OPTS "")
  endif()
  set(${OPTS} ${_OPTS} PARENT_SCOPE)
endfunction()

#-------------------------------------------------------------------------------
# Data dependencies
#-------------------------------------------------------------------------------

# Adds 'data' dependencies to a target.
#
# Parameters:
# NAME: name of the target to add data dependencies to
# DATA: List of targets and/or files in the source tree. Files should use the
#       same format as targets (i.e. iree::package::subpackage::file.txt)
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

#-------------------------------------------------------------------------------
# Executable dependencies
#-------------------------------------------------------------------------------

# iree_add_executable_dependencies
#
# Adds dependency on a target in a cross-compilation-aware way. This should
# be used for depending on targets that are used as part of the build, such
# as for generating files used for later build steps.
#
# Parameters:
# EXECUTABLE: the executable to take on dependencies
# DEPENDENCY: additional dependencies to append to target
function(iree_add_executable_dependencies EXECUTABLE DEPENDENCY)
  if(CMAKE_CROSSCOMPILING)
    add_dependencies(${EXECUTABLE} iree_host_${DEPENDENCY})
  else()
    add_dependencies(${EXECUTABLE} ${DEPENDENCY})
  endif()
endfunction()
