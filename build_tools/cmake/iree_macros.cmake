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
else()
  set(IREE_HOST_SCRIPT_EXT "sh")
endif()

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
