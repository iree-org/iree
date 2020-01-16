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

# iree_pybind_cc_library()
#
# CMake function to imitate Bazel's pybind_cc_library rule.
#
# Parameters:
# NAME: name of target
# HDRS: List of public header files for the library
# SRCS: List of source files for the library
# DEPS: List of other libraries to be linked in to the binary targets
# COPTS: List of private compile options
# DEFINES: List of public defines
# INCLUDES: Include directories to add to dependencies
# LINKOPTS: List of link options
# TYPE: Type of library to be crated: either "MODULE", "SHARED" or "STATIC" (default).
# TESTONLY: When added, this target will only be built if user passes -DIREE_BUILD_TESTS=ON to CMake.

function(iree_pybind_cc_library)

  cmake_parse_arguments(
    _RULE
    "TESTONLY"
    "NAME"
    "HDRS;SRCS;COPTS;DEFINES;LINKOPTS;DEPS;INCLUDES;TYPE"
    ${ARGN}
  )

  if(NOT _RULE_TESTONLY OR IREE_BUILD_TESTS)
    # Prefix the library with the package name, so we get: iree_package_name.
    iree_package_name(_PACKAGE_NAME)
    set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

    if(NOT _RULE_TYPE)
      set(_RULE_TYPE "STATIC")
    endif()

    string(TOUPPER "${_RULE_TYPE}" _uppercase_RULE_TYPE)
    if(NOT _uppercase_RULE_TYPE MATCHES "^(STATIC|SHARED|MODULE)")
      message(FATAL_ERROR "Unsported library TYPE for iree_pybind_cc_library: ${_RULE_TYPE}")
    endif()

    add_library(${_NAME} ${_uppercase_RULE_TYPE} "")
    target_sources(${_NAME}
      PRIVATE
        ${_RULE_SRCS}
        ${_RULE_HDRS}
    )
    target_include_directories(${_NAME}
      PUBLIC
        "$<BUILD_INTERFACE:${IREE_COMMON_INCLUDE_DIRS}>"
        "$<BUILD_INTERFACE:${_RULE_INCLUDES}>"
      PRIVATE
        # TODO(marbre): Check if both pybind includes are necessary
        ${PYBIND11_INCLUDE_DIR}
        ${pybind11_INCLUDE_DIR} 
        ${PYTHON_INCLUDE_DIRS}
    )
    target_compile_options(${_NAME}
      PRIVATE
        ${_RULE_COPTS}
        ${PYBIND_COPTS}
        ${IREE_DEFAULT_COPTS}
    )

    target_link_libraries(${_NAME}
      PUBLIC
        ${_RULE_DEPS}
      PRIVATE
        pybind11
        # TODO(marbre): Add PYTHON_HEADERS_DEPS
        ${_RULE_LINKOPTS}
        ${IREE_DEFAULT_LINKOPTS}
    )
    target_compile_definitions(${_NAME}
      PUBLIC
        ${_RULE_DEFINES}
    )

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

    if(NOT _uppercase_RULE_TYPE MATCHES "STATIC")
      set_property(TARGET ${_NAME} PROPERTY PREFIX "${PYTHON_MODULE_PREFIX}")
      set_property(TARGET ${_NAME} PROPERTY SUFFIX "${PYTHON_MODULE_EXTENSION}")
    endif()

    # Alias the iree_package_name library to iree::package::name.
    # This lets us more clearly map to Bazel and makes it possible to
    # disambiguate the underscores in paths vs. the separators.
    iree_package_ns(_PACKAGE_NS)
    add_library(${_PACKAGE_NS}::${_RULE_NAME} ALIAS ${_NAME})
    iree_package_dir(_PACKAGE_DIR)
    if(${_RULE_NAME} STREQUAL ${_PACKAGE_DIR})
      # If the library name matches the package then treat it as a default.
      # For example, foo/bar/ library 'bar' would end up as 'foo::bar'.
      add_library(${_PACKAGE_NS} ALIAS ${_NAME})
    endif()
  endif()
endfunction()
