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

###############################################################################
# Main user rules
###############################################################################

# iree_pyext_module()
#
# Builds a native python module (.so/.dylib/.pyd).
#
# Parameters:
# NAME: name of target
# MODULE_NAME: Base-name of the module.
# SRCS: List of source files for the library
# DEPS: List of other targets the test python libraries require
function(iree_pyext_module)
  cmake_parse_arguments(ARG
    ""
    "NAME;MODULE_NAME;UNIX_LINKER_SCRIPT"
    "SRCS;DEPS;COPTS;INCLUDES"
    ${ARGN})

  iree_package_ns(_PACKAGE_NS)
  # Replace dependencies passed by ::name with ::iree::package::name
  list(TRANSFORM ARG_DEPS REPLACE "^::" "${_PACKAGE_NS}::")
  list(TRANSFORM ARG_PYEXT_DEPS REPLACE "^::" "${_PACKAGE_NS}::")
  # Prefix the library with the package name, so we get: iree_package_name.
  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${ARG_NAME}")

  pybind11_add_module(
    ${_NAME}
    ${ARG_SRCS}
  )

  target_link_libraries(
    ${_NAME}
    PRIVATE ${ARG_DEPS}
  )

  set_target_properties(
    ${_NAME} PROPERTIES
      OUTPUT_NAME "${ARG_MODULE_NAME}"
  )

  target_include_directories(${_NAME}
  PUBLIC
    "$<BUILD_INTERFACE:${ARG_INCLUDES}>"
  )

  target_compile_options(
    ${_NAME} PRIVATE
    ${ARG_COPTS}
    ${IREE_DEFAULT_COPTS}
  )


  # Link flags.
  if(UNIX AND NOT APPLE)  # Apple does not support linker scripts.
    if(ARG_UNIX_LINKER_SCRIPT)
      set_target_properties(${_NAME} PROPERTIES LINK_FLAGS
        "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/${ARG_UNIX_LINKER_SCRIPT}")
    endif()
  endif()
endfunction()

# iree_py_library()
#
# CMake function to imitate Bazel's iree_py_library rule.
#
# Parameters:
# NAME: name of target
# SRCS: List of source files for the library
# DEPS: List of other targets the test python libraries require
# PYEXT_DEPS: List of deps of extensions built with iree_pyext_module
function(iree_py_library)
  cmake_parse_arguments(
    ARG
    ""
    "NAME"
    "SRCS;DEPS;PYEXT_DEPS"
    ${ARGN}
  )

  iree_package_ns(_PACKAGE_NS)
  # Replace dependencies passed by ::name with ::iree::package::name
  list(TRANSFORM ARG_DEPS REPLACE "^::" "${_PACKAGE_NS}::")

  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${ARG_NAME}")

  add_custom_target(${_NAME} ALL
    DEPENDS ${ARG_DEPS}
  )

  # Symlink each file as its own target.
  foreach(SRC_FILE ${ARG_SRCS})
    add_custom_command(
      TARGET ${_NAME}
      COMMAND ${CMAKE_COMMAND} -E create_symlink
        "${CMAKE_CURRENT_SOURCE_DIR}/${SRC_FILE}" "${CMAKE_CURRENT_BINARY_DIR}/${SRC_FILE}"
      BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/${SRC_FILE}"
    )
  endforeach()

  # Add PYEXT_DEPS.
  if(${ARG_PYEXT_DEPS})
    foreach(V ${IREE_MULTIPY_VERSIONS_EFFECTIVE})
      add_dependencies(${_NAME} ${ARG_PYEXT_DEPS})
    endforeach()
  endif()
endfunction()

# iree_py_test()
#
# CMake function to imitate Bazel's iree_py_test rule.
#
# Parameters:
# NAME: name of test
# SRCS: List of source file
# DEPS: List of deps the test requires
# LABELS: Additional labels to apply to the test. The package path is added
#     automatically.
function(iree_py_test)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "SRCS;DEPS;LABELS"
    ${ARGN}
  )

  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  iree_package_ns(_PACKAGE_NS)
  string(REPLACE "::" "/" _PACKAGE_PATH ${_PACKAGE_NS})
  set(_NAME_PATH "${_PACKAGE_PATH}/${_RULE_NAME}")
  list(APPEND _RULE_LABELS "${_PACKAGE_PATH}")

  add_test(
    NAME ${_NAME}
    COMMAND
      "${CMAKE_SOURCE_DIR}/build_tools/cmake/run_test.${IREE_HOST_SCRIPT_EXT}"
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/${_RULE_SRCS}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  set_property(TEST ${_NAME} PROPERTY LABELS "${_RULE_LABELS}")
  set_property(TEST ${_NAME} PROPERTY ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/bindings/python:$ENV{PYTHONPATH};TEST_TMPDIR=${_NAME}_${V}_test_tmpdir")
  # TODO(marbre): Find out how to add deps to tests.
  #               Similar to _RULE_DATA in iree_lit_test().
endfunction()
