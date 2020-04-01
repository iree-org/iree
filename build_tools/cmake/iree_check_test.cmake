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

# iree_check_test()
#
# Creates a test using iree-check-module for the specified source file.
#
# Mirrors the bzl rule of the same name.
#
# Parameters:
#   NAME: Name of the target
#   SRC: mlir source file to be compiled to an IREE module.
#   TARGET_BACKEND: target backend to compile for.
#   DRIVER: driver to run the module with.
#   ARGS: additional args to pass to iree-check-module. The driver and input
#       file are passed automatically.
#   LABELS: Additional labels to apply to the test. The package path and
#       "driver=${DRIVER}" are added automatically.
function(iree_check_test)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;SRC;TARGET_BACKEND;DRIVER;LABELS"
    "ARGS"
    ${ARGN}
  )
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  set(_MODULE_TARGET_NAME "${_RULE_NAME}_module")

  iree_bytecode_module(
    NAME
      "${_MODULE_TARGET_NAME}"
    SRC
      "${_RULE_SRC}"
    FLAGS
      "-iree-mlir-to-vm-bytecode-module"
      "--iree-hal-target-backends=${_RULE_TARGET_BACKEND}"
    TESTONLY
  )

  # TODO(b/146898896): It would be nice if this were something we could query
  # rather than having to know the conventions used by iree_bytecode_module.
  set(_MODULE_FILE_NAME "${_MODULE_TARGET_NAME}.module")

  # iree_bytecode_module does not define a target, only a custom command.
  # We need to create a target that depends on the command to ensure the
  # module gets built.
  # TODO(b/146898896): Do this in iree_bytecode_module and avoid having to
  # reach into the internals.
  add_custom_target(
    "${_MODULE_TARGET_NAME}"
     DEPENDS
       "${_MODULE_FILE_NAME}"
  )

  # A target specifically for the test. We could combine this with the above,
  # but we want that one to get pulled into iree_bytecode_module.
  add_custom_target("${_NAME}" ALL)
  add_dependencies(
    "${_NAME}"
    "${_MODULE_TARGET_NAME}"
    iree_modules_check_iree-check-module
  )

  string(REPLACE "_" "/" _PACKAGE_PATH ${_PACKAGE_NAME})
  set(_NAME_PATH "${_PACKAGE_PATH}:${_RULE_NAME}")

  add_test(
    NAME
      "${_NAME_PATH}"
    COMMAND
      "${CMAKE_SOURCE_DIR}/build_tools/cmake/run_test.${IREE_HOST_SCRIPT_EXT}"
      "$<TARGET_FILE:iree_modules_check_iree-check-module>"
      "--driver=${_RULE_DRIVER}"
      "${CMAKE_CURRENT_BINARY_DIR}/${_MODULE_FILE_NAME}"
      ${_RULE_ARGS}
  )

  list(APPEND _RULE_LABELS "${_PACKAGE_PATH}" "driver=${_RULE_DRIVER}")
  set_property(TEST "${_NAME_PATH}" PROPERTY REQUIRED_FILES "${_MODULE_FILE_NAME}")
  set_property(TEST "${_NAME_PATH}" PROPERTY ENVIRONMENT "TEST_TMPDIR=${_NAME}_test_tmpdir")
  set_property(TEST "${_NAME_PATH}" PROPERTY LABELS "${_RULE_LABELS}")
endfunction()


# iree_check_test_suite()
#
# Creates a test suite of iree-check-module tests.
#
# Mirrors the bzl rule of the same name.
#
# One test is generated per source and backend/driver pair.
# Parameters:
#   NAME: name of the generated test suite.
#   SRCS: source mlir files containing the module.
#   TARGET_BACKENDS: backends to compile the module for. These form pairs with
#       the DRIVERS argument (due to cmake limitations they are separate list
#       arguments). The lengths must exactly match. If no backends or drivers are
#       specified, a test will be generated for every supported pair.
#   DRIVERS: drivers to run the module with. These form pairs with the
#       TARGET_BACKENDS argument (due to cmake limitations they are separate list
#       arguments). The lengths must exactly match. If no backends or drivers are
#       specified, a test will be generated for every supported pair.
#   ARGS: additional args to pass to the underlying iree-check-module tests. The
#       driver and input file are passed automatically. To use different args per
#       test, create a separate suite or iree_check_test.
#   LABELS: Additional labels to apply to the generated tests. The package path is
#       added automatically.
function(iree_check_test_suite)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "SRCS;TARGET_BACKENDS;DRIVERS;ARGS;LABELS"
    ${ARGN}
  )
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  if(NOT DEFINED _RULE_TARGET_BACKENDS AND NOT DEFINED _RULE_DRIVERS)
    set(_RULE_TARGET_BACKENDS "vmla" "vulkan-spirv")
    set(_RULE_DRIVERS "vmla" "vulkan")
  endif()

  list(LENGTH _RULE_TARGET_BACKENDS _TARGET_BACKEND_COUNT)
  list(LENGTH _RULE_DRIVERS _DRIVER_COUNT)

  if(NOT _TARGET_BACKEND_COUNT EQUAL _DRIVER_COUNT)
    message(SEND_ERROR
        "TARGET_BACKENDS count ${_TARGET_BACKEND_COUNT} does not match DRIVERS count ${_DRIVER_COUNT}")
  endif()

  math(EXPR _MAX_INDEX "${_TARGET_BACKEND_COUNT} - 1")
  foreach(_INDEX RANGE "${_MAX_INDEX}")
    list(GET _RULE_TARGET_BACKENDS ${_INDEX} _TARGET_BACKEND)
    list(GET _RULE_DRIVERS ${_INDEX} _DRIVER)
    foreach(_SRC IN LISTS _RULE_SRCS)
      set(_TEST_NAME "${_RULE_NAME}_${_SRC}_${_TARGET_BACKEND}_${_DRIVER}")
      iree_check_test(
        NAME
	  ${_TEST_NAME}
	SRC
	  ${_SRC}
	TARGET_BACKEND
	  ${_TARGET_BACKEND}
	DRIVER
	  ${_DRIVER}
	ARGS
	  ${_RULE_ARGS}
	LABELS
	  ${_RULE_LABELS}
      )
    endforeach()
  endforeach()
endfunction()
