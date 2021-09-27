# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

function(iree_compiler_api_py_test)
  cmake_parse_arguments(
    ARG
    ""
    "NAME;MAIN"
    ""
    ${ARGN}
  )
  set(TEST_NAME "iree-compiler-api-${ARG_NAME}")
  add_test(
    NAME
      ${TEST_NAME}
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_MAIN}"
  )
  set_tests_properties(${TEST_NAME} PROPERTIES
    ENVIRONMENT PYTHONPATH=${IREE_COMPILER_API_BINARY_DIR}/python_package)
endfunction()

iree_compiler_api_py_test(
  NAME
    compiler_core_test
  MAIN
    "compiler_core_test.py"
)

iree_compiler_api_py_test(
  NAME
    compiler_tf_test
  MAIN
    "compiler_tf_test.py"
)

iree_compiler_api_py_test(
  NAME
    compiler_tflite_test
  MAIN
    "compiler_tflite_test.py"
)

iree_compiler_api_py_test(
  NAME
    compiler_xla_test
  MAIN
    "compiler_xla_test.py"
)
