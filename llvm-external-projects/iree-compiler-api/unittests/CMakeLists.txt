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

add_subdirectory(CAPI)
add_subdirectory(tools)
add_subdirectory(transforms/ireec)
