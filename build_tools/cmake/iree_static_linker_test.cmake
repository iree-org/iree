# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

# iree_static_linker_test()
#
# Creates a test with statically linked libraries. Assuming the target backend
# is llvm-cpu and the driver is the local-sync driver.
#
# Parameters:
#   NAME: Name of the target
#   SRC: mlir source file to be compiled to an IREE module.
#   INPUT_SHAPE: Module input shape, assuming "n, h, w, c" format.
#   INPUT_TYPE: Module input type, assuming native c/c++ data types
#   HAL_TYPE: HAL data type, see runtime/src/iree/hal/buffer_view.h for possible
#       options.
#   STATIC_LIB_PREFIX: llvm static library prefix.
#   MAIN_FUNCTION: Entry function call.
#   EMITC: Optional, use emitC instead of vmfb module to define the vm contrl.
#   COMPILER_FLAGS: additional flags to pass to the compiler. Bytecode output
#       format and backend flags are passed automatically.
#   LABELS: Additional labels to apply to the test. The package path and
#       "driver=local-sync" are added automatically.
#   TARGET_CPU_FEATURES: If specified, a string passed as argument to
#       --iree-llvm-target-cpu-features.
#
# Example:
#   iree_static_linker_test(
#     NAME
#       edge_detection_test
#     SRC
#       "edge_detection.mlir"
#     STATIC_LIB_PREFIX
#       edge_detection_linked_llvm_cpu
#     MAIN_FUNCTION
#       "module.edge_detect_sobel_operator"
#     INPUT_SHAPE
#       "1, 128, 128, 1"
#     INPUT_TYPE
#       float
#     HAL_TYPE
#       IREE_HAL_ELEMENT_TYPE_FLOAT_32
#     COMPILER_FLAGS
#       "--iree-input-type=mhlo"
#   )
function(iree_static_linker_test)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  # See comment in iree_check_test about this condition.
  if(NOT IREE_BUILD_COMPILER AND NOT CMAKE_CROSSCOMPILING)
    return()
  endif()

  if(NOT IREE_TARGET_BACKEND_LLVM_CPU OR NOT IREE_HAL_DRIVER_LOCAL_SYNC)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    "EMITC"
    "NAME;SRC;DRIVER;STATIC_LIB_PREFIX;MAIN_FUNCTION;INPUT_TYPE;HAL_TYPE"
    "COMPILER_FLAGS;LABELS;TARGET_CPU_FEATURES;INPUT_SHAPE"
    ${ARGN}
  )

  iree_get_executable_path(_COMPILER_TOOL "iree-compile")
  get_filename_component(_SRC_PATH "${_RULE_SRC}" REALPATH)
  iree_package_name(_PACKAGE_NAME)
  iree_package_ns(_PACKAGE_NS)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  # Set up static library name.
  set(_O_FILE_NAME "${_RULE_NAME}.o")
  set(_H_FILE_NAME "${_RULE_NAME}.h")

  # Set common iree-compile flags
  set(_COMPILER_ARGS ${_RULE_COMPILER_FLAGS})
  list(APPEND _COMPILER_ARGS "--iree-hal-target-backends=llvm-cpu")
  list(APPEND _COMPILER_ARGS  "--iree-llvm-link-embedded=false")
  list(APPEND _COMPILER_ARGS  "--iree-llvm-link-static")
  list(APPEND _COMPILER_ARGS "-iree-llvm-static-library-output-path=${_O_FILE_NAME}")
  if(_RULE_TARGET_CPU_FEATURES)
    list(APPEND _COMPILER_ARGS "-iree-llvm-target-cpu-features=${_RULE_TARGET_CPU_FEATURES}")
  endif()
  list(APPEND _COMPILER_ARGS "${_SRC_PATH}")

  if(_RULE_EMITC)
    set(_C_FILE_NAME "${_RULE_NAME}_emitc.h")
    list(APPEND _COMPILER_ARGS "-iree-mlir-to-vm-c-module")
    list(APPEND _COMPILER_ARGS "-o")
    list(APPEND _COMPILER_ARGS "${_C_FILE_NAME}")

    # Custom command for iree-compile to generate static library and C module.
    add_custom_command(
      OUTPUT
        ${_H_FILE_NAME}
        ${_O_FILE_NAME}
        ${_C_FILE_NAME}
      COMMAND ${_COMPILER_TOOL} ${_COMPILER_ARGS}
      DEPENDS ${_COMPILER_TOOL} ${_RULE_SRC}
    )

    set(_EMITC_LIB_NAME "${_NAME}_emitc")
    add_library(${_EMITC_LIB_NAME}
      STATIC
      ${_C_FILE_NAME}
    )
    target_compile_definitions(${_EMITC_LIB_NAME} PUBLIC EMITC_IMPLEMENTATION)
    SET_TARGET_PROPERTIES(
      ${_EMITC_LIB_NAME}
      PROPERTIES
        LINKER_LANGUAGE C
    )
  else()  # bytecode module path
    set(_VMFB_FILE_NAME ${_RULE_NAME}.vmfb)
    list(APPEND _COMPILER_ARGS "-o")
    list(APPEND _COMPILER_ARGS "${_VMFB_FILE_NAME}")

    # Custom command for iree-compile to generate static library and VM module.
    add_custom_command(
      OUTPUT
        ${_H_FILE_NAME}
        ${_O_FILE_NAME}
        ${_VMFB_FILE_NAME}
      COMMAND ${_COMPILER_TOOL} ${_COMPILER_ARGS}
      DEPENDS ${_COMPILER_TOOL} ${_RULE_SRC}
    )

    # Generate the embed data with the bytecode module.
    set(_MODULE_C_NAME "${_RULE_NAME}_module_c")
    set(_EMBED_H_FILE_NAME ${_MODULE_C_NAME}.h)
    set(_EMBED_C_FILE_NAME ${_MODULE_C_NAME}.c)
    iree_c_embed_data(
      NAME
        "${_MODULE_C_NAME}"
      IDENTIFIER
        "${_NAME}"
      GENERATED_SRCS
        "${_VMFB_FILE_NAME}"
      C_FILE_OUTPUT
        "${_EMBED_C_FILE_NAME}"
      H_FILE_OUTPUT
        "${_EMBED_H_FILE_NAME}"
      FLATTEN
      PUBLIC
    )
  endif(_RULE_EMITC)

  set(_LIB_NAME "${_NAME}_lib")
  add_library(${_LIB_NAME}
    STATIC
    ${_O_FILE_NAME}
  )
  SET_TARGET_PROPERTIES(
    ${_LIB_NAME}
    PROPERTIES
    LINKER_LANGUAGE C
  )
  # Set alias for this static library to be used later.
  add_library(${_PACKAGE_NS}::${_RULE_NAME}_lib ALIAS ${_LIB_NAME})

  if(NOT _RULE_INPUT_SHAPE OR
     NOT _RULE_INPUT_TYPE OR
     NOT _RULE_HAL_TYPE OR
     NOT _RULE_MAIN_FUNCTION OR
     NOT _RULE_STATIC_LIB_PREFIX)
    return()
  endif()

  # Process module input configs
  list(LENGTH _RULE_INPUT_SHAPE _INPUT_NUM)

  set(_INPUT_DIM_LIST)
  set(_INPUT_SIZE_LIST)
  set(_INPUT_SHAPE_STR "{")
  set(_INPUT_DIM_MAX 0)
  foreach(_INPUT_SHAPE_ENTRY ${_RULE_INPUT_SHAPE})
    set(_INPUT_SHAPE_STR "${_INPUT_SHAPE_STR}\{${_INPUT_SHAPE_ENTRY}\}, ")

    string(REPLACE "," ";" _INPUT_SHAPE_LIST "${_INPUT_SHAPE_ENTRY}")
    list(LENGTH _INPUT_SHAPE_LIST _INPUT_DIM)
    list(APPEND _INPUT_DIM_LIST ${_INPUT_DIM})
    if(_INPUT_DIM GREATER _INPUT_DIM_MAX)
      set(_INPUT_DIM_MAX ${_INPUT_DIM})
    endif()
    set(_INPUT_SIZE 1)
    foreach(_INPUT_DIM_VAL ${_INPUT_SHAPE_LIST})
      math(EXPR _INPUT_SIZE "${_INPUT_SIZE} * ${_INPUT_DIM_VAL}")
    endforeach()
    list(APPEND _INPUT_SIZE_LIST ${_INPUT_SIZE})
  endforeach()
  set(_INPUT_SHAPE_STR "${_INPUT_SHAPE_STR}}")
  string(REPLACE ";" ", " _INPUT_DIM_STR "${_INPUT_DIM_LIST}")
  string(REPLACE ";" ", " _INPUT_SIZE_STR "${_INPUT_SIZE_LIST}")

  # Generate the source file.
  # TODO(scotttodd): Move to build time instead of configure time?
  set(IREE_STATIC_LIB_HDR "${_H_FILE_NAME}")
  set(IREE_STATIC_LIB_QUERY_FN "${_RULE_STATIC_LIB_PREFIX}_library_query")
  set(IREE_MODULE_HDR "${_EMBED_H_FILE_NAME}")
  set(IREE_MODULE_CREATE_FN "${_NAME}_create")
  set(IREE_EMITC_HDR "${_C_FILE_NAME}")
  set(IREE_MODULE_MAIN_FN "${_RULE_MAIN_FUNCTION}")
  set(IREE_INPUT_NUM "${_INPUT_NUM}")
  set(IREE_INPUT_DIM_MAX "${_INPUT_DIM_MAX}")
  set(IREE_INPUT_DIM_ARR "${_INPUT_DIM_STR}")
  set(IREE_INPUT_TYPE "${_RULE_INPUT_TYPE}")
  set(IREE_INPUT_SIZE_ARR "${_INPUT_SIZE_STR}")
  set(IREE_INPUT_SHAPE_ARR "${_INPUT_SHAPE_STR}")
  set(IREE_HAL_TYPE "${_RULE_HAL_TYPE}")
  set(IREE_EXE_NAME ${_RULE_NAME})
  configure_file(
    "${IREE_ROOT_DIR}/build_tools/cmake/static_linker_test.c.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}.c"
  )

  iree_cc_binary(
    NAME
      ${_RULE_NAME}_run
    SRCS
      "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}.c"
    DEPS
      ::${_RULE_NAME}_lib
      iree::runtime
      iree::hal::drivers::local_sync::sync_driver
      iree::hal::local::loaders::static_library_loader
  )

  if(_RULE_EMITC)
    target_link_libraries(${_NAME}_run
        PRIVATE
          iree_vm_shims_emitc
          ${_EMITC_LIB_NAME}
    )
  else()
    target_link_libraries(${_NAME}_run PRIVATE ${_NAME}_module_c)
  endif()

  add_dependencies(iree-test-deps "${_NAME}_run")

  iree_native_test(
    NAME
      ${_RULE_NAME}
    SRC
      ::${_RULE_NAME}_run
    DRIVER
      local-sync
    LABELS
      ${_RULE_LABELS}
      ${_RULE_TARGET_CPU_FEATURES}
  )
endfunction()
