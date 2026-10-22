# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

# Generates a Hexagon FastRPC interface from an IDL file with the Hexagon SDK
# qaic IDL compiler. Requires build_tools/cmake/iree_hexagon_sdk.cmake.
#
# CMake equivalent of build_tools/bazel/iree_hexagon_interface.bzl.
#
# Parameters:
# NAME: Name of the interface. Creates a library ${NAME} with the generated
#       header and the client stub for Android aarch64 (the host calling into
#       the DSP). The generated DSP skeleton is consumed by
#       iree_hexagon_dsp_library(INTERFACE ...). The IDL file and the interface
#       defined in it must be named ${NAME}.
# IDL: The *.idl interface definition.
function(iree_hexagon_interface)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;IDL"
    ""
    ${ARGN}
  )

  iree_package_name(_PACKAGE_NAME)

  get_filename_component(_IDL "${_RULE_IDL}" ABSOLUTE)
  set(_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}")
  set(_HEADER "${_OUT_DIR}/${_RULE_NAME}.h")
  set(_STUB "${_OUT_DIR}/${_RULE_NAME}_stub.c")
  set(_SKEL "${_OUT_DIR}/${_RULE_NAME}_skel.c")

  file(MAKE_DIRECTORY "${_OUT_DIR}")
  add_custom_command(
    OUTPUT
      "${_HEADER}"
      "${_STUB}"
      "${_SKEL}"
    COMMAND
      hexagon_sdk::qaic
      -mdll
      -o "${_OUT_DIR}"
      "-I${HEXAGON_SDK_ROOT}/incs"
      "-I${HEXAGON_SDK_ROOT}/incs/stddef"
      "${_IDL}"
    DEPENDS
      "${_IDL}"
      hexagon_sdk::qaic
    COMMENT "Generating Hexagon RPC interface from ${_RULE_IDL}"
    VERBATIM
  )
  set(_GEN_TARGET "${_PACKAGE_NAME}_${_RULE_NAME}_gen")
  add_custom_target(${_GEN_TARGET}
    DEPENDS
      "${_HEADER}"
      "${_STUB}"
      "${_SKEL}"
  )

  iree_cc_library(
    NAME
      ${_RULE_NAME}
    HDRS
      "${_HEADER}"
    SRCS
      "${_STUB}"
    COPTS
      # The generated code is not warning-free.
      "-w"
    DEPS
      hexagon_sdk::cdsprpc_android_aarch64
    INCLUDES
      "$<BUILD_INTERFACE:${_OUT_DIR}>"
    PUBLIC
  )
  add_dependencies(${_PACKAGE_NAME}_${_RULE_NAME} ${_GEN_TARGET})

  # Used by iree_hexagon_dsp_library().
  set_target_properties(${_PACKAGE_NAME}_${_RULE_NAME} PROPERTIES
    IREE_HEXAGON_INTERFACE_DIR "${_OUT_DIR}"
    IREE_HEXAGON_INTERFACE_HEADER "${_HEADER}"
    IREE_HEXAGON_INTERFACE_SKEL "${_SKEL}"
  )
endfunction()
