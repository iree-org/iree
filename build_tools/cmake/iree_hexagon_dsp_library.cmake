# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

# Builds a FastRPC skeleton shared library (lib${NAME}.so) for the Hexagon DSP
# by invoking the Hexagon SDK compiler directly. Requires
# build_tools/cmake/iree_hexagon_sdk.cmake.
#
# CMake equivalent of build_tools/bazel/iree_hexagon_dsp_library.bzl.
#
# Parameters:
# NAME: Name of the target; the output is lib${NAME}.so.
# INTERFACE: Target created by iree_hexagon_interface() whose skeleton is
#            linked in.
# ARCH: Hexagon architecture version (e.g. v79).
# SRCS: C sources and headers implementing the interface.
# COPTS: Additional flags passed to hexagon-clang when compiling.
# LINKOPTS: Additional flags passed to hexagon-clang when linking.
function(iree_hexagon_dsp_library)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;INTERFACE;ARCH"
    "SRCS;COPTS;LINKOPTS"
    ${ARGN}
  )

  iree_package_name(_PACKAGE_NAME)

  if(NOT _RULE_ARCH)
    set(_RULE_ARCH "v79")
  endif()

  # The interface target may be defined in a directory processed after this
  # one, so its properties are read with generator expressions.
  set(_INTERFACE_DIR
    "$<TARGET_PROPERTY:${_RULE_INTERFACE},IREE_HEXAGON_INTERFACE_DIR>")
  set(_INTERFACE_HEADER
    "$<TARGET_PROPERTY:${_RULE_INTERFACE},IREE_HEXAGON_INTERFACE_HEADER>")
  set(_INTERFACE_SKEL
    "$<TARGET_PROPERTY:${_RULE_INTERFACE},IREE_HEXAGON_INTERFACE_SKEL>")

  set(_CLANG "${IREE_HEXAGON_TOOLS_DIR}/bin/hexagon-clang")
  set(_COPTS
    "-m${_RULE_ARCH}"
    "-G0"
    "-fpic"
    "-O2"
    "-Wall"
    # FastRPC loads the skeleton via dlopen and looks up
    # <interface>_skel_handle_invoke, so symbols have to be exported.
    "-fvisibility=default"
    "-I${_INTERFACE_DIR}"
    "-I${HEXAGON_SDK_ROOT}/incs"
    "-I${HEXAGON_SDK_ROOT}/incs/stddef"
    "-I${HEXAGON_SDK_ROOT}/rtos/qurt/compute${_RULE_ARCH}/include/qurt"
    "-I${HEXAGON_SDK_ROOT}/rtos/qurt/compute${_RULE_ARCH}/include/posix"
    ${_RULE_COPTS}
  )
  set(_LINKOPTS
    "-m${_RULE_ARCH}"
    "-G0"
    "-fpic"
    "-shared"
    "-Wl,-Bsymbolic"
    ${_RULE_LINKOPTS}
  )

  # Headers invalidate all objects; .c files are compiled. The generated
  # interface skeleton is compiled as <name>.objects/interface_skel.o.
  set(_HDRS "${_INTERFACE_HEADER}")
  set(_C_SRCS "${_INTERFACE_SKEL}")
  set(_C_SRC_NAMES "interface_skel")
  foreach(_SRC ${_RULE_SRCS})
    get_filename_component(_SRC "${_SRC}" ABSOLUTE)
    if(_SRC MATCHES "\\.c$")
      get_filename_component(_SRC_NAME "${_SRC}" NAME_WE)
      list(APPEND _C_SRCS "${_SRC}")
      list(APPEND _C_SRC_NAMES "${_SRC_NAME}")
    else()
      list(APPEND _HDRS "${_SRC}")
    endif()
  endforeach()

  set(_OBJECT_DIR "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}.objects")
  set(_OBJECTS)
  foreach(_SRC _SRC_NAME IN ZIP_LISTS _C_SRCS _C_SRC_NAMES)
    set(_OBJECT "${_OBJECT_DIR}/${_SRC_NAME}.o")
    add_custom_command(
      OUTPUT
        "${_OBJECT}"
      COMMAND
        "${CMAKE_COMMAND}" -E make_directory "${_OBJECT_DIR}"
      COMMAND
        "${_CLANG}" ${_COPTS} -c "${_SRC}" -o "${_OBJECT}"
      DEPENDS
        "${_SRC}"
        ${_HDRS}
      COMMENT "Compiling ${_SRC_NAME} for the Hexagon DSP"
      VERBATIM
    )
    list(APPEND _OBJECTS "${_OBJECT}")
  endforeach()

  set(_OUT "${CMAKE_CURRENT_BINARY_DIR}/lib${_RULE_NAME}.so")
  add_custom_command(
    OUTPUT
      "${_OUT}"
    COMMAND
      "${_CLANG}" ${_LINKOPTS} ${_OBJECTS} -o "${_OUT}"
    DEPENDS
      ${_OBJECTS}
    COMMENT "Linking Hexagon DSP library lib${_RULE_NAME}.so"
    VERBATIM
  )
  add_custom_target(${_PACKAGE_NAME}_${_RULE_NAME} ALL
    DEPENDS
      "${_OUT}"
  )
  # Name of the target generating the interface, see iree_hexagon_interface().
  add_dependencies(${_PACKAGE_NAME}_${_RULE_NAME} ${_RULE_INTERFACE}_gen)
endfunction()
