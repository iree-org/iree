# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# iree_hal_executable()
#
# Compiles an MLIR source to a HAL executable binary using iree-compile in
# hal-executable mode. Unlike iree_bytecode_module() which produces full VM
# bytecode modules, this produces backend-specific device code for a single
# target.
#
# Parameters:
# NAME: Name of target.
# SRC: MLIR source file to compile.
# TARGET_DEVICE: Target device (e.g., "local", "vulkan", "hip", "cuda").
#     Generates --iree-hal-target-device=<value>.
# FLAGS: Additional compiler flags (list of strings). For devices with
#     sub-backends (like "local"), pass the backend selection flag here
#     (e.g., "--iree-hal-local-target-device-backends=vmvx").
# EXECUTABLE_FILE_NAME: Optional output filename. Defaults to ${NAME}.bin.
# COMPILE_TOOL: Compiler tool to invoke. Defaults to "iree-compile".
# C_IDENTIFIER: Identifier for generated C embed code. If omitted, no
#     C embed code is generated.
# FRIENDLY_NAME: Optional display name for build progress.
# PUBLIC: Export under ${PACKAGE}::.
# TESTONLY: Only build if IREE_BUILD_TESTS=ON.
# DEPENDS: Additional build dependencies beyond SRC and tools.
# DEPS: Library dependencies for the generated embed cc library.
function(iree_hal_executable)
  cmake_parse_arguments(
    _RULE
    "PUBLIC;TESTONLY"
    "NAME;SRC;TARGET_DEVICE;EXECUTABLE_FILE_NAME;COMPILE_TOOL;C_IDENTIFIER;FRIENDLY_NAME"
    "FLAGS;DEPENDS;DEPS"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()

  if(NOT DEFINED _RULE_TARGET_DEVICE)
    message(SEND_ERROR "iree_hal_executable requires TARGET_DEVICE")
  endif()

  # Set default for COMPILE_TOOL.
  if(DEFINED _RULE_COMPILE_TOOL)
    set(_COMPILE_TOOL ${_RULE_COMPILE_TOOL})
  else()
    set(_COMPILE_TOOL "iree-compile")
  endif()

  if(DEFINED _RULE_EXECUTABLE_FILE_NAME)
    set(_EXECUTABLE_FILE_NAME "${_RULE_EXECUTABLE_FILE_NAME}")
  else()
    set(_EXECUTABLE_FILE_NAME "${_RULE_NAME}.bin")
  endif()

  set(_ARGS
    "--compile-mode=hal-executable"
    "--iree-hal-target-device=${_RULE_TARGET_DEVICE}"
    "--mlir-print-op-on-diagnostic=false"
  )
  list(APPEND _ARGS "${_RULE_FLAGS}")

  get_filename_component(_SRC_PATH "${_RULE_SRC}" REALPATH)
  list(APPEND _ARGS "${_SRC_PATH}")
  list(APPEND _ARGS "-o")
  list(APPEND _ARGS "${_EXECUTABLE_FILE_NAME}")

  # Add the build directory to the compiler object file search path.
  list(APPEND _ARGS "--iree-hal-executable-object-search-path=\"${IREE_BINARY_DIR}\"")

  set(_OUTPUT_FILES "${_EXECUTABLE_FILE_NAME}")

  # If targeting a local device with llvm-cpu, pass linker paths.
  if (_RULE_FLAGS MATCHES "target-device-backends=llvm-cpu")
    if (IREE_LLD_BINARY)
      list(APPEND _ARGS "--iree-llvmcpu-embedded-linker-path=\"${IREE_LLD_BINARY}\"")
      list(APPEND _ARGS "--iree-llvmcpu-wasm-linker-path=\"${IREE_LLD_BINARY}\"")
    endif()
  endif()

  iree_compile_flags_for_platform(_PLATFORM_FLAGS "${_RULE_FLAGS}")
  if(_PLATFORM_FLAGS)
    list(APPEND _ARGS ${_PLATFORM_FLAGS})
  endif()

  if(_RULE_FRIENDLY_NAME)
    set(_FRIENDLY_NAME "${_RULE_FRIENDLY_NAME}")
  else()
    get_filename_component(_FRIENDLY_NAME "${_RULE_SRC}" NAME)
  endif()

  set(_DEPENDS "")
  iree_package_ns(_PACKAGE_NAME)
  list(TRANSFORM _RULE_DEPENDS REPLACE "^::" "${_PACKAGE_NAME}::")
  foreach(_DEPEND ${_RULE_DEPENDS})
    string(REPLACE "::" "_" _DEPEND "${_DEPEND}")
    list(APPEND _DEPENDS ${_DEPEND})
  endforeach()

  add_custom_command(
    OUTPUT
      ${_OUTPUT_FILES}
    COMMAND
      ${_COMPILE_TOOL}
      ${_ARGS}
    DEPENDS
      ${_COMPILE_TOOL}
      ${_RULE_SRC}
      ${_DEPENDS}
    COMMENT
      "Generating HAL executable ${_EXECUTABLE_FILE_NAME} from ${_FRIENDLY_NAME}"
    VERBATIM
  )

  iree_package_name(_PACKAGE_NAME)
  add_custom_target("${_PACKAGE_NAME}_${_RULE_NAME}"
    DEPENDS "${_EXECUTABLE_FILE_NAME}"
  )

  if(_RULE_TESTONLY)
    set(_TESTONLY_ARG "TESTONLY")
  endif()
  if(_RULE_PUBLIC)
    set(_PUBLIC_ARG "PUBLIC")
  endif()

  if(_RULE_C_IDENTIFIER)
    iree_c_embed_data(
      NAME
        "${_RULE_NAME}_c"
      IDENTIFIER
        "${_RULE_C_IDENTIFIER}"
      SRCS
        "${_EXECUTABLE_FILE_NAME}"
      C_FILE_OUTPUT
        "${_RULE_NAME}_c.c"
      H_FILE_OUTPUT
        "${_RULE_NAME}_c.h"
      FLATTEN
        "${_PUBLIC_ARG}"
        "${_TESTONLY_ARG}"
      DEPS
        ${_RULE_DEPS}
    )
  endif()
endfunction()

# iree_hal_executables()
#
# Batch form of iree_hal_executable(). Compiles multiple MLIR sources to HAL
# executables and bundles all outputs into a single iree_c_embed_data()
# cc_library target.
#
# TOC entry names are the source stems with a .bin extension (e.g.,
# "dispatch.mlir" produces TOC entry "dispatch.bin").
#
# Multiple calls in the same directory (for different formats) are safe:
# each call's compiled outputs use a {NAME}/ subdirectory prefix for the
# output filename, and FLATTEN on the embedded data strips it from TOC entries.
#
# Parameters:
# NAME: Name of the output iree_c_embed_data cc_library target.
# SRCS: MLIR source files to compile. Each produces one executable binary.
# TARGET_DEVICE: Target device for iree-compile.
# FLAGS: Backend-specific compiler flags.
# IDENTIFIER: C identifier for generated embed data. Defaults to NAME.
# COMPILE_TOOL: Compiler tool. Defaults to "iree-compile".
# PUBLIC: Export under ${PACKAGE}::.
# TESTONLY: Only build if IREE_BUILD_TESTS=ON.
function(iree_hal_executables)
  cmake_parse_arguments(
    _RULE
    "PUBLIC;TESTONLY"
    "NAME;TARGET_DEVICE;IDENTIFIER;COMPILE_TOOL"
    "SRCS;FLAGS"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()

  if(NOT DEFINED _RULE_TARGET_DEVICE)
    message(SEND_ERROR "iree_hal_executables requires TARGET_DEVICE")
  endif()

  if(DEFINED _RULE_IDENTIFIER)
    set(_IDENTIFIER "${_RULE_IDENTIFIER}")
  else()
    set(_IDENTIFIER "${_RULE_NAME}")
  endif()

  if(_RULE_TESTONLY)
    set(_TESTONLY_ARG "TESTONLY")
  endif()
  if(_RULE_PUBLIC)
    set(_PUBLIC_ARG "PUBLIC")
  endif()

  set(_COMPILE_TOOL_ARG "")
  if(DEFINED _RULE_COMPILE_TOOL)
    set(_COMPILE_TOOL_ARG "COMPILE_TOOL" "${_RULE_COMPILE_TOOL}")
  endif()

  # Compile each MLIR source to a HAL executable.
  set(_BIN_OUTPUTS "")
  foreach(_SRC ${_RULE_SRCS})
    # Derive stem from source path: "/path/to/foo.mlir" -> "foo".
    get_filename_component(_STEM "${_SRC}" NAME_WE)

    # Output files use a {NAME}/ subdirectory prefix for uniqueness across
    # multiple iree_hal_executables() calls in the same directory.
    set(_EXECUTABLE_FILE_NAME "${_RULE_NAME}/${_STEM}.bin")
    set(_RULE_ENTRY_NAME "${_RULE_NAME}_${_STEM}")

    iree_hal_executable(
      NAME "${_RULE_ENTRY_NAME}"
      SRC "${_SRC}"
      TARGET_DEVICE "${_RULE_TARGET_DEVICE}"
      FLAGS ${_RULE_FLAGS}
      EXECUTABLE_FILE_NAME "${_EXECUTABLE_FILE_NAME}"
      ${_COMPILE_TOOL_ARG}
      ${_TESTONLY_ARG}
    )

    list(APPEND _BIN_OUTPUTS "${_EXECUTABLE_FILE_NAME}")
  endforeach()

  # Bundle all compiled executables into a single embedded data library.
  iree_c_embed_data(
    NAME
      "${_RULE_NAME}"
    IDENTIFIER
      "${_IDENTIFIER}"
    SRCS
      ${_BIN_OUTPUTS}
    C_FILE_OUTPUT
      "${_RULE_NAME}.c"
    H_FILE_OUTPUT
      "${_RULE_NAME}.h"
    FLATTEN
      "${_PUBLIC_ARG}"
      "${_TESTONLY_ARG}"
  )
endfunction()
