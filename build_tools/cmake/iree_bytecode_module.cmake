# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# iree_bytecode_module()
#
# Builds an IREE bytecode module.
#
# Parameters:
# NAME: Name of target (see Note).
# SRC: Source file to compile into a bytecode module.
# FLAGS: Flags to pass to the compiler tool (list of strings).
#     `--output-format=vm-bytecode` is included automatically.
# MODULE_FILE_NAME: Optional. When specified, sets the output bytecode module
#    file name. When not specified, a default file name will be generated from
#    ${NAME}.
# COMPILE_TOOL: Compiler tool to invoke (CMake target). The default tool is
#     "iree-compile".
# C_IDENTIFIER: Identifier to use for generate c embed code.
#     If omitted then no C embed code will be generated.
# STATIC_LIB_PATH: When added, the module is compiled into a LLVM static
#     library with the specified library path.
# FRIENDLY_NAME: Optional. Name to use to display build progress info.
# PUBLIC: Add this so that this library will be exported under ${PACKAGE}::
#     Also in IDE, target will appear in ${PACKAGE} folder while non PUBLIC
#     will be in ${PACKAGE}/internal.
# TESTONLY: When added, this target will only be built if IREE_BUILD_TESTS=ON.
# DEPENDS: Optional. Additional dependencies beyond SRC and the tools.
# DEPS: Library dependencies to add to the generated embed cc library.
#
# Note:
# By default, iree_bytecode_module will create a module target named ${NAME} and
# a library named ${NAME}_c. The library has an alias target iree::${NAME}_c.
# The module doesn't have an alias due to a lack of support on custom target.
# The iree:: form should always be used to reduce namespace pollution.
function(iree_bytecode_module)
  cmake_parse_arguments(
    _RULE
    "PUBLIC;TESTONLY"
    "NAME;SRC;MODULE_FILE_NAME;COMPILE_TOOL;C_IDENTIFIER;FRIENDLY_NAME;STATIC_LIB_PATH"
    "FLAGS;DEPENDS;DEPS"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()

  if(_RULE_STATIC_LIB_PATH AND
     NOT (IREE_TARGET_BACKEND_LLVM_CPU OR IREE_HOST_BIN_DIR))
    message(SEND_ERROR "Static library only supports llvm-cpu backend")
  endif()

  # Set default for COMPILE_TOOL.
  if(DEFINED _RULE_COMPILE_TOOL)
    set(_COMPILE_TOOL ${_RULE_COMPILE_TOOL})
  else()
    set(_COMPILE_TOOL "iree-compile")
  endif()

  if(DEFINED _RULE_MODULE_FILE_NAME)
    set(_MODULE_FILE_NAME "${_RULE_MODULE_FILE_NAME}")
  else()
    set(_MODULE_FILE_NAME "${_RULE_NAME}.vmfb")
  endif()

  set(_ARGS
    "--output-format=vm-bytecode"
    "--mlir-print-op-on-diagnostic=false"
  )
  list(APPEND _ARGS "${_RULE_FLAGS}")

  get_filename_component(_SRC_PATH "${_RULE_SRC}" REALPATH)
  list(APPEND _ARGS "${_SRC_PATH}")
  list(APPEND _ARGS "-o")
  list(APPEND _ARGS "${_MODULE_FILE_NAME}")

  # Add the build directory to the compiler object file search path by default.
  # Users can add their own additional directories as needed.
  list(APPEND _ARGS "--iree-hal-executable-object-search-path=\"${IREE_BINARY_DIR}\"")

  set(_OUTPUT_FILES "${_MODULE_FILE_NAME}")

  # If the llvm-cpu backend is used, pass build-system-dependent flags.
  #
  # This crude check for target backend llvm-cpu is borrowed from
  # iree_compile_flags_for_platform(). This should be made more robust and
  # shared in a common helper.
  if (_RULE_FLAGS MATCHES "iree-hal-target-backends=llvm-cpu")
    if (IREE_LLD_BINARY)
      # Pass build-system-dependent linker paths.
      list(APPEND _ARGS "--iree-llvmcpu-embedded-linker-path=\"${IREE_LLD_BINARY}\"")
      list(APPEND _ARGS "--iree-llvmcpu-wasm-linker-path=\"${IREE_LLD_BINARY}\"")
    endif()
    # Note: --iree-llvmcpu-system-linker-path is left unspecified.

    # Check LLVM static library setting. If the static libary output path is set,
    # retrieve the object path and the corresponding header file path.
    if(_RULE_STATIC_LIB_PATH)
      list(APPEND _ARGS "--iree-llvmcpu-link-embedded=false")
      list(APPEND _ARGS "--iree-llvmcpu-link-static")
      list(APPEND _ARGS "--iree-llvmcpu-static-library-output-path=${_RULE_STATIC_LIB_PATH}")

      string(REPLACE ".o" ".h" _STATIC_HDR_PATH "${_RULE_STATIC_LIB_PATH}")
      list(APPEND _OUTPUT_FILES "${_RULE_STATIC_LIB_PATH}" "${_STATIC_HDR_PATH}")
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
      ${_LINKER_TOOL_EXECUTABLE}
      ${_RULE_SRC}
      ${_DEPENDS}
    COMMENT
      "Generating ${_MODULE_FILE_NAME} from ${_FRIENDLY_NAME}"
    VERBATIM
  )

  # Only add iree_${NAME} as custom target doesn't support aliasing to
  # iree::${NAME}.
  iree_package_name(_PACKAGE_NAME)
  add_custom_target("${_PACKAGE_NAME}_${_RULE_NAME}"
    DEPENDS "${_MODULE_FILE_NAME}"
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
        "${_RULE_NAME}.vmfb"
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
