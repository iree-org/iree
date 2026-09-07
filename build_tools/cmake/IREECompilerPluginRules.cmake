# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Builds a dynamic compiler plugin from an IREE install tree.
#
# Separate from the in-tree rule, which subtracts the compiler's archives from
# the plugin's link closure. Out of tree none of the plugin's archives are in
# the compiler.

find_package(Python3 REQUIRED COMPONENTS Interpreter)

# LLVM's tools, not binutils: the rename works on demangled text, and the two
# demanglers disagree on some symbols, which would then not resolve at load.
#
# Looked up on first use, after the caller has found LLVM, so they come from
# the compiler's LLVM. Set them to override.
macro(_iree_find_rename_tools)
  find_program(IREE_LLVM_NM
    NAMES llvm-nm HINTS "${LLVM_TOOLS_BINARY_DIR}" REQUIRED)
  find_program(IREE_LLVM_CXXFILT
    NAMES llvm-cxxfilt HINTS "${LLVM_TOOLS_BINARY_DIR}" REQUIRED)
  find_program(IREE_LLVM_OBJCOPY
    NAMES llvm-objcopy HINTS "${LLVM_TOOLS_BINARY_DIR}" REQUIRED)
endmacro()

# Parameters:
# PLUGIN_ID: Id the plugin reports and --iree-plugin= activates.
# TARGET: Static library carrying the registration.
# EXTRA_ARCHIVES: Further static libraries the plugin must carry.
function(iree_compiler_register_dynamic_plugin)
  cmake_parse_arguments(_RULE "" "PLUGIN_ID;TARGET" "EXTRA_ARCHIVES" ${ARGN})

  if(NOT _RULE_PLUGIN_ID OR NOT _RULE_TARGET)
    message(FATAL_ERROR "PLUGIN_ID and TARGET are required")
  endif()

  _iree_find_rename_tools()

  set(_name "iree_compiler_plugin_${_RULE_PLUGIN_ID}")
  set(_dir "${CMAKE_CURRENT_BINARY_DIR}/${_name}.renamed")
  set(_index 0)
  set(_renamed)
  foreach(_input "$<TARGET_FILE:${_RULE_TARGET}>" ${_RULE_EXTRA_ARCHIVES})
    set(_map "${_dir}/${_index}.rename_map")
    set(_out "${_dir}/${_index}.renamed.a")
    add_custom_command(
      OUTPUT "${_map}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${_dir}"
      COMMAND
        "${Python3_EXECUTABLE}" "${IREE_COMPILER_RENAME_SCRIPT}"
        --nm "${IREE_LLVM_NM}"
        --cxxfilt "${IREE_LLVM_CXXFILT}"
        --input "${_input}"
        --out "${_map}"
        --symbol-prefix "${IREE_COMPILER_ABI_PREFIX}"
      DEPENDS "${_RULE_TARGET}" "${IREE_COMPILER_RENAME_SCRIPT}"
      COMMENT "Computing llvm/mlir rename map for ${_name}"
      VERBATIM
    )
    add_custom_command(
      OUTPUT "${_out}"
      COMMAND
        "${IREE_LLVM_OBJCOPY}" "--redefine-syms=${_map}" "${_input}" "${_out}"
      DEPENDS "${_input}" "${_map}"
      COMMENT "Renaming llvm/mlir symbols in ${_name}"
      VERBATIM
    )
    list(APPEND _renamed "${_out}")
    math(EXPR _index "${_index} + 1")
  endforeach()

  # MODULE libraries need a source.
  set(_stub "${CMAKE_CURRENT_BINARY_DIR}/${_name}_stub.c")
  if(NOT EXISTS "${_stub}")
    file(WRITE "${_stub}" "// Empty stub; content comes from the renamed archive.\n")
  endif()
  add_library(${_name} MODULE "${_stub}")
  if(APPLE)
    target_link_options(${_name} PRIVATE
      "-Wl,-force_load,${_renamed}"
      "-Wl,-undefined,dynamic_lookup"
    )
  else()
    # ELF leaves undefined symbols alone.
    target_link_options(${_name} PRIVATE
      "-Wl,--whole-archive" ${_renamed} "-Wl,--no-whole-archive"
    )
  endif()
  add_custom_target(${_name}_renamed_deps DEPENDS ${_renamed})
  add_dependencies(${_name} ${_name}_renamed_deps)
  set_target_properties(${_name} PROPERTIES LINK_DEPENDS "${_renamed}")
endfunction()
