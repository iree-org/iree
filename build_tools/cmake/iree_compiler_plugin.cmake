# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Ids of all plugins that we have statically linked.
set_property(GLOBAL PROPERTY IREE_COMPILER_LINKED_PLUGIN_IDS "")
# Libraries to add to the plugin registry for all statically linked plugins.
set_property(GLOBAL PROPERTY IREE_COMPILER_LINKED_PLUGIN_LIBS "")

# iree_compiler_register_plugin()
# Within a plugin package, registers the plugin by id with the build system,
# associating it with a registration target.
function(iree_compiler_register_plugin)
  cmake_parse_arguments(
    _RULE
    ""
    "PACKAGE;PLUGIN_ID;TARGET"
    ""
    ${ARGN}
  )

  # Prefix the library with the package name, so we get: iree_package_name.
  if(_RULE_PACKAGE)
    set(_PACKAGE_NS "${_RULE_PACKAGE}")
    string(REPLACE "::" "_" _PACKAGE_NAME ${_RULE_PACKAGE})
  else()
    iree_package_ns(_PACKAGE_NS)
    iree_package_name(_PACKAGE_NAME)
  endif()

  # Replace target passed by ::name with iree::package::name
  list(TRANSFORM _RULE_TARGET REPLACE "^::" "${_PACKAGE_NS}::")

  # TODO: Can have more control on what gets linked.
  message(STATUS "Registering static linked compiler plugin '${_RULE_PLUGIN_ID}' (${_RULE_TARGET})")
  set_property(GLOBAL APPEND PROPERTY IREE_COMPILER_LINKED_PLUGIN_IDS "${_RULE_PLUGIN_ID}")
  set_property(GLOBAL APPEND PROPERTY IREE_COMPILER_LINKED_PLUGIN_LIBS "${_RULE_TARGET}")
endfunction()

# iree_compiler_configure_plugins()
# Configures all in-tree and out-of-tree plugins based on global settings.
function(iree_compiler_configure_plugins)
  set(IREE_COMPILER_PLUGINS_BINARY_DIR "${IREE_BINARY_DIR}/compiler/plugins")
  file(MAKE_DIRECTORY "${IREE_COMPILER_PLUGINS_BINARY_DIR}")

  # Generate an include() in compiler/plugins/CMakeLists.txt for each plugin dir.
  set(_contents)
  foreach(_d ${IREE_COMPILER_BUILTIN_PLUGIN_PATHS})
    cmake_path(ABSOLUTE_PATH _d BASE_DIRECTORY "${IREE_SOURCE_DIR}" NORMALIZE)
    set(_f "${_d}/iree_compiler_plugin_group.cmake")
    if(NOT EXISTS "${_f}")
      message(SEND_ERROR "Specified plugin directory ${_d} does not contain an iree_compiler_plugin_group.cmake file")
      continue()
    endif()
    string(APPEND _contents "include(${_f})\n")
  endforeach()
  foreach(_d ${IREE_COMPILER_PLUGIN_PATHS})
    cmake_path(ABSOLUTE_PATH _d BASE_DIRECTORY "${IREE_SOURCE_DIR}" NORMALIZE)
    set(_f "${_d}/iree_compiler_plugin_group.cmake")
    if(NOT EXISTS "${_f}")
      message(SEND_ERROR "Specified plugin directory ${_d} does not contain an iree_compiler_plugin_group.cmake file")
      continue()
    endif()
    string(APPEND _contents "include(${_f})\n")
  endforeach()
  file(CONFIGURE OUTPUT "${IREE_COMPILER_PLUGINS_BINARY_DIR}/CMakeLists.txt"
    CONTENT "${_contents}" @ONLY
  )
  unset(_contents)
  unset(_d)
  unset(_f)

  message(STATUS "Configuring compiler plugins")
  # Force enable BUILD_SHARED_LIBS for the compiler if instructed.
  set(_IREE_ORIG_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
  if(IREE_COMPILER_BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
  endif()

  add_subdirectory("${IREE_COMPILER_PLUGINS_BINARY_DIR}" "${IREE_COMPILER_PLUGINS_BINARY_DIR}")

  # Reset BUILD_SHARED_LIBS.
  set(BUILD_SHARED_LIBS ${_IREE_ORIG_BUILD_SHARED_LIBS} CACHE BOOL "" FORCE)
endfunction()
