# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(IREE_COMPILER_PLUGINS "" CACHE STRING "List of named in-tree plugins (under compiler/plugins) to statically compile")
set(IREE_COMPILER_PLUGIN_PATHS "" CACHE STRING "Paths to external compiler plugins")

# Ids of all plugins that have been included in the configure step. This
# may include plugins that we do not statically link but we do build.
set_property(GLOBAL PROPERTY IREE_COMPILER_INCLUDED_PLUGIN_IDS "")
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

  # Do nothing if plugin registration is taking place standalone.
  # See the IREE_COMPILER_IN_ADD_PLUGIN variable set in
  # iree_compiler_add_plugin.
  if(NOT IREE_COMPILER_IN_ADD_PLUGIN)
    message(STATUS "Not registering plugin (out of tree)")
    return()
  endif()

  # TODO: Can have more control on what gets linked.
  message(STATUS "Registering static linked compiler plugin ${_RULE_PLUGIN_ID}")
  set_property(GLOBAL APPEND PROPERTY IREE_COMPILER_LINKED_PLUGIN_IDS "${_RULE_PLUGIN_ID}")
  set_property(GLOBAL APPEND PROPERTY IREE_COMPILER_LINKED_PLUGIN_LIBS "${_RULE_TARGET}")
endfunction()

# iree_compiler_configure_plugins()
# Configures all in-tree and out-of-tree plugins based on global settings.
function(iree_compiler_configure_plugins)
  # Process in-tree plugins.
  foreach(_plugin_id ${IREE_COMPILER_PLUGINS})
    set(_plugin_src_dir "${IREE_SOURCE_DIR}/compiler/plugins/${_plugin_id}")
    iree_compiler_add_plugin("${_plugin_id}" "${_plugin_src_dir}")
  endforeach()
  unset(_plugin_id)
  unset(_plugin_src_dir)

  # Process out of tree plugins.
  foreach(_plugin_src_dir ${IREE_COMPILER_PLUGIN_PATHS})
    # TODO: Support some path mangling to allow overriding the plugin id
    # if it is not literally the last path component.
    cmake_path(ABSOLUTE_PATH _plugin_src_dir BASE_DIRECTORY "${IREE_SOURCE_DIR}" NORMALIZE)
    cmake_path(GET _plugin_src_dir FILENAME _plugin_id)
    iree_compiler_add_plugin("${_plugin_id}" "${_plugin_src_dir}")
  endforeach()
endfunction()

# iree_compiler_add_plugin(src bin)
# Adds a compiler plugin based on id and source directory.
function(iree_compiler_add_plugin plugin_id plugin_src_dir)
  # Add a guard so that we know if we are adding an in-tree plugin.
  if(IREE_COMPILER_IN_ADD_PLUGIN)
    message(FATAL_ERROR "Cannot recursively add plugins")
  endif()

  message(STATUS "Adding static compiler plugin ${plugin_id} (from ${plugin_src_dir})")
  get_property(_existing_plugin_ids GLOBAL PROPERTY IREE_COMPILER_INCLUDED_PLUGIN_IDS)
  if("${plugin_id}" IN_LIST _existing_plugin_ids)
    message(SEND_ERROR "Plugin already registered: ${_plugin_id}")
    return()
  endif()

  # Include it.
  set(IREE_COMPILER_IN_ADD_PLUGIN "${plugin_id}")
  set_property(GLOBAL APPEND PROPERTY IREE_COMPILER_INCLUDED_PLUGIN_IDS "${plugin_id}")
  set(_binary_dir "${IREE_BINARY_DIR}/compiler/plugins/${_plugin_id}")

  # Force enable BUILD_SHARED_LIBS for the compiler if instructed.
  set(_IREE_ORIG_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
  if(IREE_COMPILER_BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
  endif()

  add_subdirectory("${plugin_src_dir}" "${_binary_dir}")

  # Reset BUILD_SHARED_LIBS.
  set(BUILD_SHARED_LIBS ${_IREE_ORIG_BUILD_SHARED_LIBS} CACHE BOOL "" FORCE)

  unset(IREE_COMPILER_IN_ADD_PLUGIN)
endfunction()
