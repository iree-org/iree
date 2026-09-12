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

  message(STATUS "Registering static linked compiler plugin '${_RULE_PLUGIN_ID}' (${_RULE_TARGET})")
  set_property(GLOBAL APPEND PROPERTY IREE_COMPILER_LINKED_PLUGIN_IDS "${_RULE_PLUGIN_ID}")
  set_property(GLOBAL APPEND PROPERTY IREE_COMPILER_LINKED_PLUGIN_LIBS "${_RULE_TARGET}")
endfunction()

# Registers a hal plugin that can be activated with
# -DIREE_EXTERNAL_HAL_DRIVERS=NAME.
#
# This is usually either done at the top-level CMakeLists for IREE or
# in an iree_runtime_plugin.cmake file included via IREE_CMAKE_PLUGIN_PATHS.
#
# Since external HAL driver building is typically fully optional, they should
# be in a CMake directory tree that is not included by default in the project.
# Use the SOURCE_DIR and BINARY_DIR params to configure. If the BINARY_DIR
# is relative, it will be relative to runtime/plugins/hal/drivers. If omitted,
# it will be runtime/plugins/hal/drivers/${NAME}
#
# Args:
#  NAME: The name of the hal driver.
#  OPTIONAL: Whether the driver is optional and it is not an error if it is
#    disabled.
#  SOURCE_DIR: Source directory to include if the plugin is enabled.
#  BINARY_DIR: Binary directory corresponding to the SOURCE_DIR.
#  DRIVER_TARGET: CMake target to be linked to.
#  REGISTER_FN: C symbol of the registration function.
function(iree_register_external_hal_driver)
  cmake_parse_arguments(
    _RULE
    "OPTIONAL"
    "NAME;SOURCE_DIR;BINARY_DIR;DRIVER_TARGET;REGISTER_FN"
    ""
    ${ARGN}
  )

  # Normalize binary dir.
  if(NOT _RULE_BINARY_DIR)
    set(_RULE_BINARY_DIR "${_RULE_NAME}")
  endif()
  cmake_path(ABSOLUTE_PATH _RULE_BINARY_DIR
    BASE_DIRECTORY "${IREE_BINARY_DIR}/runtime/plugins/hal/drivers")

  string(TOUPPER "${_RULE_NAME}" _NAME_SPEC)
  string(REGEX REPLACE "-" "_" _NAME_SPEC ${_NAME_SPEC})

  get_property(_prev_available GLOBAL PROPERTY IREE_EXTERNAL_HAL_DRIVERS_AVAILABLE)
  if(_RULE_NAME IN_LIST _prev_available)
    message(SEND_ERROR "iree_register_external_hal_driver(NAME ${_RULE_NAME}) called more than once")
  endif()

  set_property(GLOBAL APPEND PROPERTY IREE_EXTERNAL_HAL_DRIVERS_AVAILABLE "${_RULE_NAME}")
  if(_RULE_OPTIONAL)
    set_property(GLOBAL_PROPERTY "IREE_EXTERNAL_${_NAME_SPEC}_HAL_DRIVER_OPTIONAL" TRUE)
  endif()
  set_property(GLOBAL PROPERTY "IREE_EXTERNAL_${_NAME_SPEC}_HAL_DRIVER_SOURCE_DIR"
    "${_RULE_SOURCE_DIR}")
  set_property(GLOBAL PROPERTY "IREE_EXTERNAL_${_NAME_SPEC}_HAL_DRIVER_BINARY_DIR"
    "${_RULE_BINARY_DIR}")
  set_property(GLOBAL PROPERTY "IREE_EXTERNAL_${_NAME_SPEC}_HAL_DRIVER_TARGET"
    "${_RULE_DRIVER_TARGET}")
  set_property(GLOBAL PROPERTY "IREE_EXTERNAL_${_NAME_SPEC}_HAL_DRIVER_REGISTER"
    "${_RULE_REGISTER_FN}")
endfunction()

# iree_include_cmake_plugin_dirs()
# Configures all in-tree and out-of-tree plugins based on global settings.
function(iree_include_cmake_plugin_dirs)
  cmake_parse_arguments(
    _RULE
    ""
    "LOG_LABEL;BINARY_DIR;PLUGIN_CMAKE_FILE"
    ""
    ${ARGN}
  )

  file(MAKE_DIRECTORY "${_RULE_BINARY_DIR}")

  # Generate an include() in BINARY_DIR/CMakeLists.txt to sub-include each
  # plugin cmake file.
  set(_contents)
  foreach(_d ${IREE_CMAKE_BUILTIN_PLUGIN_PATHS} ${IREE_CMAKE_PLUGIN_PATHS})
    cmake_path(ABSOLUTE_PATH _d BASE_DIRECTORY "${IREE_SOURCE_DIR}" NORMALIZE)
    set(_f "${_d}/${_RULE_PLUGIN_CMAKE_FILE}")
    if(NOT EXISTS "${_f}")
      continue()
    endif()
    string(APPEND _contents "include(${_f})\n")
  endforeach()
  file(CONFIGURE OUTPUT "${_RULE_BINARY_DIR}/CMakeLists.txt"
    CONTENT "${_contents}" @ONLY
  )
  unset(_contents)
  unset(_d)
  unset(_f)

  message(STATUS "Configuring IREE ${_RULE_LOG_LABEL} plugins")
  add_subdirectory("${_RULE_BINARY_DIR}" "${_RULE_BINARY_DIR}")
endfunction()

# iree_compiler_register_dynamic_plugin()
# Builds a registration target into a module the compiler dlopens. The dynamic
# counterpart of iree_compiler_register_plugin, which links the same kind of
# target into the compiler instead.
#
# Whole-archive linked: the entry point is found by dlsym, not by references
# from the link. The target's own deps are left out, to resolve from
# libIREECompiler in the host process.
#
# Args:
#  PACKAGE: Package namespace override, as iree_compiler_register_plugin.
#  PLUGIN_ID: Id the plugin registers under. Also names the module.
#  TARGET: The registration target.
function(iree_compiler_register_dynamic_plugin)
  cmake_parse_arguments(
    _RULE
    ""
    "PACKAGE;PLUGIN_ID;TARGET"
    ""
    ${ARGN}
  )

  if(NOT IREE_COMPILER_DYNAMIC_PLUGINS)
    return()
  endif()
  include(iree_renamed_link)

  if(_RULE_PACKAGE)
    set(_PACKAGE_NS "${_RULE_PACKAGE}")
  else()
    iree_package_ns(_PACKAGE_NS)
  endif()

  # Replace target passed by ::name with iree::package::name
  list(TRANSFORM _RULE_TARGET REPLACE "^::" "${_PACKAGE_NS}::")

  # The rename consumes objects, so reach through the alias to the object
  # library iree_cc_library compiles them into. Header-only (INTERFACE)
  # libraries have none and cannot carry a plugin.
  get_target_property(_ALIASED_TARGET "${_RULE_TARGET}" ALIASED_TARGET)
  if(_ALIASED_TARGET)
    set(_TARGET_NAME "${_ALIASED_TARGET}")
  else()
    set(_TARGET_NAME "${_RULE_TARGET}")
  endif()
  set(_OBJECTS_NAME "${_TARGET_NAME}.objects")
  if(NOT TARGET "${_OBJECTS_NAME}")
    message(SEND_ERROR
        "iree_compiler_register_dynamic_plugin: '${_RULE_TARGET}' has no compiled "
        "objects (header-only libraries cannot carry a plugin)")
    return()
  endif()

  message(STATUS "Registering dynamic compiler plugin '${_RULE_PLUGIN_ID}' (${_RULE_TARGET})")

  set(_NAME "iree_compiler_plugin_${_RULE_PLUGIN_ID}")
  iree_renamed_link_sanitize_name("${_NAME}" _BASE)
  # The target's own libraries are not in libIREECompiler, so the plugin has to
  # carry them, renamed alike. The walk skips whatever the compiler holds.
  iree_collect_static_link_closure(_CLOSURE_LIBS _CLOSURE_OTHER "${_TARGET_NAME}")
  get_property(_COMPILER_LIBS GLOBAL PROPERTY
    IREE_RENAMED_SHARED_IMPL_OBJECT_LIBS)
  set(_CLOSURE_ARCHIVES)
  foreach(_LIB ${_CLOSURE_LIBS})
    if(_LIB IN_LIST _COMPILER_LIBS)
      continue()
    endif()
    iree_renamed_link_sanitize_name("${_NAME}_${_LIB}" _CLOSURE_BASE)
    iree_renamed_archive(_CLOSURE_RENAMED
      NAME "${_CLOSURE_BASE}"
      INPUT "$<TARGET_FILE:${_LIB}>"
      DEPENDS "${_LIB}"
    )
    list(APPEND _CLOSURE_ARCHIVES "${_CLOSURE_RENAMED}")
  endforeach()

  iree_renamed_archive_from_objects(_RENAMED
    NAME
      "${_BASE}"
    TARGET
      "${_OBJECTS_NAME}"
  )

  # MODULE libraries need at least one source; the payload arrives via
  # whole-archive linking of the renamed archive.
  set(_STUB "${CMAKE_CURRENT_BINARY_DIR}/${_NAME}_stub.c")
  if(NOT EXISTS "${_STUB}")
    file(WRITE "${_STUB}" "// Empty stub; content comes from the renamed archive.\n")
  endif()
  add_library(${_NAME} MODULE "${_STUB}")
  if(APPLE)
    target_link_options(${_NAME} PRIVATE
      "-Wl,-force_load,${_RENAMED}"
      ${_CLOSURE_ARCHIVES}
      "-Wl,-undefined,dynamic_lookup"
    )
  else()
    # -shared leaves undefined symbols unresolved by default on ELF.
    target_link_options(${_NAME} PRIVATE
      "-Wl,--whole-archive" "${_RENAMED}" "-Wl,--no-whole-archive"
      "-Wl,--start-group" ${_CLOSURE_ARCHIVES} "-Wl,--end-group"
    )
  endif()
  add_custom_target(${_NAME}_renamed_deps
    DEPENDS "${_RENAMED}" ${_CLOSURE_ARCHIVES})
  add_dependencies(${_NAME} ${_NAME}_renamed_deps)
  set_target_properties(${_NAME} PROPERTIES
    LINK_DEPENDS "${_RENAMED};${_CLOSURE_ARCHIVES}")
endfunction()
