# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

function(iree_add_alias_library to_target from_target)
  add_library(${to_target} ALIAS ${from_target})
  # Yes: Leading-lowercase property names are load bearing and the recommended
  # way to do this: https://gitlab.kitware.com/cmake/cmake/-/issues/19261
  # We have to export the aliases on the target, and the when we generate
  # IREEConfig.cmake, generate code to re-establish the alias on import.
  # Yeah, this is completely sane.
  set_property(TARGET ${from_target} APPEND PROPERTY iree_ALIAS_TO ${to_target})
endfunction()

function(iree_install_targets)
  cmake_parse_arguments(
    _RULE
    "FIX_INCLUDE_DIRS"
    "COMPONENT;EXPORT_SET"
    "HDRS;TARGETS"
    ${ARGN}
  )

  # Determine install component. It can be explicit on the target or implicit
  # from the CMake variable IREE_INSTALL_LIBRARY_TARGETS_DEFAULT_COMPONENT 
  # (usually set at the directory level). Note that truthy evaluation is 
  # intended: Installation can be suppressed by setting "COMPONENT OFF".
  set(_INSTALL_COMPONENT "${_RULE_COMPONENT}")
  if(NOT _INSTALL_COMPONENT AND IREE_INSTALL_LIBRARY_TARGETS_DEFAULT_COMPONENT)
    set(_INSTALL_COMPONENT "${IREE_INSTALL_LIBRARY_TARGETS_DEFAULT_COMPONENT}")
  endif()
  if(NOT _INSTALL_COMPONENT)
    return()
  endif()

  # Ditto for export set name.
  set(_EXPORT_SET "${_RULE_EXPORT_SET}")
  if(NOT _EXPORT_SET AND IREE_INSTALL_LIBRARY_TARGETS_DEFAULT_EXPORT_SET)
    set(_EXPORT_SET "${IREE_INSTALL_LIBRARY_TARGETS_DEFAULT_EXPORT_SET}")
  endif()
  if(NOT _EXPORT_SET)
    message(SEND_ERROR "Installing ${_RULE_TARGETS}: An install COMPONENT was set but EXPORT_SET was not")
  endif()

  # Process targets.
  set_property(TARGET ${_RULE_TARGETS} APPEND PROPERTY EXPORT_PROPERTIES iree_ALIAS_TO)
  foreach(_target ${_RULE_TARGETS})
    if(_RULE_FIX_INCLUDE_DIRS)
      get_target_property(_include_dirs ${_target} INTERFACE_INCLUDE_DIRECTORIES)
      list(TRANSFORM _include_dirs PREPEND "$<BUILD_INTERFACE:")
      list(TRANSFORM _include_dirs APPEND ">")
      set_target_properties(${_target} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_include_dirs}")
    endif()
  endforeach()

  # Add it to the global property that will be processed at the end of the build.
  set_property(GLOBAL APPEND PROPERTY "IREE_EXPORT_TARGETS_${_EXPORT_SET}" ${_RULE_TARGETS})

  # The export name is set at a directory level to control export.
  install(
    TARGETS ${_RULE_TARGETS}
    EXPORT IREEExported-${_EXPORT_SET}
    COMPONENT "${_INSTALL_COMPONENT}"
    EXCLUDE_FROM_ALL
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  # Install headers if the rule declares them and the directory level
  # root path is truthy.
  if(_RULE_HDRS AND IREE_HDRS_ROOT_PATH)
    foreach(_hdr_file ${_RULE_HDRS})
      set(_hdr_abs_path "${CMAKE_CURRENT_SOURCE_DIR}/${_hdr_file}")
      if(NOT EXISTS "${_hdr_abs_path}")
        # Assume it is generated in the binary dir.
        set(_hdr_abs_path "${CMAKE_CURRENT_BINARY_DIR}/${_hdr_file}")
      endif()
      set(_hdr_base_relative "${CMAKE_CURRENT_SOURCE_DIR}/${_hdr_file}")
      cmake_path(
        RELATIVE_PATH _hdr_base_relative
        BASE_DIRECTORY "${IREE_HDRS_ROOT_PATH}")
      cmake_path(
        GET _hdr_base_relative
        PARENT_PATH _rel_path
      )
      install(
        FILES "${_hdr_abs_path}"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${_rel_path}"
        COMPONENT "${_INSTALL_COMPONENT}"
        EXCLUDE_FROM_ALL
      )
    endforeach()
  endif()
endfunction()

# Generates build and install tree export CMake files for a given named
# export set.
# This will export any targets added to iree_install_targets above with the
# same export set name (either explicit or at the directory level).
# 
# This function must be called at a location after which all targets that are
# part of the export set have been added (typically at the top-level).
# Exports will not be generated if no targets were added for that export set
# name.
#
# If a COMPONENT is specified, the exports will be installed with that 
# component. Defaults to IREECMakeExports.
function(iree_generate_export_targets)
  cmake_parse_arguments(
    _RULE
    ""
    "COMPONENT;EXPORT_SET;INSTALL_DESTINATION"
    ""
    ${ARGN}
  )

  if(NOT _RULE_EXPORT_SET)
    message(FATAL_ERROR "EXPORT_SET is required")
    return()
  endif()

  get_property(_export_targets GLOBAL PROPERTY "IREE_EXPORT_TARGETS_${_RULE_EXPORT_SET}")
  if(NOT _export_targets)
    message(STATUS "Skipping generation of export set ${_RULE_EXPORT_SET} (no targets matched)")
  endif()

  set(_component "${_RULE_COMPONENT}")
  if(NOT _component)
    set(_component "IREECMakeExports")
  endif()

  export(TARGETS ${_export_targets}
    FILE "${CMAKE_CURRENT_BINARY_DIR}/IREETargets-${_RULE_EXPORT_SET}.cmake")

  install(
    EXPORT "IREEExported-${_RULE_EXPORT_SET}"
    COMPONENT "${_component}"
    FILE "IREETargets-${_RULE_EXPORT_SET}.cmake"
    DESTINATION "${_RULE_INSTALL_DESTINATION}"
  )

  # Clear the export targets so that innocent aggregating projects don't
  # get in trouble if they use our setup.
  set_property(GLOBAL PROPERTY "IREE_EXPORT_TARGETS_${_RULE_EXPORT_SET}")
endfunction()

# Adds a convenience install target for a component.
function(iree_add_install_target)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;COMPONENT"
    "DEPENDS;ADD_TO"
    ${ARGN}
  )
  set(_depends ${_RULE_DEPENDS})

  if(NOT _RULE_COMPONENT)
    # Just create stub targets.
    add_custom_target(${_RULE_NAME})
    add_custom_target(${_RULE_NAME}-stripped)
  else()
    # Create targets to install a component.
    set(_options -DCMAKE_INSTALL_COMPONENT="${_RULE_COMPONENT}")

    # Non-stripped.
    add_custom_target(
      ${_RULE_NAME}
      COMMAND "${CMAKE_COMMAND}"
              ${_options}
              -P "${IREE_BINARY_DIR}/cmake_install.cmake"
      USES_TERMINAL)
    set_target_properties(${_RULE_NAME} PROPERTIES FOLDER "Component Install Targets")

    # Stripped.
    add_custom_target(
      ${_RULE_NAME}-stripped
      COMMAND "${CMAKE_COMMAND}"
              ${_options}
              -DCMAKE_INSTALL_DO_STRIP=1
              -P "${IREE_BINARY_DIR}/cmake_install.cmake"
      USES_TERMINAL)
    set_target_properties(${_RULE_NAME} PROPERTIES FOLDER "Component Install Targets")
  endif()

  if(_depends)
    add_dependencies(${_RULE_NAME} ${_depends})
    add_dependencies(${_RULE_NAME}-stripped ${_depends})
  endif()

  if(_RULE_ADD_TO)
    foreach(_add_to ${_RULE_ADD_TO})
      add_dependencies(${_add_to} ${_RULE_NAME})
      add_dependencies(${_add_to}-stripped ${_RULE_NAME}-stripped)
    endforeach()
  endif()
endfunction()
