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
    ""
    "HDRS;TARGETS"
    ${ARGN}
  )

  set_property(TARGET ${_RULE_TARGETS} APPEND PROPERTY EXPORT_PROPERTIES iree_ALIAS_TO)
  foreach(_target ${_RULE_TARGETS})
    if(_RULE_FIX_INCLUDE_DIRS)
      get_target_property(_include_dirs ${_target} INTERFACE_INCLUDE_DIRECTORIES)
      list(TRANSFORM _include_dirs PREPEND "$<BUILD_INTERFACE:")
      list(TRANSFORM _include_dirs APPEND ">")
      set_target_properties(${_target} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_include_dirs}")
    endif()
  endforeach()

  set_property(GLOBAL APPEND PROPERTY IREE_EXPORT_TARGETS ${_RULE_TARGETS})

  # The export name is set at a directory level to control export.
  install(
    TARGETS ${_RULE_TARGETS}
    EXPORT IREEExported
    COMPONENT IREEDevLibraries
    EXCLUDE_FROM_ALL
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

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
        COMPONENT IREEDevLibraries
        EXCLUDE_FROM_ALL
      )
    endforeach()
  endif()
endfunction()

function(iree_generate_export_targets)
  get_property(_export_targets GLOBAL PROPERTY IREE_EXPORT_TARGETS)
  export(TARGETS ${_export_targets}
    FILE ${IREE_BINARY_DIR}/lib/cmake/IREE/IREETargets.cmake)

  install(
    EXPORT IREEExported
    FILE "IREETargets.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/IREE"
  )

  # Clear the export targets so that innocent aggregating projects don't
  # get in trouble if they use our setup.
  set_property(GLOBAL PROPERTY IREE_EXPORT_TARGETS)
endfunction()
