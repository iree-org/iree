# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This package defines "mondo" static and shared libraries corresponding
# to various slices of IREE's codebase (intended for external consumption).
# Generally, libraries are aggregated per-directory.

# iree_aggregate_library()
#
# Builds aggregate static (and shared in the future) libraries from all
# objects built in libraries as part of a list of packages.
#
# Parameters:
# NAME: name of target (see Note)
# DEPS: List of other libraries to be linked in to the binary targets.
# PACKAGES: List of underscore_cased package names like "iree_hal". Does
# not recurse into sub-packages.
# OUTPUT_NAME: The 'OUTPUT_NAME' property, which controls the actual filename
# on disk.
#
# iree_cc_library(
#   NAME
#     awesome
#   PACKAGES
#     iree_hal
#     iree_vm
# )

function(iree_aggregate_library)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;OUTPUT_NAME"
    "DEPS;PACKAGES"
    ${ARGN}
  )

  # Replace dependencies passed by ::name with iree::package::name
  iree_package_ns(_PACKAGE_NS)
  list(TRANSFORM _RULE_DEPS REPLACE "^::" "${_PACKAGE_NS}::")

  # Prefix the library with the package name, so we get: iree_package_name.
  iree_package_name(_PACKAGE_NAME)
  set(_STATIC_NAME "${_PACKAGE_NAME}_${_RULE_NAME}_static")

  if(NOT _RULE_OUTPUT_NAME)
    set(_RULE_OUTPUT_NAME "${_RULE_NAME}")
  endif()
  # Accumulate generator expressions for each package.
  set(_OBJECTS)
  foreach(pkg ${_RULE_PACKAGES})
    list(APPEND _OBJECTS
      $<GENEX_EVAL:$<TARGET_PROPERTY:__IREE_OBJECT_MAPPINGS__,IREE_OBJECTS_${pkg}>>)
  endforeach()

  add_library(${_STATIC_NAME} STATIC ${_OBJECTS})
  set_target_properties(${_STATIC_NAME}
    PROPERTIES
      OUTPUT_NAME ${_RULE_OUTPUT_NAME}
  )
  add_library(${_PACKAGE_NS}::${_RULE_NAME} ALIAS ${_STATIC_NAME})

  # Make it convenient for in-CMake users to find headers, link options, etc.
  target_include_directories(${_STATIC_NAME}
    INTERFACE
      "$<BUILD_INTERFACE:${IREE_SOURCE_DIR}>"
      "$<BUILD_INTERFACE:${IREE_BINARY_DIR}>"
  )
  target_link_options(${_STATIC_NAME}
    INTERFACE
      ${IREE_DEFAULT_LINKOPTS}
  )
  target_link_libraries(${_STATIC_NAME}
    PUBLIC
      ${_RULE_DEPS}
  )

  # TODO: Also build a shared library if PIC and enabled, etc.

endfunction()
