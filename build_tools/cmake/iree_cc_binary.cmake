# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

# iree_cc_binary()
#
# CMake function to imitate Bazel's cc_binary rule.
#
# Parameters:
# NAME: name of target (see Usage below)
# SRCS: List of source files for the binary
# DATA: List of other targets and files required for this binary
# DEPS: List of other libraries to be linked in to the binary targets
# DISABLE_LLVM_LINK_LLVM_DYLIB: Disables linking against the libLLVM.so dynamic
#   library, even if the build is configured to do so. This must be used with
#   care as it can only contain dependencies and be used by binaries that also
#   so disable it (either in upstream LLVM or locally). In practice, it is used
#   for LLVM dependency chains that must always result in static-linked tools.
# COPTS: List of private compile options
# DEFINES: List of public defines
# LINKOPTS: List of link options
# TESTONLY: for testing; won't compile when tests are disabled
# HOSTONLY: host only; compile using host toolchain when cross-compiling
# SETUP_INSTALL_RPATH: Sets an install RPATH which assumes the standard
#   directory layout (to be used if linking against installed shared libs).
# INSTALL_COMPONENT: CMake install component (Defaults to "IREETool-${_RULE_NAME}").
# Note:
# iree_cc_binary will create a binary called ${PACKAGE_NAME}_${NAME}, e.g.
# iree_base_foo with an alias to ${PACKAGE_NS}::${NAME}.
#
# Usage:
# iree_cc_library(
#   NAME
#     awesome
#   HDRS
#     "a.h"
#   SRCS
#     "a.cc"
#   PUBLIC
# )
#
# iree_cc_binary(
#   NAME
#     awesome_tool
#   SRCS
#     "awesome-tool-main.cc"
#   DEPS
#     iree::awesome
# )
function(iree_cc_binary)
  cmake_parse_arguments(
    _RULE
    "EXCLUDE_FROM_ALL;HOSTONLY;TESTONLY;SETUP_INSTALL_RPATH;DISABLE_LLVM_LINK_LLVM_DYLIB"
    "NAME;INSTALL_COMPONENT"
    "SRCS;COPTS;DEFINES;LINKOPTS;DATA;DEPS"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()

  # Prefix the library with the package name, so we get: iree_package_name
  iree_package_name(_PACKAGE_NAME)
  iree_package_ns(_PACKAGE_NS)
  if("${_PACKAGE_NAME}" STREQUAL "")
    set(_NAME "${_RULE_NAME}")
  else()
    set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")
  endif()

  if(_DEBUG_IREE_PACKAGE_NAME)
    message(STATUS "  : iree_cc_binary(${_NAME})")
  endif()

  add_executable(${_NAME} "")

  if(NOT "${_PACKAGE_NS}" STREQUAL "")
    # Alias the iree_package_name binary to iree::package::name.
    # This lets us more clearly map to Bazel and makes it possible to
    # disambiguate the underscores in paths vs. the separators.
    if(_DEBUG_IREE_PACKAGE_NAME)
      message(STATUS "  + alias ${_PACKAGE_NS}::${_RULE_NAME}")
    endif()
    add_executable(${_PACKAGE_NS}::${_RULE_NAME} ALIAS ${_NAME})

    # If the binary name matches the package then treat it as a default. For
    # example, foo/bar/ library 'bar' would end up as 'foo::bar'. This isn't
    # likely to be common for binaries, but is consistent with the behavior for
    # libraries and in Bazel.
    iree_package_dir(_PACKAGE_DIR)
    if("${_RULE_NAME}" STREQUAL "${_PACKAGE_DIR}")
      add_executable(${_PACKAGE_NS} ALIAS ${_NAME})
    endif()
  endif()

  set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "${_RULE_NAME}")
  if(_RULE_SRCS)
    target_sources(${_NAME}
      PRIVATE
        ${_RULE_SRCS}
    )
  else()
    set(_DUMMY_SRC "${CMAKE_CURRENT_BINARY_DIR}/${_NAME}_dummy.cc")
    iree_make_empty_file("${_DUMMY_SRC}")
    target_sources(${_NAME}
      PRIVATE
        ${_DUMMY_SRC}
    )
  endif()
  target_include_directories(${_NAME} SYSTEM
    PUBLIC
      "$<BUILD_INTERFACE:${IREE_SOURCE_DIR}>"
      "$<BUILD_INTERFACE:${IREE_BINARY_DIR}>"
  )
  target_compile_definitions(${_NAME}
    PUBLIC
      ${_RULE_DEFINES}
  )
  target_compile_options(${_NAME}
    PRIVATE
      ${IREE_DEFAULT_COPTS}
      ${_RULE_COPTS}
  )
  target_link_options(${_NAME}
    PRIVATE
      ${IREE_DEFAULT_LINKOPTS}
      ${_RULE_LINKOPTS}
  )

  # Replace dependencies passed by ::name with iree::package::name
  list(TRANSFORM _RULE_DEPS REPLACE "^::" "${_PACKAGE_NS}::")
  if(NOT _RULE_DISABLE_LLVM_LINK_LLVM_DYLIB)
    iree_redirect_llvm_dylib_deps(_RULE_DEPS)
  endif()

  # Implicit deps.
  if(IREE_IMPLICIT_DEFS_CC_DEPS)
    list(APPEND _RULE_DEPS ${IREE_IMPLICIT_DEFS_CC_DEPS})
  endif()

  target_link_libraries(${_NAME}
    PUBLIC
      ${_RULE_DEPS}
  )
  iree_add_data_dependencies(NAME ${_NAME} DATA ${_RULE_DATA})

  # Add all IREE targets to a folder in the IDE for organization.
  set_property(TARGET ${_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER}/binaries)

  set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD ${IREE_CXX_STANDARD})
  set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

  set(_INSTALL_COMPONENT "${_RULE_INSTALL_COMPONENT}")
  if(NOT _INSTALL_COMPONENT)
    set(_INSTALL_COMPONENT "IREETool-${_RULE_NAME}")
  endif()

  if(_RULE_EXCLUDE_FROM_ALL)
    set_property(TARGET ${_NAME} PROPERTY EXCLUDE_FROM_ALL ON)
    install(TARGETS ${_NAME}
            RENAME ${_RULE_NAME}
            COMPONENT ${_INSTALL_COMPONENT}
            RUNTIME DESTINATION bin
            BUNDLE DESTINATION bin
            EXCLUDE_FROM_ALL)
  else()
    install(TARGETS ${_NAME}
      RENAME ${_RULE_NAME}
      COMPONENT ${_INSTALL_COMPONENT}
      RUNTIME DESTINATION bin
      BUNDLE DESTINATION bin)
  endif()

  # Setup RPATH if on a Unix-like system. We have two use cases that we are
  # handling here:
  #   1. Install tree layouts like bin/ and lib/ directories that are
  #      peers.
  #   2. Single directory bundles (language bindings do this) where the
  #      shared library is placed next to the consumer.
  #
  # The common solution is to use an RPATH of the origin and the
  # lib/ directory that may be a peer of the origin. Distributions
  # outside of this setup will need to do their own manipulation.
  if(_RULE_SETUP_INSTALL_RPATH)
    if(APPLE OR UNIX)
      set(_origin_prefix "\$ORIGIN")
      if(APPLE)
        set(_origin_prefix "@loader_path")
      endif()
      # See: https://cmake.org/cmake/help/latest/module/GNUInstallDirs.html
      # Assume relative path as a sibling of the lib dir.
      set(_lib_dir "${CMAKE_INSTALL_LIBDIR}")
      if (NOT _lib_dir)
        set(_lib_dir "lib")
      endif()
      set(_install_rpath "${_origin_prefix}:${_origin_prefix}/../${_lib_dir}")
      if(_lib_dir)
        cmake_path(IS_ABSOLUTE _lib_dir _is_abs_libdir)
        if(_is_abs_libdir)
          # Use the libdir verbatim.
          set(_install_rpath "${_origin_prefix}:${_lib_dir}")
        endif()
      endif()
      set_target_properties(${_NAME} PROPERTIES
        BUILD_WITH_INSTALL_RPATH OFF
        INSTALL_RPATH "${_install_rpath}"
      )
    endif()
  endif()

  # Set up Info.plist properties when building macOS/iOS app bundles.
  get_target_property(APPLE_BUNDLE ${_NAME} MACOSX_BUNDLE)
  if (APPLE_BUNDLE)
    set_target_properties(${_NAME} PROPERTIES
      MACOSX_BUNDLE_BUNDLE_NAME "${_RULE_NAME}"
      MACOSX_BUNDLE_GUI_IDENTIFIER "dev.iree.${_RULE_NAME}"
      MACOSX_BUNDLE_COPYRIGHT "Copyright © 2023 The IREE Authors"
      # These are just placeholder version numbers until we define proper
      # version scheme and support.
      MACOSX_BUNDLE_BUNDLE_VERSION 0.1
      MACOSX_BUNDLE_SHORT_VERSION_STRING 0.1
      MACOSX_BUNDLE_LONG_VERSION_STRING 0.1)
  endif()
endfunction()
