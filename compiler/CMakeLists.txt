# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# On Windows, DLLs go to the runtime directory and import
# libraries go to the lib directory.
# TODO: We should really be dumping binaries into bin/ not
# tools/. This must line up with binaries built this way because
# DLLs must be in the same directory as the binary.
# See: https://github.com/openxla/iree/issues/11297
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tools")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
if(WIN32)
  set(IREE_COMPILER_DYLIB_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}" CACHE INTERNAL "")
  set(IREE_COMPILER_DYLIB_INSTALL_PREFIX "bin/")
else()
  set(IREE_COMPILER_DYLIB_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" CACHE INTERNAL "")
  set(IREE_COMPILER_DYLIB_INSTALL_PREFIX "lib/")
endif()

# Always build the C bindings, since the API is available apart from
# actually building the compiler.
add_subdirectory(bindings/c)

# The compiler implementation is gated on the global setting.
if(IREE_BUILD_COMPILER)
  # Force enable BUILD_SHARED_LIBS for the compiler if instructed.
  set(_IREE_ORIG_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
  if(IREE_COMPILER_BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
  endif()

  # Must configure plugins before processing the compiler sources so that
  # the static link list can be set.
  iree_include_cmake_plugin_dirs(
    LOG_LABEL
      compiler
    BINARY_DIR
      "${IREE_BINARY_DIR}/compiler/plugins"
    PLUGIN_CMAKE_FILE
      "iree_compiler_plugin.cmake"
  )
  add_subdirectory(src)

  # Reset BUILD_SHARED_LIBS.
  set(BUILD_SHARED_LIBS ${_IREE_ORIG_BUILD_SHARED_LIBS} CACHE BOOL "" FORCE)

  # Copy Python packaging files to the build dir so that we can install from
  # there.
  if(IREE_BUILD_PYTHON_BINDINGS)
    configure_file(pyproject.toml pyproject.toml COPYONLY)
    configure_file(setup.py setup.py @ONLY)
    add_subdirectory(bindings/python)
  endif()

  # Post processing.
  get_property(_iree_compiler_dylib_depends GLOBAL PROPERTY IREE_COMPILER_DYLIB_DEPENDS)
  if(_iree_compiler_dylib_depends)
    add_dependencies(iree_compiler_API_SharedImpl ${_iree_compiler_dylib_depends})
  endif()
endif()

