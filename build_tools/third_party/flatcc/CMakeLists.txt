# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# ----------------------------------------------------------------------
# This file mimics the targets in bazel flatcc/BUILD.overlay using only CMake
# targets provided in flatcc's config package file.
#
# Sticking to upstream CMake targets, rather than creating new CMake targets
# built using source files inside the flatcc submodule, ensures that we could
# (optionally) build using a system installed version of flatcc with
# find_package. Upstream's targets are slightly larger than those built by
# bazel, but LTO should remove any unused code.
# ----------------------------------------------------------------------

# Configure flatcc build options
if(IREE_HOST_BIN_DIR)
  set(FLATCC_RTONLY ON) # No need to build CLI, it will be imported
  iree_import_binary(NAME iree-flatcc-cli)
else()
  set(FLATCC_RTONLY OFF) # Build CLI in addition to runtime library
endif()
set(FLATCC_TEST OFF)
cmake_policy(PUSH)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
add_subdirectory(${IREE_ROOT_DIR}/third_party/flatcc ${CMAKE_CURRENT_BINARY_DIR}/flatcc EXCLUDE_FROM_ALL)
cmake_policy(POP)

# flatcc sets some properties to odd values, not sure why, we just set them to
# something normal.
if(NOT IREE_HOST_BIN_DIR)
  set_target_properties(flatcc flatccrt flatcc_cli PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/flatcc/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/flatcc/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/flatcc/bin"
  )
else()
  set_target_properties(flatccrt PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/flatcc/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/flatcc/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/flatcc/bin"
  )
endif()

# Add aliases to match bazel build
iree_add_alias_library(flatcc::parsing flatccrt)
iree_add_alias_library(flatcc::runtime flatccrt)

# Library install
iree_install_targets(
  TARGETS flatccrt
  COMPONENT IREEBundledLibraries
  EXPORT_SET Runtime
)

if(NOT IREE_HOST_BIN_DIR)
  # Create build-time alias for IREE naming
  add_executable(iree-flatcc-cli ALIAS flatcc_cli)

  # Install the CLI tool, use a variant of the install command that supports
  # renaming.
  install(
    PROGRAMS $<TARGET_FILE:flatcc_cli>
    DESTINATION bin
    RENAME iree-flatcc-cli
    COMPONENT IREETools-Runtime
  )
endif()
