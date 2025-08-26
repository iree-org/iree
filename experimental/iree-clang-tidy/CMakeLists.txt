# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build IREE-specific clang-tidy checks as a module.
# This requires LLVM/Clang to be built with clang-tools-extra.

# Build as a clang-tidy module plugin.
# We avoid IREE build goo here to make this something we could build separately.
add_library(IREEClangTidyModule MODULE
  IREEClangTidyModule.cpp
  IREETestCheck.cpp
)

target_compile_options(IREEClangTidyModule PRIVATE
  -Wall
  -Wextra
  -fno-exceptions
)

# Disable RTTI to match LLVM/Clang build settings.
if(NOT LLVM_ENABLE_RTTI)
  if(MSVC)
    target_compile_options(IREEClangTidyModule PRIVATE /GR-)
  else()
    target_compile_options(IREEClangTidyModule PRIVATE -fno-rtti)
  endif()
endif()

# Include directories for LLVM/Clang headers.
target_include_directories(IREEClangTidyModule PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${CLANG_INCLUDE_DIRS}
)

# Add clang-tidy headers if available.
# This is only required in bundled builds but could be used when unbundled if
# clang-tools-extra lives somewhere else.
if(CLANG_TOOLS_EXTRA_INCLUDE_DIR)
  target_include_directories(IREEClangTidyModule PRIVATE
    ${CLANG_TOOLS_EXTRA_INCLUDE_DIR}/clang-tidy
  )
endif()

# Don't link against any libraries - let symbols be resolved from the main
# binary. This keeps the final binary tiny. I'm not sure how stable it is or
# what other issues may arise, but it Works On My Machine(tm).
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # Linux: Allow undefined symbols to be resolved at load time.
  set_target_properties(IREEClangTidyModule PROPERTIES
    LINK_FLAGS "-Wl,--allow-shlib-undefined"
  )
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  # macOS: Allow dynamic lookup of undefined symbols.
  set_target_properties(IREEClangTidyModule PROPERTIES
    LINK_FLAGS "-Wl,-undefined,dynamic_lookup"
  )
endif()

# Set output properties.
set_target_properties(IREEClangTidyModule PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  PREFIX ""
)

iree_add_all_subdirs()
