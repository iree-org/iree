# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# iree_cc_fuzz()
#
# CMake function to create a libFuzzer-based fuzz target.
#
# Parameters:
# NAME: name of target. This name is used for the generated executable.
# SRCS: List of source files for the fuzzer (must define LLVMFuzzerTestOneInput).
# DATA: List of other targets and files required for this binary.
# DEPS: List of other libraries to be linked in to the binary targets.
# COPTS: List of private compile options.
# DEFINES: List of public defines.
# LINKOPTS: List of link options.
# LABELS: Additional labels to apply to the target.
#
# Note:
# - Fuzz targets require IREE_BUILD_TESTS=ON AND IREE_ENABLE_FUZZING=ON.
# - Fuzz targets are NOT added to CTest (they run differently than tests).
# - Fuzz targets are excluded from the default 'all' target (build explicitly).
# - Binary name is ${NAME} in the bin directory.
#
# Usage:
# iree_cc_fuzz(
#   NAME
#     unicode_fuzz
#   SRCS
#     "unicode_fuzz.cc"
#   DEPS
#     iree::base::internal::unicode
# )
function(iree_cc_fuzz)
  # Fuzz targets require both tests enabled AND fuzzing enabled.
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()
  if(NOT IREE_ENABLE_FUZZING)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "SRCS;COPTS;DEFINES;LINKOPTS;DATA;DEPS;LABELS"
    ${ARGN}
  )

  # Prefix the library with the package name, so we get: iree_package_name
  iree_package_name(_PACKAGE_NAME)
  iree_package_ns(_PACKAGE_NS)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  add_executable(${_NAME} "")
  # Alias the iree_package_name fuzz binary to iree::package::name.
  add_executable(${_PACKAGE_NS}::${_RULE_NAME} ALIAS ${_NAME})

  set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "${_RULE_NAME}")
  target_sources(${_NAME}
    PRIVATE
      ${_RULE_SRCS}
  )
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

  # Link with libFuzzer runtime. The -fsanitize=fuzzer flag provides the main()
  # function and fuzzing driver. All other code is compiled with
  # -fsanitize=fuzzer-no-link (set in iree_setup_toolchain.cmake) for coverage
  # instrumentation without linking the fuzzer runtime.
  target_link_options(${_NAME}
    PRIVATE
      ${IREE_DEFAULT_LINKOPTS}
      ${_RULE_LINKOPTS}
      "-fsanitize=fuzzer"
  )

  # Replace dependencies passed by ::name with iree::package::name
  list(TRANSFORM _RULE_DEPS REPLACE "^::" "${_PACKAGE_NS}::")

  # Implicit deps.
  if(IREE_IMPLICIT_DEFS_CC_DEPS)
    list(APPEND _RULE_DEPS ${IREE_IMPLICIT_DEFS_CC_DEPS})
  endif()

  target_link_libraries(${_NAME}
    PUBLIC
      ${_RULE_DEPS}
  )
  iree_add_data_dependencies(NAME ${_NAME} DATA ${_RULE_DATA})

  # Add all IREE fuzz targets to a folder in the IDE for organization.
  set_property(TARGET ${_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER}/fuzz)

  set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD ${IREE_CXX_STANDARD})
  set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

  # Exclude from 'all' target - fuzz targets must be built explicitly.
  set_property(TARGET ${_NAME} PROPERTY EXCLUDE_FROM_ALL ON)
endfunction()
