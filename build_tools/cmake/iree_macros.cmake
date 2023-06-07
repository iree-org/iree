# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

#-------------------------------------------------------------------------------
# Missing CMake Variables
#-------------------------------------------------------------------------------

if(${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Windows")
  set(IREE_HOST_SCRIPT_EXT "bat")
  # https://gitlab.kitware.com/cmake/cmake/-/issues/17553
  set(IREE_HOST_EXECUTABLE_SUFFIX ".exe")
else()
  set(IREE_HOST_SCRIPT_EXT "sh")
  set(IREE_HOST_EXECUTABLE_SUFFIX "")
endif()


#-------------------------------------------------------------------------------
# IREE_ARCH: identifies the target CPU architecture. May be empty when this is
# ill-defined, such as multi-architecture builds.
# This should be kept consistent with the C preprocessor token IREE_ARCH defined
# in target_platform.h.
#-------------------------------------------------------------------------------

# First, get the raw CMake architecture name, not yet normalized. Even that is
# non-trivial: it usually is CMAKE_SYSTEM_PROCESSOR, but on some platforms, we
# have to read other variables instead.
if(CMAKE_OSX_ARCHITECTURES)
  # Borrowing from:
  # https://boringssl.googlesource.com/boringssl/+/c5f0e58e653d2d9afa8facc090ce09f8aaa3fa0d/CMakeLists.txt#43
  # https://github.com/google/XNNPACK/blob/2eb43787bfad4a99bdb613111cea8bc5a82f390d/CMakeLists.txt#L40
  list(LENGTH CMAKE_OSX_ARCHITECTURES NUM_ARCHES)
  if(${NUM_ARCHES} EQUAL 1)
    # Only one arch in CMAKE_OSX_ARCHITECTURES, use that.
    set(_IREE_UNNORMALIZED_ARCH "${CMAKE_OSX_ARCHITECTURES}")
  endif()
  # Leaving _IREE_UNNORMALIZED_ARCH empty disables arch code paths. We will
  # issue a performance warning about that below.
elseif(CMAKE_GENERATOR MATCHES "^Visual Studio " AND CMAKE_GENERATOR_PLATFORM)
  # Borrowing from:
  # https://github.com/google/XNNPACK/blob/2eb43787bfad4a99bdb613111cea8bc5a82f390d/CMakeLists.txt#L50
  set(_IREE_UNNORMALIZED_ARCH "${CMAKE_GENERATOR_PLATFORM}")
else()
  set(_IREE_UNNORMALIZED_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
endif()

string(TOLOWER "${_IREE_UNNORMALIZED_ARCH}" _IREE_UNNORMALIZED_ARCH_LOWERCASE)

# Normalize _IREE_UNNORMALIZED_ARCH into IREE_ARCH.
if(EMSCRIPTEN)
  # TODO: figure what to do about the wasm target, which masquerades as x86.
  # This is the one case where the IREE_ARCH CMake variable is currently
  # inconsistent with the IREE_ARCH C preprocessor token.
  set(IREE_ARCH "")
elseif((_IREE_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "aarch64") OR
        (_IREE_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "arm64") OR
        (_IREE_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "arm64e") OR
        (_IREE_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "arm64ec"))
  set(IREE_ARCH "arm_64")
elseif((_IREE_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "arm") OR
        (_IREE_UNNORMALIZED_ARCH_LOWERCASE MATCHES "^armv[5-8]"))
  set(IREE_ARCH "arm_32")
elseif((_IREE_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "x86_64") OR
        (_IREE_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "amd64") OR
        (_IREE_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "x64"))
  set(IREE_ARCH "x86_64")
elseif((_IREE_UNNORMALIZED_ARCH_LOWERCASE MATCHES "^i[3-7]86$") OR
        (_IREE_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "x86") OR
        (_IREE_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "win32"))
  set(IREE_ARCH "x86_32")
elseif(_IREE_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "riscv64")
  set(IREE_ARCH "riscv_64")
elseif(_IREE_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "riscv32")
  set(IREE_ARCH "riscv_32")
elseif(_IREE_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "")
  set(IREE_ARCH "")
  message(WARNING "Performance advisory: architecture-specific code paths "
    "disabled because no target architecture was specified or we didn't know "
    "which CMake variable to read. Some relevant CMake variables:\n"
    "CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}\n"
    "CMAKE_GENERATOR=${CMAKE_GENERATOR}\n"
    "CMAKE_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}\n"
    "CMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}\n"
    )
else()
  set(IREE_ARCH "")
  message(SEND_ERROR "Unrecognized target architecture ${_IREE_UNNORMALIZED_ARCH_LOWERCASE}")
endif()

# iree_arch_to_llvm_arch()
#
# Helper mapping an architecture in IREE's naming scheme (as in IREE_ARCH)
# to an architecture in LLVM's naming scheme (as in LLVM target triples).
function(iree_arch_to_llvm_arch DST_LLVM_ARCH_VARIABLE SRC_ARCH)
  if("${SRC_ARCH}" STREQUAL "arm_64")
    set(${DST_LLVM_ARCH_VARIABLE} "aarch64" PARENT_SCOPE)
  elseif("${SRC_ARCH}" STREQUAL "arm_32")
    set(${DST_LLVM_ARCH_VARIABLE} "arm" PARENT_SCOPE)
  elseif("${SRC_ARCH}" STREQUAL "x86_64")
    set(${DST_LLVM_ARCH_VARIABLE} "x86_64" PARENT_SCOPE)
  elseif("${SRC_ARCH}" STREQUAL "x86_32")
    set(${DST_LLVM_ARCH_VARIABLE} "i386" PARENT_SCOPE)
  elseif("${SRC_ARCH}" STREQUAL "riscv_64")
    set(${DST_LLVM_ARCH_VARIABLE} "riscv64" PARENT_SCOPE)
  elseif("${SRC_ARCH}" STREQUAL "riscv_32")
    set(${DST_LLVM_ARCH_VARIABLE} "riscv32" PARENT_SCOPE)
  elseif("${SRC_ARCH}" STREQUAL "wasm_64")
    set(${DST_LLVM_ARCH_VARIABLE} "wasm64" PARENT_SCOPE)
  elseif("${SRC_ARCH}" STREQUAL "wasm_32")
    set(${DST_LLVM_ARCH_VARIABLE} "wasm32" PARENT_SCOPE)
  else()
    message(SEND_ERROR "What is the LLVM name of the architecture that we call ${SRC_ARCH} ?")
    set(${DST_LLVM_ARCH_VARIABLE} "unknown" PARENT_SCOPE)
  endif()
endfunction()

# iree_arch_to_llvm_target()
#
# Helper mapping an architecture in IREE's naming scheme (as in IREE_ARCH)
# to a LLVM CPU target (as in LLVM_TARGETS_TO_BUILD, the CMake variable).
function(iree_arch_to_llvm_target DST_LLVM_TARGET_VARIABLE SRC_ARCH)
  if("${SRC_ARCH}" STREQUAL "arm_64")
    set(${DST_LLVM_TARGET_VARIABLE} "AArch64" PARENT_SCOPE)
  elseif("${SRC_ARCH}" STREQUAL "arm_32")
    set(${DST_LLVM_TARGET_VARIABLE} "ARM" PARENT_SCOPE)
  elseif("${SRC_ARCH}" MATCHES "^x86_")
    set(${DST_LLVM_TARGET_VARIABLE} "X86" PARENT_SCOPE)
  elseif("${SRC_ARCH}" MATCHES "^riscv_")
    set(${DST_LLVM_TARGET_VARIABLE} "RISCV" PARENT_SCOPE)
  elseif("${SRC_ARCH}" MATCHES "^wasm_")
    set(${DST_LLVM_TARGET_VARIABLE} "WebAssembly" PARENT_SCOPE)
  else()
    message(SEND_ERROR "What is the LLVM target handling of the architecture that we call ${SRC_ARCH} ?")
    set(${DST_LLVM_TARGET_VARIABLE} "" PARENT_SCOPE)
  endif()
endfunction()

# iree_compiler_targeting_iree_arch
#
# Helper returning true if we are building the IREE compiler with the llvm-cpu
# backend enabled and with the LLVM target supporting the CPU architecture
# give in IREE's naming scheme (as in IREE_ARCH).
function(iree_compiler_targeting_iree_arch DST_VAR SRC_ARCH)
  if (NOT IREE_BUILD_COMPILER OR NOT IREE_TARGET_BACKEND_LLVM_CPU)
    set(${DST_VAR} OFF PARENT_SCOPE)
    return()
  endif()
  
  iree_arch_to_llvm_target(_LLVM_TARGET "${SRC_ARCH}")
  if (_LLVM_TARGET IN_LIST LLVM_TARGETS_TO_BUILD)
    set(${DST_VAR} ON PARENT_SCOPE)
  else()
    set(${DST_VAR} OFF PARENT_SCOPE)
  endif()
endfunction()

#-------------------------------------------------------------------------------
# General utilities
#-------------------------------------------------------------------------------

# iree_to_bool
#
# Sets `variable` to `ON` if `value` is true and `OFF` otherwise.
function(iree_to_bool VARIABLE VALUE)
  if(VALUE)
    set(${VARIABLE} "ON" PARENT_SCOPE)
  else()
    set(${VARIABLE} "OFF" PARENT_SCOPE)
  endif()
endfunction()

# iree_append_list_to_string
#
# Joins ${ARGN} together as a string separated by " " and appends it to
# ${VARIABLE}.
function(iree_append_list_to_string VARIABLE)
  if(NOT "${ARGN}" STREQUAL "")
    string(JOIN " " _ARGN_STR ${ARGN})
    set(${VARIABLE} "${${VARIABLE}} ${_ARGN_STR}" PARENT_SCOPE)
  endif()
endfunction()


#-------------------------------------------------------------------------------
# Packages and Paths
#-------------------------------------------------------------------------------

# Sets ${PACKAGE_NS} to the IREE-root relative package name in C++ namespace
# format (::).
#
# Examples:
#   compiler/src/iree/compiler/Utils/CMakeLists.txt -> iree::compiler::Utils
#   runtime/src/iree/base/CMakeLists.txt -> iree::base
#   tests/e2e/CMakeLists.txt -> iree::tests::e2e
function(iree_package_ns PACKAGE_NS)
  if(DEFINED IREE_PACKAGE_ROOT_DIR)
    # If an enclosing package root dir is set, then the package is just the
    # relative part after that.
    cmake_path(RELATIVE_PATH CMAKE_CURRENT_LIST_DIR
      BASE_DIRECTORY "${IREE_PACKAGE_ROOT_DIR}"
      OUTPUT_VARIABLE _PACKAGE)
    if(_PACKAGE STREQUAL ".")
      set(_PACKAGE "")
    endif()
    if(IREE_PACKAGE_ROOT_PREFIX)
      if("${_PACKAGE}" STREQUAL "")
        set(_PACKAGE "${IREE_PACKAGE_ROOT_PREFIX}")
      else()
        set(_PACKAGE "${IREE_PACKAGE_ROOT_PREFIX}/${_PACKAGE}")
      endif()
    endif()
  else()
    # Get the relative path of the current dir (i.e. runtime/src/iree/vm).
    string(REPLACE ${IREE_ROOT_DIR} "" _IREE_RELATIVE_PATH ${CMAKE_CURRENT_LIST_DIR})
    string(SUBSTRING ${_IREE_RELATIVE_PATH} 1 -1 _IREE_RELATIVE_PATH)

    if(NOT ${CMAKE_CURRENT_LIST_DIR} MATCHES "^${IREE_ROOT_DIR}/.*")
      # Function is being called from outside IREE. Use the source-relative path.
      # Please check the README.md to see the potential risk.
      string(REPLACE ${PROJECT_SOURCE_DIR} "" _SOURCE_RELATIVE_PATH ${CMAKE_CURRENT_LIST_DIR})
      string(SUBSTRING ${_SOURCE_RELATIVE_PATH} 1 -1 _SOURCE_RELATIVE_PATH)
      set(_PACKAGE "${_SOURCE_RELATIVE_PATH}")

    # If changing the directory/package mapping rules, please also implement
    # the corresponding rule in:
    #   build_tools/bazel_to_cmake/bazel_to_cmake_targets.py
    # Some sub-trees form their own roots for package purposes. Rewrite them.
    else()
      message(SEND_ERROR "iree_package_ns(): Could not determine package for ${CMAKE_CURRENT_LIST_DIR}")
      set(_PACKAGE "iree/unknown")
    endif()
  endif()

  string(REPLACE "/" "::" _PACKAGE_NS "${_PACKAGE}")

  if(_DEBUG_IREE_PACKAGE_NAME)
    message(STATUS "iree_package_ns(): map ${CMAKE_CURRENT_LIST_DIR} -> ${_PACKAGE_NS}")
  endif()

  set(${PACKAGE_NS} ${_PACKAGE_NS} PARENT_SCOPE)
endfunction()

# Sets ${PACKAGE_NAME} to the IREE-root relative package name.
#
# Example when called from iree/base/CMakeLists.txt:
#   iree_base
function(iree_package_name PACKAGE_NAME)
  iree_package_ns(_PACKAGE_NS)
  string(REPLACE "::" "_" _PACKAGE_NAME "${_PACKAGE_NS}")
  set(${PACKAGE_NAME} ${_PACKAGE_NAME} PARENT_SCOPE)
endfunction()

# Sets ${PACKAGE_PATH} to the IREE-root relative package path.
#
# Example when called from iree/base/CMakeLists.txt:
#   iree/base
function(iree_package_path PACKAGE_PATH)
  iree_package_ns(_PACKAGE_NS)
  string(REPLACE "::" "/" _PACKAGE_PATH ${_PACKAGE_NS})
  set(${PACKAGE_PATH} ${_PACKAGE_PATH} PARENT_SCOPE)
endfunction()

# Sets ${PACKAGE_DIR} to the directory name of the current package.
#
# Example when called from iree/base/CMakeLists.txt:
#   base
function(iree_package_dir PACKAGE_DIR)
  iree_package_ns(_PACKAGE_NS)
  string(FIND "${_PACKAGE_NS}" "::" _END_OFFSET REVERSE)
  math(EXPR _END_OFFSET "${_END_OFFSET} + 2")
  string(SUBSTRING ${_PACKAGE_NS} ${_END_OFFSET} -1 _PACKAGE_DIR)
  set(${PACKAGE_DIR} ${_PACKAGE_DIR} PARENT_SCOPE)
endfunction()

#-------------------------------------------------------------------------------
# select()-like Evaluation
#-------------------------------------------------------------------------------

# Appends ${OPTS} with a list of values based on the current compiler.
#
# Example:
#   iree_select_compiler_opts(COPTS
#     CLANG
#       "-Wno-foo"
#       "-Wno-bar"
#     CLANG_CL
#       "/W3"
#     GCC
#       "-Wsome-old-flag"
#     MSVC
#       "/W3"
#   )
#
# Note that variables are allowed, making it possible to share options between
# different compiler targets.
function(iree_select_compiler_opts OPTS)
  cmake_parse_arguments(
    PARSE_ARGV 1
    _IREE_SELECTS
    ""
    ""
    "ALL;CLANG;CLANG_GTE_10;CLANG_CL;MSVC;GCC;CLANG_OR_GCC;MSVC_OR_CLANG_CL"
  )
  # OPTS is a variable containing the *name* of the variable being populated, so
  # we need to dereference it twice.
  set(_OPTS "${${OPTS}}")
  list(APPEND _OPTS "${_IREE_SELECTS_ALL}")
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    list(APPEND _OPTS "${_IREE_SELECTS_GCC}")
    list(APPEND _OPTS "${_IREE_SELECTS_CLANG_OR_GCC}")
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    if(MSVC)
      list(APPEND _OPTS ${_IREE_SELECTS_CLANG_CL})
      list(APPEND _OPTS ${_IREE_SELECTS_MSVC_OR_CLANG_CL})
    else()
      list(APPEND _OPTS ${_IREE_SELECTS_CLANG})
      if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10)
        list(APPEND _OPTS ${_IREE_SELECTS_CLANG_GTE_10})
      endif()
      list(APPEND _OPTS ${_IREE_SELECTS_CLANG_OR_GCC})
    endif()
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    list(APPEND _OPTS ${_IREE_SELECTS_MSVC})
    list(APPEND _OPTS ${_IREE_SELECTS_MSVC_OR_CLANG_CL})
  else()
    message(ERROR "Unknown compiler: ${CMAKE_CXX_COMPILER}")
    list(APPEND _OPTS "")
  endif()
  set(${OPTS} ${_OPTS} PARENT_SCOPE)
endfunction()

#-------------------------------------------------------------------------------
# Data dependencies
#-------------------------------------------------------------------------------

# Adds 'data' dependencies to a target.
#
# Parameters:
# NAME: name of the target to add data dependencies to
# DATA: List of targets and/or files in the source tree (relative to the
# project root).
function(iree_add_data_dependencies)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "DATA"
    ${ARGN}
  )

  if(NOT _RULE_DATA)
    return()
  endif()

  foreach(_DATA_LABEL ${_RULE_DATA})
    if(TARGET ${_DATA_LABEL})
      add_dependencies(${_RULE_NAME} ${_DATA_LABEL})
    else()
      # Not a target, assume to be a file instead.
      set(_FILE_PATH ${_DATA_LABEL})

      # Create a target which copies the data file into the build directory.
      # If this file is included in multiple rules, only create the target once.
      string(REPLACE "::" "_" _DATA_TARGET ${_DATA_LABEL})
      string(REPLACE "/" "_" _DATA_TARGET ${_DATA_TARGET})
      if(NOT TARGET ${_DATA_TARGET})
        set(_INPUT_PATH "${PROJECT_SOURCE_DIR}/${_FILE_PATH}")
        set(_OUTPUT_PATH "${PROJECT_BINARY_DIR}/${_FILE_PATH}")
        add_custom_target(${_DATA_TARGET}
          COMMAND ${CMAKE_COMMAND} -E copy ${_INPUT_PATH} ${_OUTPUT_PATH}
        )
      endif()

      add_dependencies(${_RULE_NAME} ${_DATA_TARGET})
    endif()
  endforeach()
endfunction()

#-------------------------------------------------------------------------------
# Emscripten
#-------------------------------------------------------------------------------

# A global counter to guarantee unique names for js library files.
set(_LINK_JS_COUNTER 1)

# Links a JavaScript library to a target using --js-library=file.js.
#
# This function is only supported when running under Emscripten (emcmake).
# This implementation is forked from `em_add_tracked_link_flag()` in
# https://github.com/emscripten-core/emscripten/blob/main/cmake/Modules/Platform/Emscripten.cmake
# with changes to be compatible with IREE project style and CMake conventions.
#
# Parameters:
# TARGET: Name of the target to link against
# SRCS: List of JavaScript source files to link
function(iree_link_js_library)
  cmake_parse_arguments(
    _RULE
    ""
    "TARGET"
    "SRCS"
    ${ARGN}
  )

  # Convert from aliased, possibly package-relative, names to target names.
  iree_package_ns(_PACKAGE_NS)
  string(REGEX REPLACE "^::" "${_PACKAGE_NS}::" _RULE_TARGET ${_RULE_TARGET})
  string(REPLACE "::" "_" _RULE_TARGET ${_RULE_TARGET})

  foreach(_SRC_FILE ${_RULE_SRCS})
    # If the JS file is changed, we want to relink dependent binaries, but
    # unfortunately it is not possible to make a link step depend directly on a
    # source file. Instead, we must make a dummy no-op build target on that
    # source file, and make the original target depend on that dummy target.

    # Sanitate the source .js filename to a good dummy filename.
    get_filename_component(_JS_NAME "${_SRC_FILE}" NAME)
    string(REGEX REPLACE "[/:\\\\.\ ]" "_" _DUMMY_JS_TARGET ${_JS_NAME})
    set(_DUMMY_LIB_NAME ${_RULE_TARGET}_${_LINK_JS_COUNTER}_${_DUMMY_JS_TARGET})
    set(_DUMMY_C_NAME "${CMAKE_BINARY_DIR}/${_DUMMY_JS_TARGET}_tracker.c")

    # Create a new static library target that with a single dummy .c file.
    add_library(${_DUMMY_LIB_NAME} STATIC ${_DUMMY_C_NAME})
    # Make the dummy .c file depend on the .js file we are linking, so that if
    # the .js file is edited, the dummy .c file, and hence the static library
    # will be rebuild (no-op). This causes the main application to be
    # relinked, which is what we want. This approach was recommended by
    # http://www.cmake.org/pipermail/cmake/2010-May/037206.html
    add_custom_command(
      OUTPUT ${_DUMMY_C_NAME}
      COMMAND ${CMAKE_COMMAND} -E touch ${_DUMMY_C_NAME}
      DEPENDS ${_SRC_FILE}
    )
    target_link_libraries(${_RULE_TARGET}
      PUBLIC
        ${_DUMMY_LIB_NAME}
    )

    # Link the js-library to the target.
    # When a linked library starts with a "-" cmake will just add it to the
    # linker command line as it is. The advantage of doing it this way is
    # that the js-library will also be automatically linked to targets that
    # depend on this target.
    get_filename_component(_SRC_ABSOLUTE_PATH "${_SRC_FILE}" ABSOLUTE)
    target_link_libraries(${_RULE_TARGET}
      PUBLIC
        "--js-library \"${_SRC_ABSOLUTE_PATH}\""
    )

    math(EXPR _LINK_JS_COUNTER "${_LINK_JS_COUNTER} + 1")
  endforeach()
endfunction()

#-------------------------------------------------------------------------------
# Tool symlinks
#-------------------------------------------------------------------------------

# iree_symlink_tool
#
# Adds a command to TARGET which symlinks a tool from elsewhere
# (FROM_TOOL_TARGET_NAME) to a local file name (TO_EXE_NAME) in the current
# binary directory.
#
# Parameters:
#   TARGET: Local target to which to add the symlink command (i.e. an
#     iree_py_library, etc).
#   FROM_TOOL_TARGET: Target of the tool executable that is the source of the
#     link.
#   TO_EXE_NAME: The executable name to output in the current binary dir.
function(iree_symlink_tool)
  cmake_parse_arguments(
    _RULE
    ""
    "TARGET;FROM_TOOL_TARGET;TO_EXE_NAME"
    ""
    ${ARGN}
  )

  # Transform TARGET
  iree_package_ns(_PACKAGE_NS)
  iree_package_name(_PACKAGE_NAME)
  set(_TARGET "${_PACKAGE_NAME}_${_RULE_TARGET}")
  set(_FROM_TOOL_TARGET ${_RULE_FROM_TOOL_TARGET})
  set(_TO_TOOL_PATH "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_TO_EXE_NAME}${CMAKE_EXECUTABLE_SUFFIX}")
  get_filename_component(_TO_TOOL_DIR "${_TO_TOOL_PATH}" DIRECTORY)


  add_custom_command(
    TARGET "${_TARGET}"
    BYPRODUCTS
      "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_TO_EXE_NAME}${CMAKE_EXECUTABLE_SUFFIX}"
    COMMAND
      ${CMAKE_COMMAND} -E make_directory "${_TO_TOOL_DIR}"
    COMMAND
      ${CMAKE_COMMAND} -E create_symlink
        "$<TARGET_FILE:${_FROM_TOOL_TARGET}>"
        "${_TO_TOOL_PATH}"
  )
endfunction()


#-------------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------------

# iree_check_defined
#
# A lightweight way to check that all the given variables are defined. Useful
# in cases like checking that a function has been passed all required arguments.
# Doesn't give usage-specific error messages, but still significantly better
# than no error checking.
# Variable names should be passed directly without quoting or dereferencing.
# Example:
#   iree_check_defined(_SOME_VAR _AND_ANOTHER_VAR)
macro(iree_check_defined)
  foreach(_VAR ${ARGN})
    if(NOT DEFINED "${_VAR}")
      message(SEND_ERROR "${_VAR} is not defined")
    endif()
  endforeach()
endmacro()

# iree_validate_required_arguments
#
# Validates that no arguments went unparsed or were given no values and that all
# required arguments have values. Expects to be called after
# cmake_parse_arguments and verifies that the variables it creates have been
# populated as appropriate.
function(iree_validate_required_arguments
         PREFIX
         REQUIRED_ONE_VALUE_KEYWORDS
         REQUIRED_MULTI_VALUE_KEYWORDS)
  if(DEFINED ${PREFIX}_UNPARSED_ARGUMENTS)
    message(SEND_ERROR "Unparsed argument(s): '${${PREFIX}_UNPARSED_ARGUMENTS}'")
  endif()
  if(DEFINED ${PREFIX}_KEYWORDS_MISSING_VALUES)
    message(SEND_ERROR
            "No values for field(s) '${${PREFIX}_KEYWORDS_MISSING_VALUES}'")
  endif()

  foreach(_KEYWORD IN LISTS REQUIRED_ONE_VALUE_KEYWORDS REQUIRED_MULTI_VALUE_KEYWORDS)
    if(NOT DEFINED ${PREFIX}_${_KEYWORD})
      message(SEND_ERROR "Missing required argument ${_KEYWORD}")
    endif()
  endforeach()
endfunction()

# iree_compile_flags_for_patform
#
# Helper function to add necessary compile flags based on platform-specific
# configurations. Note the flags are added for cpu backends only.
function(iree_compile_flags_for_platform OUT_FLAGS IN_FLAGS)
  if(NOT (IN_FLAGS MATCHES "iree-hal-target-backends=llvm-cpu" OR
          IN_FLAGS MATCHES "iree-hal-target-backends=vmvx"))
    set(${OUT_FLAGS} "" PARENT_SCOPE)
    return()
  endif()

  if(ANDROID AND NOT IN_FLAGS MATCHES "iree-llvmcpu-target-triple")
    # Android's CMake toolchain defines some variables that we can use to infer
    # the appropriate target triple from the configured settings:
    # https://developer.android.com/ndk/guides/cmake#android_platform
    #
    # In typical CMake fashion, the various strings are pretty fuzzy and can
    # have multiple values like "latest", "android-25"/"25"/"android-N-MR1".
    #
    # From looking at the toolchain file, ANDROID_PLATFORM_LEVEL seems like it
    # should pretty consistently be just a number we can use for target triple.
    set(_TARGET_TRIPLE "aarch64-none-linux-android${ANDROID_PLATFORM_LEVEL}")
    list(APPEND _FLAGS "--iree-llvmcpu-target-triple=${_TARGET_TRIPLE}")
  endif()

  if(IREE_ARCH STREQUAL "riscv_64" AND
     CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
     NOT IN_FLAGS MATCHES "iree-llvmcpu-target-triple")
    # RV64 Linux crosscompile toolchain can support iree-compile with
    # specific CPU flags. Add the llvm flags to support RV64 RVV codegen if
    # llvm-target-triple is not specified.
    list(APPEND _FLAGS ${RISCV64_TEST_DEFAULT_LLVM_FLAGS})
  elseif(IREE_ARCH STREQUAL "riscv_32" AND
         CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
         NOT IN_FLAGS MATCHES "iree-llvmcpu-target-triple")
    # RV32 Linux crosscompile toolchain can support iree-compile with
    # specific CPU flags. Add the llvm flags to support RV32 RVV codegen if
    # llvm-target-triple is not specified.
    list(APPEND _FLAGS ${RISCV32_TEST_DEFAULT_LLVM_FLAGS})
  endif()

  if(EMSCRIPTEN AND NOT IN_FLAGS MATCHES "iree-llvmcpu-target-triple")
    set(_EMSCRIPTEN_TEST_DEFAULT_FLAGS
      "--iree-llvmcpu-target-triple=wasm32-unknown-emscripten"
    )
    list(APPEND _FLAGS ${_EMSCRIPTEN_TEST_DEFAULT_FLAGS})
  endif()

  set(${OUT_FLAGS} "${_FLAGS}" PARENT_SCOPE)
endfunction()
