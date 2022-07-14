# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Appends ${VALUE} to each argument.
function(iree_append_to_lists VALUE)
  foreach(_VARIABLE ${ARGN})
    set(${_VARIABLE} "${${_VARIABLE}} ${VALUE}" PARENT_SCOPE)
  endforeach(_VARIABLE)
endfunction()

if(IREE_ENABLE_LLD)
  if(IREE_USE_LINKER)
    message(FATAL_ERROR "IREE_ENABLE_LLD and IREE_USE_LINKER can't be set at the same time")
  endif()
  set(IREE_USE_LINKER "lld")
endif()

if(IREE_USE_LINKER)
  set(IREE_LINKER_FLAG "-fuse-ld=${IREE_USE_LINKER}")

  # Depending on how the C compiler is invoked, it may trigger an unused
  # argument warning about -fuse-ld, which can foul up compiler flag detection,
  # causing false negatives. We lack a finer grained way to suppress such a
  # thing, and this is deemed least bad.
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    iree_append_to_lists("-Wno-unused-command-line-argument"
      CMAKE_REQUIRED_FLAGS
      CMAKE_EXE_LINKER_FLAGS
      CMAKE_MODULE_LINKER_FLAGS
      CMAKE_SHARED_LINKER_FLAGS
    )
  endif()

  iree_append_to_lists("${IREE_LINKER_FLAG}"
    CMAKE_REQUIRED_FLAGS
    CMAKE_EXE_LINKER_FLAGS
    CMAKE_MODULE_LINKER_FLAGS
    CMAKE_SHARED_LINKER_FLAGS
  )
  include(CheckCXXSourceCompiles)
  include(CheckCSourceCompiles)
  set(MINIMAL_SRC "int main() { return 0; }")
  check_cxx_source_compiles("${MINIMAL_SRC}" CXX_SUPPORTS_CUSTOM_LINKER)
  check_c_source_compiles("${MINIMAL_SRC}" CC_SUPPORTS_CUSTOM_LINKER)

  # Note: if you see errors here, check
  #   * logs in CMakeFiles/CMakeError.log in your build directory
  #   * that you have a recent version of your chosen linker (for example:
  #     install the version of lld that we use in our Docker images)
  if(NOT CXX_SUPPORTS_CUSTOM_LINKER)
    message(FATAL_ERROR "Compiler '${CMAKE_CXX_COMPILER}' does not support '${IREE_LINKER_FLAG}'")
  endif()
  if(NOT CC_SUPPORTS_CUSTOM_LINKER)
    message(FATAL_ERROR "Compiler '${CMAKE_C_COMPILER}' does not support '${IREE_LINKER_FLAG}'")
  endif()
endif()
