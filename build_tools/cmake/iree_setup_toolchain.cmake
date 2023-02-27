# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CheckCXXCompilerFlag)
include(CheckLinkerFlag)
include(CheckSymbolExists)

# Appends ${VALUE} to each argument.
function(iree_append_to_lists VALUE)
  foreach(_VARIABLE ${ARGN})
    set(${_VARIABLE} "${${_VARIABLE}} ${VALUE}" PARENT_SCOPE)
  endforeach(_VARIABLE)
endfunction()

#-------------------------------------------------------------------------------
# Supports dynamic library loading.
#-------------------------------------------------------------------------------

set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_DL_LIBS})
check_symbol_exists(dlopen dlfcn.h IREE_HAVE_DLOPEN)
unset(CMAKE_REQUIRED_LIBRARIES)
if(WIN32 OR IREE_HAVE_DLOPEN)
  set(IREE_HAVE_DYNAMIC_LIBRARY_LOADING ON)
else()
  set(IREE_HAVE_DYNAMIC_LIBRARY_LOADING OFF)
endif()

#-------------------------------------------------------------------------------
# Compiler flag support
#-------------------------------------------------------------------------------

check_cxx_compiler_flag(-fvisibility=default IREE_SUPPORTS_VISIBILITY_DEFAULT)

#-------------------------------------------------------------------------------
# Linker setup
#-------------------------------------------------------------------------------

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

#-------------------------------------------------------------------------------
# Sanitizer configurations
#-------------------------------------------------------------------------------

# Note: we add these flags to the global CMake flags, not to IREE-specific
# variables such as IREE_DEFAULT_COPTS so that all symbols are consistently
# defined with the same sanitizer flags, including e.g. standard library
# symbols that might be used by both IREE and non-IREE (e.g. LLVM) code.

if(IREE_ENABLE_ASAN)
  string(APPEND CMAKE_CXX_FLAGS " -fsanitize=address")
  string(APPEND CMAKE_C_FLAGS " -fsanitize=address")
  # If doing any kind of shared library builds, then we have to link against
  # the shared libasan, and the user will be responsible for adding the
  # appropriate path to LD_LIBRARY_PATH (or else binaries will fail to launch).
  if(BUILD_SHARED_LIBS OR IREE_COMPILER_BUILD_SHARED_LIBS)
    string(APPEND CMAKE_EXE_LINKER_FLAGS " -shared-libasan")
    string(APPEND CMAKE_SHARED_LINKER_FLAGS " -shared-libasan")
  endif()
endif()
if(IREE_ENABLE_MSAN)
  string(APPEND CMAKE_CXX_FLAGS " -fsanitize=memory")
  string(APPEND CMAKE_C_FLAGS " -fsanitize=memory")
endif()
if(IREE_ENABLE_TSAN)
  string(APPEND CMAKE_CXX_FLAGS " -fsanitize=thread")
  string(APPEND CMAKE_C_FLAGS " -fsanitize=thread")
endif()
if(IREE_ENABLE_UBSAN)
  string(APPEND CMAKE_CXX_FLAGS " -fsanitize=undefined")
  string(APPEND CMAKE_C_FLAGS " -fsanitize=undefined")
endif()

#-------------------------------------------------------------------------------
# Build performance optimizations
#-------------------------------------------------------------------------------

# Split DWARF breaks debug information out of object files and stores them in
# separate .dwo files. This reduces a lot of needless I/O during normal build
# activities. It consists of the -gsplit-dwarf compiler flag and (for maximum
# effect) the --gdb-index linker flag, which just emits an index to binaries
# instead of full debug contents. gdb-index is supported by gold and partially
# supported by LLD (LLD supports it if split-dwarf objects were compiled with
# ggnu-pubnames).
# If https://gitlab.kitware.com/cmake/cmake/-/issues/21179 is ever implemented,
# use that.
if(IREE_ENABLE_SPLIT_DWARF)
  check_cxx_compiler_flag(-gsplit-dwarf IREE_SUPPORTS_SPLIT_DWARF)
  if(IREE_SUPPORTS_SPLIT_DWARF)
    # Also add -ggnu-pubnames for compilation because it links faster and lld
    # doesn't do the slow path without it.
    iree_append_to_lists(" -gsplit-dwarf -ggnu-pubnames"
      CMAKE_C_FLAGS_DEBUG
      CMAKE_CXX_FLAGS_DEBUG
      CMAKE_C_FLAGS_RELWITHDEBINFO
      CMAKE_CXX_FLAGS_RELWITHDEBINFO
    )
  endif()
  check_linker_flag(CXX "-Wl,--gdb-index" IREE_SUPPORTS_GDB_INDEX)
  if(IREE_SUPPORTS_GDB_INDEX)
    message(STATUS "Enabling gdb-index (binaries with debug info are not relocatable)")
    iree_append_to_lists(" -Wl,--gdb-index"
      CMAKE_EXE_LINKER_FLAGS_DEBUG
      CMAKE_MODULE_LINKER_FLAGS_DEBUG
      CMAKE_SHARED_LINKER_FLAGS_DEBUG
      CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO
      CMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO
      CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO
    )
  endif()
endif()

# Thin archives makes static archives that only link to backing object files
# instead of embedding them. This makes them non-relocatable but is almost
# always the right thing outside of certain deployment/packaging scenarios.
if(IREE_ENABLE_THIN_ARCHIVES)
  execute_process(COMMAND ${CMAKE_AR} -V OUTPUT_VARIABLE IREE_AR_VERSION)
  if ("${IREE_AR_VERSION}" MATCHES "^GNU ar|LLVM")
    message(STATUS "Enabling thin archives (static libraries will not be relocatable)")
    set(CMAKE_C_ARCHIVE_APPEND "<CMAKE_AR> qT <TARGET> <LINK_FLAGS> <OBJECTS>")
    set(CMAKE_CXX_ARCHIVE_APPEND "<CMAKE_AR> qT <TARGET> <LINK_FLAGS> <OBJECTS>")
    set(CMAKE_C_ARCHIVE_CREATE "<CMAKE_AR> crT <TARGET> <LINK_FLAGS> <OBJECTS>")
    set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> crT <TARGET> <LINK_FLAGS> <OBJECTS>")
  else()
    message(WARNING "Thin archives requested but not supported by ar")
  endif()
endif()
