# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

# iree_redirect_llvm_dylib_deps(DEPS_VAR)
#
# Filters a list of CC dependencies, making alterations as needed to
# to redirect LLVM libraries to the libLLVM.so dynamic library, when LLVM
# has been configured to link against it.
# This is necessary to preserve the one-definition rule in the build graph
# in a consistent way as to how AddLLVM.cmake does it.
# Note that this only really works if libLLVM.so was configured to contain
# "all" components. If this ever becomes unworkable, we may need to port the
# component naming logic and selectively choose when to divert.
function(iree_redirect_llvm_dylib_deps DEPS_VAR)
  set(_deps ${${DEPS_VAR}})
  set(_new_deps)
  set(_modified FALSE)
  foreach(_dep ${_deps})
    # If linking against the LLVM dylib, then divert any LLVM prefixed
    # targets there.
    if(LLVM_LINK_LLVM_DYLIB AND _dep MATCHES "^LLVM")
      list(APPEND _new_deps "LLVM")
      set(_modified TRUE)
    else()
      list(APPEND _new_deps "${_dep}")
    endif()
  endforeach()
  if(_modified)
    set(${DEPS_VAR} "${_new_deps}" PARENT_SCOPE)
  endif()
endfunction()

# iree_cc_library()
#
# CMake function to imitate Bazel's cc_library rule.
#
# Parameters:
# PACKAGE: Name of the package (overrides actual path)
# NAME: name of target (see Note)
# HDRS: List of public header files for the library
# TEXTUAL_HDRS: List of public header files that cannot be compiled on their own
# SRCS: List of source files for the library
# DATA: List of other targets and files required for this binary
# DEPS: List of other libraries to be linked in to the binary targets
# DISABLE_LLVM_LINK_LLVM_DYLIB: Disables linking against the libLLVM.so dynamic
#   library, even if the build is configured to do so. This must be used with
#   care as it can only contain dependencies and be used by binaries that also
#   so disable it (either in upstream LLVM or locally). In practice, it is used
#   for LLVM dependency chains that must always result in static-linked tools.
# COPTS: List of private compile options
# DEFINES: List of public defines
# INCLUDES: Include directories to add to dependencies
# SYSTEM_INCLUDES: Include directories that should be used with "SYSTEM" scope,
#   which makes them more tolerant to certain classes of warnings and issues.
# LINKOPTS: List of link options
# PUBLIC: Add this so that this library will be exported under iree::
# Also in IDE, target will appear in IREE folder while non PUBLIC will be in IREE/internal.
# TESTONLY: When added, this target will only be built if user passes -DIREE_BUILD_TESTS=ON to CMake.
# SHARED: If set, will compile to a shared object.
# WINDOWS_DEF_FILE: If set, will add a windows .def file to a shared library link
# Note:
# By default, iree_cc_library will always create a library named iree_${NAME},
# and alias target iree::${NAME}. The iree:: form should always be used.
# This is to reduce namespace pollution.
#
# iree_cc_library(
#   NAME
#     awesome
#   HDRS
#     "a.h"
#   SRCS
#     "a.cc"
# )
# iree_cc_library(
#   NAME
#     fantastic_lib
#   SRCS
#     "b.cc"
#   DEPS
#     iree::package::awesome # not "awesome" !
#   PUBLIC
# )
#
# iree_cc_library(
#   NAME
#     main_lib
#   ...
#   DEPS
#     iree::package::fantastic_lib
# )

function(iree_cc_library)
  cmake_parse_arguments(
    _RULE
    "PUBLIC;TESTONLY;SHARED;DISABLE_LLVM_LINK_LLVM_DYLIB"
    "PACKAGE;NAME;WINDOWS_DEF_FILE"
    "HDRS;TEXTUAL_HDRS;SRCS;COPTS;DEFINES;LINKOPTS;DATA;DEPS;INCLUDES;SYSTEM_INCLUDES"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()

  # Prefix the library with the package name, so we get: iree_package_name.
  if(_RULE_PACKAGE)
    set(_PACKAGE_NS "${_RULE_PACKAGE}")
    string(REPLACE "::" "_" _PACKAGE_NAME ${_RULE_PACKAGE})
  else()
    iree_package_ns(_PACKAGE_NS)
    iree_package_name(_PACKAGE_NAME)
  endif()
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")
  set(_OBJECTS_NAME ${_NAME}.objects)

  if(_DEBUG_IREE_PACKAGE_NAME)
    message(STATUS "  : iree_cc_library(${_NAME})")
  endif()

  # Replace dependencies passed by ::name with iree::package::name
  list(TRANSFORM _RULE_DEPS REPLACE "^::" "${_PACKAGE_NS}::")
  if(NOT _RULE_DISABLE_LLVM_LINK_LLVM_DYLIB)
    iree_redirect_llvm_dylib_deps(_RULE_DEPS)
  endif()

  # Check if this is a header-only library.
  # Note that as of February 2019, many popular OS's (for example, Ubuntu
  # 16.04 LTS) only come with cmake 3.5 by default.  For this reason, we can't
  # use list(FILTER...)
  set(_CC_SRCS "${_RULE_SRCS}")
  foreach(_SRC_FILE IN LISTS _CC_SRCS)
    if(${_SRC_FILE} MATCHES ".*\\.(h|inc)")
      list(REMOVE_ITEM _CC_SRCS "${_SRC_FILE}")
    endif()
  endforeach()
  if("${_CC_SRCS}" STREQUAL "")
    set(_RULE_IS_INTERFACE 1)
  else()
    set(_RULE_IS_INTERFACE 0)
  endif()

  # Wrap user specified INCLUDES in the $<BUILD_INTERFACE:>
  # generator.
  list(TRANSFORM _RULE_INCLUDES PREPEND "$<BUILD_INTERFACE:")
  list(TRANSFORM _RULE_INCLUDES APPEND ">")
  list(TRANSFORM _RULE_SYSTEM_INCLUDES PREPEND "$<BUILD_INTERFACE:")
  list(TRANSFORM _RULE_SYSTEM_INCLUDES APPEND ">")

  # Implicit deps.
  if(IREE_IMPLICIT_DEFS_CC_DEPS)
    list(APPEND _RULE_DEPS ${IREE_IMPLICIT_DEFS_CC_DEPS})
  endif()

  if(NOT _RULE_IS_INTERFACE)
    add_library(${_OBJECTS_NAME} OBJECT)
    if(_RULE_SHARED OR BUILD_SHARED_LIBS)
      add_library(${_NAME} SHARED "$<TARGET_OBJECTS:${_OBJECTS_NAME}>")
      if(_RULE_WINDOWS_DEF_FILE AND WIN32)
        target_sources(${_NAME} PRIVATE "${_RULE_WINDOWS_DEF_FILE}")
      endif()
    else()
      add_library(${_NAME} STATIC "$<TARGET_OBJECTS:${_OBJECTS_NAME}>")
      if(_RULE_WINDOWS_DEF_FILE AND WIN32)
        message(SEND_ERROR "If specifying a .def file library must be shared")
      endif()
    endif()

    # Sources get added to the object library.
    target_sources(${_OBJECTS_NAME}
      PRIVATE
        ${_RULE_SRCS}
        ${_RULE_TEXTUAL_HDRS}
        ${_RULE_HDRS}
    )

    # Keep track of objects transitively in our special property.
    set_property(TARGET ${_NAME} PROPERTY
      INTERFACE_IREE_TRANSITIVE_OBJECTS "$<TARGET_OBJECTS:${_OBJECTS_NAME}>")
    _iree_cc_library_add_object_deps(${_NAME} ${_RULE_DEPS})

    # We define everything else on the regular rule. However, the object
    # library needs compiler definition related properties, so we forward them.
    # We also forward link libraries -- not because the OBJECT libraries do
    # linking but because they get transitive compile definitions from them.
    # Yes. This is state of the art.
    target_include_directories(${_OBJECTS_NAME}
      PUBLIC
        $<TARGET_PROPERTY:${_NAME},INTERFACE_INCLUDE_DIRECTORIES>
    )
    target_include_directories(${_OBJECTS_NAME} SYSTEM
      PUBLIC
        $<TARGET_PROPERTY:${_NAME},INTERFACE_SYSTEM_INCLUDE_DIRECTORIES>
    )
    target_include_directories(${_OBJECTS_NAME}
      PUBLIC
        $<TARGET_PROPERTY:${_NAME},INTERFACE_INCLUDE_DIRECTORIES>
    )
    target_compile_options(${_OBJECTS_NAME}
      PRIVATE
        $<TARGET_PROPERTY:${_NAME},COMPILE_OPTIONS>
    )
    target_compile_definitions(${_OBJECTS_NAME}
      PUBLIC
        $<TARGET_PROPERTY:${_NAME},INTERFACE_COMPILE_DEFINITIONS>
    )
    target_link_libraries(${_OBJECTS_NAME}
      PUBLIC
        $<TARGET_PROPERTY:${_NAME},INTERFACE_LINK_LIBRARIES>
    )

    target_include_directories(${_NAME}
      PUBLIC
        "$<BUILD_INTERFACE:${IREE_SOURCE_DIR}>"
        "$<BUILD_INTERFACE:${IREE_BINARY_DIR}>"
    )
    target_include_directories(${_NAME}
      PUBLIC
        ${_RULE_INCLUDES}
    )
    target_include_directories(${_NAME}
      SYSTEM PUBLIC
        ${_RULE_SYSTEM_INCLUDES}
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
    target_link_libraries(${_NAME}
      PUBLIC
        ${_RULE_DEPS}
        ${IREE_THREADS_DEPS}
    )

    iree_add_data_dependencies(NAME ${_NAME} DATA ${_RULE_DATA})
    target_compile_definitions(${_NAME}
      PUBLIC
        ${_RULE_DEFINES}
    )

    # If in BUILD_SHARED_LIBS mode, then we need to make sure that visibility
    # is not hidden. We default to hidden visibility in the main copts so
    # need to undo it here.
    # TODO: Switch to the CXX_VISIBILITY_PRESET property and fix the global
    # hidden setting to follow suit.
    if(BUILD_SHARED_LIBS AND IREE_SUPPORTS_VISIBILITY_DEFAULT)
      target_compile_options(${_OBJECTS_NAME} PRIVATE
        "-fvisibility=default"
      )
    endif()

    # Add all IREE targets to a folder in the IDE for organization.
    if(_RULE_PUBLIC)
      set_property(TARGET ${_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER})
      set_property(TARGET ${_OBJECTS_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER})
    elseif(_RULE_TESTONLY)
      set_property(TARGET ${_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER}/test)
      set_property(TARGET ${_OBJECTS_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER}/test)
    else()
      set_property(TARGET ${_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER}/internal)
      set_property(TARGET ${_OBJECTS_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER}/internal)
    endif()

    # INTERFACE libraries can't have the CXX_STANDARD property set so only
    # set here.
    set_property(TARGET ${_OBJECTS_NAME} PROPERTY CXX_STANDARD ${IREE_CXX_STANDARD})
    set_property(TARGET ${_OBJECTS_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
  else()
    # Generating header-only library.
    add_library(${_NAME} INTERFACE)
    target_include_directories(${_NAME}
      INTERFACE
        "$<BUILD_INTERFACE:${IREE_SOURCE_DIR}>"
        "$<BUILD_INTERFACE:${IREE_BINARY_DIR}>"
        ${_RULE_INCLUDES}
    )
    target_include_directories(${_NAME}
      SYSTEM INTERFACE
        ${_RULE_SYSTEM_INCLUDES}
    )
    target_link_options(${_NAME}
      INTERFACE
        ${IREE_DEFAULT_LINKOPTS}
        ${_RULE_LINKOPTS}
    )
    target_link_libraries(${_NAME}
      INTERFACE
        ${_RULE_DEPS}
    )
    _iree_cc_library_add_object_deps(${_NAME} ${_RULE_DEPS})
    iree_add_data_dependencies(NAME ${_NAME} DATA ${_RULE_DATA})
    target_compile_definitions(${_NAME}
      INTERFACE
        ${_RULE_DEFINES}
    )
  endif()

  if(NOT _RULE_TESTONLY)
    iree_install_targets(
      TARGETS ${_NAME}
      HDRS ${_RULE_HDRS} ${_RULE_TEXTUAL_HDRS}
    )
  endif()

  # Alias the iree_package_name library to iree::package::name.
  # This lets us more clearly map to Bazel and makes it possible to
  # disambiguate the underscores in paths vs. the separators.
  if(_DEBUG_IREE_PACKAGE_NAME)
    message(STATUS "  + alias ${_PACKAGE_NS}::${_RULE_NAME}")
  endif()
  iree_add_alias_library(${_PACKAGE_NS}::${_RULE_NAME} ${_NAME})

  if(NOT "${_PACKAGE_NS}" STREQUAL "")
    # If the library name matches the final component of the package then treat
    # it as a default. For example, foo/bar/ library 'bar' would end up as
    # 'foo::bar'.
    iree_package_dir(_PACKAGE_DIR)
    if("${_RULE_NAME}" STREQUAL "${_PACKAGE_DIR}")
      iree_add_alias_library(${_PACKAGE_NS} ${_NAME})
    endif()
  endif()
endfunction()

# _iree_cc_library_add_object_deps()
#
# Helper to add deps to an iree_cc_library. This only operates on the unaliased
# raw name (i.e. 'iree_vm_vm'), not aliased names (i.e. 'iree::vm').
#
# This appends to two properties:
#   INTERFACE_IREE_TRANSITIVE_OBJECTS: Transitive list of all objects from
#     this library and all "iree::" prefixed dependent libraries. This will
#     allow you to create mondo objects for any transtive libraries that are
#     part of IREE, but it will not contain outside.
#   INTERFACE_IREE_TRANSITIVE_OBJECT_LIBS: Transitive list of any dependency
#     targets that are not under teh "iree::" namespace but are encountered
#     in the dependency dag.
function(_iree_cc_library_add_object_deps name)
  foreach(_DEP_TARGET ${ARGN})
    if(_DEP_TARGET MATCHES "^iree::")
      set_property(TARGET ${name} APPEND PROPERTY
        INTERFACE_IREE_TRANSITIVE_OBJECTS
        "$<GENEX_EVAL:$<TARGET_PROPERTY:${_DEP_TARGET},INTERFACE_IREE_TRANSITIVE_OBJECTS>>"
      )
      set_property(TARGET ${name} APPEND PROPERTY
      INTERFACE_IREE_TRANSITIVE_OBJECT_LIBS
        "$<GENEX_EVAL:$<TARGET_PROPERTY:${_DEP_TARGET},INTERFACE_IREE_TRANSITIVE_OBJECT_LIBS>>"
      )
    else()
      set_property(TARGET ${name} APPEND PROPERTY
        INTERFACE_IREE_TRANSITIVE_OBJECT_LIBS
        ${_DEP_TARGET}
      )
    endif()
  endforeach()
endfunction()

# iree_cc_unified_library()
#
# Creates a unified library out of the iree:: namespaced transitive deps+self
# of some ROOT library. The resulting library will contain the union of all
# objects from all transitive library-deps in the iree:: namespace. Such
# libraries are typically the only libraries that we install for outside use
# and they must only be used by leaf demos or out of tree libraries/executables.
# Commingling with any regular libraries will result in duplicate symbols.
#
# Note that the resulting library will not contain any libraries outside of the
# iree:: namespace but will be configured to link to them. For external use
# it is expected that they will be installed and used separately as needed.
#
# Compile and link options are forwarded from the ROOT target non-transitively.
# Ensure that this target directly references all definitions that need to
# be exported to end consumers.
#
# Parameters:
# NAME: name of target
# ROOT: Root target library to extract objects and deps from.
function(iree_cc_unified_library)
  cmake_parse_arguments(
    _RULE
    "SHARED"
    "NAME;ROOT"
    ""
    ${ARGN}
  )

  # Replace dependencies passed by ::name with iree::package::name
  iree_package_ns(_PACKAGE_NS)
  list(TRANSFORM _RULE_ROOT REPLACE "^::" "${_PACKAGE_NS}::")

  # Prefix the library with the package name, so we get: iree_package_name.
  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  # Evaluate the object and libs.
  set(_OBJECTS "$<REMOVE_DUPLICATES:$<GENEX_EVAL:$<TARGET_PROPERTY:${_RULE_ROOT},INTERFACE_IREE_TRANSITIVE_OBJECTS>>>")
  set(_LIBS "$<REMOVE_DUPLICATES:$<GENEX_EVAL:$<TARGET_PROPERTY:${_RULE_ROOT},INTERFACE_IREE_TRANSITIVE_OBJECT_LIBS>>>")

  # For debugging, write out evaluated objects to a file.
  # This cannot be enabled for Xcode given that Xcode does not support
  # per-configuration sources.
  if (NOT "${CMAKE_GENERATOR}" STREQUAL "Xcode")
    file(GENERATE OUTPUT "${_RULE_NAME}.$<CONFIG>.contents.txt" CONTENT
      "OBJECTS:\n${_OBJECTS}\n\nLIBS:\n${_LIBS}\n")
  endif()
  if(_RULE_SHARED)
    add_library(${_NAME} SHARED ${_OBJECTS})
  else()
    add_library(${_NAME} STATIC ${_OBJECTS})
  endif()

  target_link_libraries(${_NAME}
    PUBLIC
      ${_LIBS}
  )

  # Forward compile usage requirements from the root library.
  target_include_directories(${_NAME}
    PUBLIC
      $<TARGET_PROPERTY:${_RULE_ROOT},INTERFACE_INCLUDE_DIRECTORIES>
  )
  target_include_directories(${_NAME}
    PUBLIC
      $<TARGET_PROPERTY:${_RULE_ROOT},INTERFACE_INCLUDE_DIRECTORIES>
  )
  target_compile_options(${_NAME}
    PRIVATE
      $<TARGET_PROPERTY:${_RULE_ROOT},COMPILE_OPTIONS>
  )
  target_compile_definitions(${_NAME}
    PUBLIC
      $<TARGET_PROPERTY:${_RULE_ROOT},INTERFACE_COMPILE_DEFINITIONS>
  )
  target_link_libraries(${_NAME}
    PUBLIC
      $<TARGET_PROPERTY:${_RULE_ROOT},INTERFACE_LINK_LIBRARIES>
  )

  iree_install_targets(
    TARGETS ${_NAME}
  )

  # Alias the iree_package_name library to iree::package::name.
  # This lets us more clearly map to Bazel and makes it possible to
  # disambiguate the underscores in paths vs. the separators.
  iree_add_alias_library(${_PACKAGE_NS}::${_RULE_NAME} ${_NAME})

  # If the library name matches the final component of the package then treat
  # it as a default. For example, foo/bar/ library 'bar' would end up as
  # 'foo::bar'.
  iree_package_dir(_PACKAGE_DIR)
  if(${_RULE_NAME} STREQUAL ${_PACKAGE_DIR})
    iree_add_alias_library(${_PACKAGE_NS} ${_NAME})
  endif()
endfunction()

# iree_cc_library_exclude_from_all(target exclude)
#
# For a target previously defined in the same package, set the
# EXCLUDE_FROM_ALL property.
#
# This is necessary because cc_library targets consist of multiple sub-targets
# and they all must have the property set.
function(iree_cc_library_exclude_from_all target exclude_from_all)
  iree_package_ns(_PACKAGE_NS)
  iree_package_name(_PACKAGE_NAME)

  set(_NAME "${_PACKAGE_NAME}_${target}")
  set(_OBJECTS_NAME ${_NAME}.objects)

  set_target_properties(${_NAME} ${_OBJECTS_NAME}
    PROPERTIES EXCLUDE_FROM_ALL ${exclude_from_all})
endfunction()
