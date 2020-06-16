# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(CMakeParseArguments)

if (NOT DEFINED _IREE_PY_EXTENSION_NAMES)
  set(_IREE_PY_EXTENSION_NAMES "")
endif()

# iree_py_extension()
#
# CMake function to imitate Bazel's iree_py_extension rule.
#
# Parameters:
# NAME: name of target
# HDRS: List of public header files for the library
# SRCS: List of source files for the library
# DEPS: List of other libraries to be linked in to the py extension targets
# COPTS: List of private compile options
# DEFINES: List of public defines
# INCLUDES: Include directories to add to dependencies
# LINKOPTS: List of link options
# TYPE: Type of library to be crated: either "MODULE", "SHARED" or "STATIC" (default).
# TESTONLY: When added, this target will only be built if user passes -DIREE_BUILD_TESTS=ON to CMake.

function(iree_py_extension)

  cmake_parse_arguments(
    _RULE
    "TESTONLY"
    "NAME"
    "HDRS;SRCS;COPTS;DEFINES;LINKOPTS;DEPS;INCLUDES;TYPE"
    ${ARGN}
  )

  iree_package_ns(_PACKAGE_NS)
  # Replace dependencies passed by ::name with ::iree::package::name
  list(TRANSFORM _RULE_DEPS REPLACE "^::" "${_PACKAGE_NS}::")

  if(NOT _RULE_TESTONLY OR IREE_BUILD_TESTS)
    # Prefix the library with the package name, so we get: iree_package_name.
    iree_package_name(_PACKAGE_NAME)
    set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

    if(NOT _RULE_TYPE)
      set(_RULE_TYPE "STATIC")
    endif()

    string(TOUPPER "${_RULE_TYPE}" _uppercase_RULE_TYPE)
    if(NOT _uppercase_RULE_TYPE MATCHES "^(STATIC|SHARED|MODULE)")
      message(FATAL_ERROR "Unsported library TYPE for iree_pybind_cc_library: ${_RULE_TYPE}")
    endif()


    add_library(${_NAME} ${_uppercase_RULE_TYPE} "")
    target_sources(${_NAME}
      PRIVATE
        ${_RULE_SRCS}
        ${_RULE_TEXTUAL_HDRS}
        ${_RULE_HDRS}
    )

    target_include_directories(${_NAME}
      PUBLIC
        "$<BUILD_INTERFACE:${IREE_COMMON_INCLUDE_DIRS}>"
        "$<BUILD_INTERFACE:${_RULE_INCLUDES}>"
      PRIVATE
        ${PYBIND11_INCLUDE_DIR}
        ${PYTHON_INCLUDE_DIR}
    )

    target_compile_options(${_NAME}
      PRIVATE
        ${_RULE_COPTS}
        ${IREE_DEFAULT_COPTS}
    )

    target_compile_definitions(${_NAME}
      PUBLIC
        ${_RULE_DEFINES}
    )

    set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "${_RULE_NAME}")

    if(NOT _uppercase_RULE_TYPE MATCHES "STATIC")
      set_property(TARGET ${_NAME} PROPERTY PREFIX "${PYTHON_MODULE_PREFIX}")
      set_property(TARGET ${_NAME} PROPERTY SUFFIX "${PYTHON_MODULE_EXTENSION}")
    endif()

    # Alias the iree_package_name library to iree::package::name.
    # This lets us more clearly map to Bazel and makes it possible to
    # disambiguate the underscores in paths vs. the separators.
    add_library(${_PACKAGE_NS}::${_RULE_NAME} ALIAS ${_NAME})
    iree_package_dir(_PACKAGE_DIR)
    if(${_RULE_NAME} STREQUAL ${_PACKAGE_DIR})
      # If the library name matches the package then treat it as a default.
      # For example, foo/bar/ library 'bar' would end up as 'foo::bar'.
      add_library(${_PACKAGE_NS} ALIAS ${_NAME})
    endif()
    # Defer computing transitive dependencies and calling target_link_libraries()
    # until all libraries have been declared.
    # Track target and deps, use in iree_complete_py_extension_link_options() later.
    set_property(GLOBAL APPEND PROPERTY _IREE_PY_EXTENSION_NAMES "${_NAME}")
    set_property(TARGET ${_NAME} PROPERTY DIRECT_DEPS ${_RULE_DEPS})
  endif()
endfunction()

# Lists all transitive dependencies of DIRECT_DEPS in TRANSITIVE_DEPS.
function(_iree_transitive_dependencies DIRECT_DEPS TRANSITIVE_DEPS)
  set(_TRANSITIVE "")

  foreach(_DEP ${DIRECT_DEPS})
    _iree_transitive_dependencies_helper(${_DEP} _TRANSITIVE)
  endforeach(_DEP)

  set(${TRANSITIVE_DEPS} "${_TRANSITIVE}" PARENT_SCOPE)
endfunction()

# Recursive helper function for _iree_transitive_dependencies.
# Performs a depth-first search through the dependency graph, appending all
# dependencies of TARGET to the TRANSITIVE_DEPS list.
function(_iree_transitive_dependencies_helper TARGET TRANSITIVE_DEPS)
  if (NOT TARGET "${TARGET}")
    # Excluded from the project, or invalid name? Just ignore.
    return()
  endif()

  # Resolve aliases, canonicalize name formatting.
  get_target_property(_ALIASED_TARGET ${TARGET} ALIASED_TARGET)
  if(_ALIASED_TARGET)
    set(_TARGET_NAME ${_ALIASED_TARGET})
  else()
    string(REPLACE "::" "_" _TARGET_NAME ${TARGET})
  endif()

  set(_RESULT "${${TRANSITIVE_DEPS}}")
  if (${_TARGET_NAME} IN_LIST _RESULT)
    # Already visited, ignore.
    return()
  endif()

  # Append this target to the list. Dependencies of this target will be added
  # (if valid and not already visited) in recursive function calls.
  list(APPEND _RESULT ${_TARGET_NAME})

  # Check for non-target identifiers again after resolving the alias.
  if (NOT TARGET ${_TARGET_NAME})
    return()
  endif()

  # Get the list of direct dependencies for this target.
  get_target_property(_TARGET_TYPE ${_TARGET_NAME} TYPE)
  if(NOT ${_TARGET_TYPE} STREQUAL "INTERFACE_LIBRARY")
    get_target_property(_TARGET_DEPS ${_TARGET_NAME} LINK_LIBRARIES)
  else()
    get_target_property(_TARGET_DEPS ${_TARGET_NAME} INTERFACE_LINK_LIBRARIES)
  endif()

  if(_TARGET_DEPS)
    # Recurse on each dependency.
    foreach(_TARGET_DEP ${_TARGET_DEPS})
      _iree_transitive_dependencies_helper(${_TARGET_DEP} _RESULT)
    endforeach(_TARGET_DEP)
  endif()

  # Propagate the augmented list up to the parent scope.
  set(${TRANSITIVE_DEPS} "${_RESULT}" PARENT_SCOPE)
endfunction()

# Sets target_link_libraries() on all registered py extensions.
# This must be called after all libraries have been declared.
function(iree_complete_py_extension_link_options)
  get_property(_NAMES GLOBAL PROPERTY _IREE_PY_EXTENSION_NAMES)

  foreach(_NAME ${_NAMES})
    get_target_property(_DIRECT_DEPS ${_NAME} DIRECT_DEPS)

    # List all dependencies, including transitive dependencies, then split the
    # dependency list into one for whole archive (ALWAYSLINK) and one for
    # standard linking (which only links in symbols that are directly used).
    _iree_transitive_dependencies("${_DIRECT_DEPS}" _TRANSITIVE_DEPS)
    set(_ALWAYS_LINK_DEPS "")
    set(_STANDARD_DEPS "")
    foreach(_DEP ${_TRANSITIVE_DEPS})
      # Check if _DEP is a library with the ALWAYSLINK property set.
      set(_DEP_IS_ALWAYSLINK OFF)
      if (TARGET ${_DEP})
        get_target_property(_DEP_TYPE ${_DEP} TYPE)
        if(${_DEP_TYPE} STREQUAL "INTERFACE_LIBRARY")
          # Can't be ALWAYSLINK since it's an INTERFACE library.
          # We also can't even query for the property, since it isn't allowlisted.
        else()
          get_target_property(_DEP_IS_ALWAYSLINK ${_DEP} ALWAYSLINK)
        endif()
      endif()

      # Append to the corresponding list of deps.
      if(_DEP_IS_ALWAYSLINK)
        list(APPEND _ALWAYS_LINK_DEPS ${_DEP})

        # For MSVC, also add a `-WHOLEARCHIVE:` version of the dep.
        # CMake treats -WHOLEARCHIVE[:lib] as a link flag and will not actually
        # try to link the library in, so we need the flag *and* the dependency.
        # For macOS, also add a `-Wl,-force_load` version of the dep.
        if(MSVC)
          get_target_property(_ALIASED_TARGET ${_DEP} ALIASED_TARGET)
          if (_ALIASED_TARGET)
            list(APPEND _ALWAYS_LINK_DEPS "-WHOLEARCHIVE:${_ALIASED_TARGET}")
          else()
            list(APPEND _ALWAYS_LINK_DEPS "-WHOLEARCHIVE:${_DEP}")
          endif()
        elseif(APPLE)
          get_target_property(_ALIASED_TARGET ${_DEP} ALIASED_TARGET)
          if (_ALIASED_TARGET)
            list(APPEND _ALWAYS_LINK_DEPS "-Wl,-force_load $<TARGET_FILE:${_ALIASED_TARGET}>")
          else()
            list(APPEND _ALWAYS_LINK_DEPS "-Wl,-force_load $<TARGET_FILE:${_DEP}>")
          endif()
        endif()
      else()
        list(APPEND _STANDARD_DEPS ${_DEP})
      endif()
    endforeach(_DEP)

    # Call into target_link_libraries with the lists of deps.
    if(MSVC OR APPLE)
      target_link_libraries(${_NAME}
        PUBLIC
          ${_ALWAYS_LINK_DEPS}
          ${_STANDARD_DEPS}
        PRIVATE
          ${_RULE_LINKOPTS}
          ${PYTHON_LIBRARY}
      )
    else()
      target_link_libraries(${_NAME}
        PUBLIC
          "-Wl,--whole-archive"
          ${_ALWAYS_LINK_DEPS}
          "-Wl,--no-whole-archive"
          ${_STANDARD_DEPS}
        PRIVATE
          ${_RULE_LINKOPTS}
          ${PYTHON_LIBRARY}
      )
    endif()
  endforeach(_NAME)
endfunction()
