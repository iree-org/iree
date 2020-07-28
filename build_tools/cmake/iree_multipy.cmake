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

###############################################################################
# Configuration
###############################################################################

function(iree_multipy_configure)
  # Configure the defaults.
  # Note that this is using the pybind11 configuration vars, which creates
  # a fragile dependency. It would be better to derive these locally.
  if(PYTHONLIBS_FOUND)
    set(IREE_MULTIPY_DEFAULT_EXECUTABLE "${PYTHON_EXECUTABLE}" CACHE INTERNAL "Python executable" )
    set(IREE_MULTIPY_DEFAULT_INCLUDE_DIRS "${PYTHON_INCLUDE_DIRS}" CACHE INTERNAL "Python include dirs" )
    set(IREE_MULTIPY_DEFAULT_LIBRARIES "${PYTHON_LIBRARIES}" CACHE INTERNAL "Python libraries")
    set(IREE_MULTIPY_DEFAULT_PREFIX "${PYTHON_MODULE_PREFIX}" CACHE INTERNAL "Python module prefix")
    set(IREE_MULTIPY_DEFAULT_SUFFIX "${PYTHON_MODULE_SUFFIX}" CACHE INTERNAL "Python module suffix")
    set(IREE_MULTIPY_DEFAULT_EXTENSION "${PYTHON_MODULE_EXTENSION}" CACHE INTERNAL "Python module extension")
  endif()

  if(IREE_MULTIPY_VERSIONS)
    set(IREE_MULTIPY_VERSIONS_EFFECTIVE "${IREE_MULTIPY_VERSIONS}" CACHE INTERNAL "Python extension versions")
  else()
    message(STATUS "Multi-python extension versions not found: using defaults")
    set(IREE_MULTIPY_VERSIONS_EFFECTIVE "DEFAULT" CACHE INTERNAL "Python extension versions")
  endif()

  # Report the multipy config.
  message(STATUS "Multipy extension versions: ${IREE_MULTIPY_VERSIONS_EFFECTIVE}")
  foreach(V ${IREE_MULTIPY_VERSIONS_EFFECTIVE})
    message(STATUS "  - Multipy version ${V}")
    message(STATUS "    : EXECUTABLE = ${IREE_MULTIPY_${V}_EXECUTABLE}")
    message(STATUS "    : INCLUDE_DIRS = ${IREE_MULTIPY_${V}_INCLUDE_DIRS}")
    message(STATUS "    : LIBRARIES = ${IREE_MULTIPY_${V}_LIBRARIES}")
    message(STATUS "    : PREFIX = ${IREE_MULTIPY_${V}_PREFIX}")
    message(STATUS "    : SUFFIX = ${IREE_MULTIPY_${V}_SUFFIX}")
    message(STATUS "    : EXTENSION = ${IREE_MULTIPY_${V}_EXTENSION}")

    # Check for required settings.
    if(NOT IREE_MULTIPY_${V}_INCLUDE_DIRS)
      message(FATAL " MULTIPY version ${V}: No IREE_MULTIPY_${VER}_EXECUTABLE var")
    endif()
    if(NOT IREE_MULTIPY_${V}_INCLUDE_DIRS)
      message(FATAL " MULTIPY version ${V}: No IREE_MULTIPY_${VER}_INCLUDE_DIRS var")
    endif()
    if(NOT IREE_MULTIPY_${V}_EXTENSION)
      message(FATAL " MULTIPY version ${V}: No IREE_MULTIPY_${VER}_EXTENSION var")
    endif()
  endforeach()
endfunction()

macro(_setup_iree_pyext_names)
  iree_package_ns(_PACKAGE_NS)
  # Replace dependencies passed by ::name with ::iree::package::name
  list(TRANSFORM ARG_DEPS REPLACE "^::" "${_PACKAGE_NS}::")
  list(TRANSFORM ARG_PYEXT_DEPS REPLACE "^::" "${_PACKAGE_NS}::")
  # Prefix the library with the package name, so we get: iree_package_name.
  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${ARG_NAME}")
endmacro()

macro(_alias_iree_pyext_library declared_name version target)
  # Alias the iree_package_name library to iree::package::name.
  # This lets us more clearly map to Bazel and makes it possible to
  # disambiguate the underscores in paths vs. the separators.
  add_library(${_PACKAGE_NS}::${ARG_NAME}__${version} ALIAS ${target})
  iree_package_dir(_PACKAGE_DIR)
endmacro()

###############################################################################
# Main user rules
###############################################################################

# iree_pyext_module()
#
# Builds a native python module (.so/.dylib/.pyd).
#
# Parameters:
# NAME: name of target
# MODULE_NAME: Base-name of the module.
# SRCS: List of source files for the library
# COPTS: C options
# DEPS: List of other targets the test python libraries require
# PYEXT_DEPS: List of deps of extensions built with iree_pyext_(library|module)
function(iree_pyext_module)
  cmake_parse_arguments(ARG
    ""
    "NAME;MODULE_NAME;UNIX_LINKER_SCRIPT"
    "SRCS;COPTS;DEPS;PYEXT_DEPS"
    ${ARGN})
  _setup_iree_pyext_names()

  add_custom_target(${_NAME})

  foreach(V ${IREE_MULTIPY_VERSIONS_EFFECTIVE})
    set(VER_NAME "${_NAME}__${V}")

    # If configured to link against libraries, build in SHARED mode (which
    # disallows undefined symbols). Otherwise, build in MODULE mode, which
    # does not enforce that. This should naturally do the right thing on
    # each platform based on whether configured with a list of libraries to
    # link or not.
    set(LIBRARY_TYPE MODULE)
    if(IREE_MULTIPY_${V}_LIBRARIES)
      set(LIBRARY_TYPE SHARED)
    endif()

    add_library(${VER_NAME} ${LIBRARY_TYPE} ${ARG_SRCS})
    add_dependencies(${_NAME} ${VER_NAME})
    set_target_properties(
      ${VER_NAME} PROPERTIES
        OUTPUT_NAME "${ARG_MODULE_NAME}"
        PREFIX "${IREE_MULTIPY_${V}_PREFIX}"
        SUFFIX "${IREE_MULTIPY_${V}_SUFFIX}${IREE_MULTIPY_${V}_EXTENSION}"
    )

    # Link flags.
    if(UNIX AND NOT APPLE)  # Apple does not support linker scripts.
      if(ARG_UNIX_LINKER_SCRIPT)
        set_target_properties(${VER_NAME} PROPERTIES LINK_FLAGS
          "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/${ARG_UNIX_LINKER_SCRIPT}")
      endif()
    endif()

    iree_pyext_pybind11_options(${VER_NAME})
    target_include_directories(${VER_NAME}
      PUBLIC
        "${IREE_MULTIPY_${V}_INCLUDE_DIRS}"
        "$<BUILD_INTERFACE:${IREE_COMMON_INCLUDE_DIRS}>"
    )
    target_link_libraries(${VER_NAME}
      PRIVATE
        ${IREE_DEFAULT_LINKOPTS}
        ${IREE_MULTIPY_${V}_LIBRARIES}
    )
    target_compile_options(${VER_NAME}
      INTERFACE
        ${IREE_DEFAULT_COPTS}
      PRIVATE
        ${ARG_COPTS}
    )

    # Defer computing transitive dependencies and calling target_link_libraries()
    # until all libraries have been declared.
    # Track target and deps, use in iree_complete_py_extension_link_options() later.
    # See iree_complete_py_extension_link_options() in iree_py_extension.cmake
    # TODO: Move that implementation here.
    set(TRANSFORMED_PYEXT_DEPS "${ARG_PYEXT_DEPS}")
    list(TRANSFORM TRANSFORMED_PYEXT_DEPS APPEND "__${V}")
    set_property(GLOBAL APPEND PROPERTY _IREE_PY_EXTENSION_NAMES "${VER_NAME}")
    set_property(TARGET ${VER_NAME} PROPERTY DIRECT_DEPS ${ARG_DEPS} ${TRANSFORMED_PYEXT_DEPS})
    _alias_iree_pyext_library("${ARG_NAME}" "${V}" ${VER_NAME})
  endforeach()
endfunction()

# iree_pyext_library()
#
# Builds a C++ library to be included in an iree_pyext_module.
#
# Parameters:
# NAME: name of target
# SRCS: List of source files for the library
# COPTS: C options
# DEPS: List of other targets the test python libraries require
# PYEXT_DEPS: List of deps of extensions built with iree_pyext_(library|module)
function(iree_pyext_library)
  cmake_parse_arguments(ARG
    ""
    "NAME"
    "SRCS;COPTS;DEPS;PYEXT_DEPS"
    ${ARGN})
  _setup_iree_pyext_names()

  foreach(V ${IREE_MULTIPY_VERSIONS_EFFECTIVE})
    set(VER_NAME "${_NAME}__${V}")
    add_library(${VER_NAME} STATIC ${ARG_SRCS})
    iree_pyext_pybind11_options(${VER_NAME})
    target_include_directories(${VER_NAME}
      PUBLIC
        "${IREE_MULTIPY_${V}_INCLUDE_DIRS}"
        "$<BUILD_INTERFACE:${IREE_COMMON_INCLUDE_DIRS}>"
    )
    set(TRANSFORMED_PYEXT_DEPS "${ARG_PYEXT_DEPS}")
    list(TRANSFORM TRANSFORMED_PYEXT_DEPS APPEND "__${V}")
    target_link_libraries(${VER_NAME}
      PUBLIC
        ${ARG_DEPS}
        ${TRANSFORMED_PYEXT_DEPS}
      PRIVATE
        ${IREE_DEFAULT_LINKOPTS}
    )
    target_compile_options(${VER_NAME}
      INTERFACE
        ${IREE_DEFAULT_COPTS}
      PRIVATE
        ${ARG_COPTS}
    )
    _alias_iree_pyext_library("${ARG_NAME}" "${V}" ${VER_NAME})
  endforeach()
endfunction()

# iree_py_library()
#
# CMake function to imitate Bazel's iree_py_library rule.
#
# Parameters:
# NAME: name of target
# SRCS: List of source files for the library
# DEPS: List of other targets the test python libraries require
# PYEXT_DEPS: List of deps of extensions built with iree_pyext_module
function(iree_py_library)
  cmake_parse_arguments(
    ARG
    ""
    "NAME"
    "SRCS;DEPS;PYEXT_DEPS"
    ${ARGN}
  )

  iree_package_ns(_PACKAGE_NS)
  # Replace dependencies passed by ::name with ::iree::package::name
  list(TRANSFORM ARG_DEPS REPLACE "^::" "${_PACKAGE_NS}::")

  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${ARG_NAME}")

  # Add path to each source file
  list(TRANSFORM ARG_SRCS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

  add_custom_target(${_NAME} ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${ARG_SRCS} "${CMAKE_CURRENT_BINARY_DIR}/"
    DEPENDS ${ARG_DEPS}
  )

  # Add PYEXT_DEPS.
  if(${ARG_PYEXT_DEPS})
    foreach(V ${IREE_MULTIPY_VERSIONS_EFFECTIVE})
      list(TRANSFORM ARG_PYEXT_DEPS APPEND "__${V}")
      add_dependencies(${_NAME} ${ARG_PYEXT_DEPS})
    endforeach()
  endif()
endfunction()

function(iree_pyext_pybind11_options name)
  target_include_directories(${name}
    PRIVATE
      ${PYBIND11_INCLUDE_DIR}
  )
  target_compile_options(${name}
    PRIVATE
      $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
        -frtti -fexceptions
        # Noisy pybind warnings
        -Wno-unused-value
        -Wno-covered-switch-default
      >
      $<$<CXX_COMPILER_ID:MSVC>:
        # Enable RTTI and exceptions.
        /EHsc /GR>
  )
  set_target_properties(
    ${name} PROPERTIES CXX_VISIBILITY_PRESET "hidden")
endfunction()

# iree_py_test()
#
# CMake function to imitate Bazel's iree_py_test rule.
#
# Parameters:
# NAME: name of test
# SRCS: List of source file
# DEPS: List of deps the test requires
# LABELS: Additional labels to apply to the test. The package path is added
#     automatically.

function(iree_py_test)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "SRCS;DEPS;LABELS"
    ${ARGN}
  )

  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  iree_package_ns(_PACKAGE_NS)
  string(REPLACE "::" "/" _PACKAGE_PATH ${_PACKAGE_NS})
  set(_NAME_PATH "${_PACKAGE_PATH}:${_RULE_NAME}")
  list(APPEND _RULE_LABELS "${_PACKAGE_PATH}")

  foreach(V ${IREE_MULTIPY_VERSIONS_EFFECTIVE})
    set(VER_NAME "${_NAME_PATH}__${V}")
    add_test(
      NAME ${VER_NAME}
      COMMAND
        "${CMAKE_SOURCE_DIR}/build_tools/cmake/run_test.${IREE_HOST_SCRIPT_EXT}"
        "${IREE_MULTIPY_${V}_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/${_RULE_SRCS}"
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    set_property(TEST ${VER_NAME} PROPERTY LABELS "${_RULE_LABELS}")
    set_property(TEST ${VER_NAME} PROPERTY ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/bindings/python:$ENV{PYTHONPATH};TEST_TMPDIR=${_NAME}_${V}_test_tmpdir")
    # TODO(marbre): Find out how to add deps to tests.
    #               Similar to _RULE_DATA in iree_lit_test().
  endforeach()
endfunction()

###############################################################################
# Always-link/transitive dependency management
###############################################################################

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
      )
    endif()
  endforeach(_NAME)
endfunction()
