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

function(iree_multipy_configure)
  # Configure the defaults.
  # Note that this is using the pybind11 configuration vars, which creates
  # a fragile dependency. It would be better to derive these locally.
  set(IREE_MULTIPY_DEFAULT_INCLUDE_DIRS "${PYTHON_INCLUDE_DIRS}" CACHE INTERNAL "Python include dirs" )
  set(IREE_MULTIPY_DEFAULT_LIBRARIES "${PYTHON_LIBRARIES}" CACHE INTERNAL "Python libraries")
  set(IREE_MULTIPY_DEFAULT_PREFIX "${PYTHON_MODULE_PREFIX}" CACHE INTERNAL "Python module prefix")
  set(IREE_MULTIPY_DEFAULT_SUFFIX "${PYTHON_MODULE_SUFFIX}" CACHE INTERNAL "Python module suffix")
  set(IREE_MULTIPY_DEFAULT_EXTENSION "${PYTHON_MODULE_EXTENSION}" CACHE INTERNAL "Python module extension")

  if(IREE_MULTIPY_VERSIONS)
    set(IREE_MULTIPY_VERSIONS_EFFECTIVE "${IREE_MULTIPY_VERSIONS}")
  else()
    message(STATUS "Multi-python extension versions not found: using defaults")
    set(IREE_MULTIPY_VERSIONS_EFFECTIVE "DEFAULT" CACHE INTERNAL "Python extension versions")
  endif()

  # Report the multipy config.
  message(STATUS "Multipy extension versions: ${IREE_MULTIPY_VERSIONS_EFFECTIVE}")
  foreach(V ${IREE_MULTIPY_VERSIONS_EFFECTIVE})
    message(STATUS "  - Multipy version ${V}")
    message(STATUS "    : INCLUDE_DIRS = ${IREE_MULTIPY_${V}_INCLUDE_DIRS}")
    message(STATUS "    : LIBRARIES = ${IREE_MULTIPY_${V}_LIBRARIES}")
    message(STATUS "    : PREFIX = ${IREE_MULTIPY_${V}_PREFIX}")
    message(STATUS "    : SUFFIX = ${IREE_MULTIPY_${V}_SUFFIX}")
    message(STATUS "    : EXTENSION = ${IREE_MULTIPY_${V}_EXTENSION}")

    # Only INCLUDE_DIRS and EXTENSION are needed for all configs.
    if(NOT IREE_MULTIPY_${V}_INCLUDE_DIRS)
      message(FATAL "MULTIPY config ${V}: No IREE_MULTIPY_{VER}_INCLUDE_DIRS var")
    endif()
    if(NOT IREE_MULTIPY_${V}_EXTENSION)
      message(FATAL "MULTIPY config ${V}: No IREE_MULTIPY_{VER}_EXTENSION var")
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
    "NAME;MODULE_NAME"
    "SRCS;COPTS;DEPS;PYEXT_DEPS"
    ${ARGN})
  _setup_iree_pyext_names()

  add_custom_target(${_NAME})

  foreach(V ${IREE_MULTIPY_VERSIONS_EFFECTIVE})
    set(VER_NAME "${_NAME}__${V}")
    add_library(${VER_NAME} SHARED ${ARG_SRCS})
    add_dependencies(${_NAME} ${VER_NAME})
    set_target_properties(
      ${VER_NAME} PROPERTIES
        OUTPUT_NAME "${ARG_MODULE_NAME}"
        PREFIX "${IREE_MULTIPY_${V}_PREFIX}"
        SUFFIX "${IREE_MULTIPY_${V}_SUFFIX}${IREE_MULTIPY_${V}_EXTENSION}"
    )

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
    list(TRANSFORM ARG_PYEXT_DEPS APPEND "__${V}")
    set_property(GLOBAL APPEND PROPERTY _IREE_PY_EXTENSION_NAMES "${VER_NAME}")
    set_property(TARGET ${VER_NAME} PROPERTY DIRECT_DEPS ${ARG_DEPS} ${ARG_PYEXT_DEPS})

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
    list(TRANSFORM ARG_PYEXT_DEPS APPEND "__${V}")
    target_link_libraries(${VER_NAME}
      PUBLIC
        ${ARG_DEPS}
        ${ARG_PYEXT_DEPS}
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
  list(TRANSFORM _RULE_SRCS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

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
