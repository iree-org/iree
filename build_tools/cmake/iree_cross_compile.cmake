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

include(iree_macros)

# iree_create_configuration
#
# Creates custom commands and targets for an IREE configuration. An IREE
# configuration means a new IREE CMake invocation with its own set of
# parameters.
#
# This function defines a custom target, `iree_configure_${CONFIG_NAME}`,
# to drive the generation of a new IREE configuration's `CMakeCache.txt`
# file. Callers can then depend on either the `CMakeCache.txt` file or the
# `iree_configure_${CONFIG_NAME}` target to make sure the configuration
# is invoked as a dependency.
#
# This function is typically useful when cross-compiling towards another
# architecture. For example, when cross-compiling towards Android, we need
# to have certain tools first compiled on the host so that we can use them
# to programmatically generate some source code to be compiled together
# with other checked-in source code. Those host tools will be generated
# by another CMake invocation configured by this function.
#
# Supported CMake options:
# - IREE_<CONFIG_NAME>_BINARY_ROOT: the root directory for containing IREE build
#   artifacts for the given `CONFIG_NAME`. If not specified in caller, this is
#   set to a directory named as `CONFIG_NAME` under the current CMake binary
#   directory.
# - IREE_<CONFIG_NAME>_C_COMPILER: C compiler for the given `CONFIG_NAME`.
#   This must be defined by the caller.
# - IREE_<CONFIG_NAME>_CXX_COMPILER: C++ compiler for the given `CONFIG_NAME`.
#   This must be defined by the caller.
# - IREE_<CONFIG_NAME>_<option>: switch for the given `option` specifically for
#   `CONFIG_NAME`. If missing, default to OFF for bool options; default to
#   IREE_<option> for non-bool variables.
function(iree_create_configuration CONFIG_NAME)
  # Set IREE_${CONFIG_NAME}_BINARY_ROOT if missing.
  if(NOT DEFINED IREE_${CONFIG_NAME}_BINARY_ROOT)
    set(IREE_${CONFIG_NAME}_BINARY_ROOT "${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}")
    set(IREE_${CONFIG_NAME}_BINARY_ROOT ${IREE_${CONFIG_NAME}_BINARY_ROOT} PARENT_SCOPE)
    message(STATUS "Setting ${CONFIG_NAME} build directory to ${IREE_${CONFIG_NAME}_BINARY_ROOT}")
  endif()

  set(_CONFIG_BINARY_ROOT ${IREE_${CONFIG_NAME}_BINARY_ROOT})

  set(_CONFIG_C_COMPILER ${IREE_${CONFIG_NAME}_C_COMPILER})
  set(_CONFIG_CXX_COMPILER ${IREE_${CONFIG_NAME}_CXX_COMPILER})

  # Check the compilers are specified in the caller.
  if("${_CONFIG_C_COMPILER}" STREQUAL "")
    message(FATAL_ERROR "Must define IREE_${CONFIG_NAME}_C_COMPILER for \"${CONFIG_NAME}\" configuration build")
  endif()
  if("${_CONFIG_CXX_COMPILER}" STREQUAL "")
    message(FATAL_ERROR "Must define IREE_${CONFIG_NAME}_CXX_COMPILER for \"${CONFIG_NAME}\" configuration build")
  endif()

  add_custom_command(OUTPUT ${_CONFIG_BINARY_ROOT}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_CONFIG_BINARY_ROOT}
    COMMENT "Creating ${_CONFIG_BINARY_ROOT}...")

  # Give it a custom target so we can drive the generation manually
  # when useful.
  add_custom_target(iree_prepare_${CONFIG_NAME}_dir DEPENDS ${_CONFIG_BINARY_ROOT})

  # LINT.IfChange(iree_cross_compile_options)
  iree_to_bool(_CONFIG_ENABLE_RUNTIME_TRACING "${IREE_${CONFIG_NAME}_ENABLE_RUNTIME_TRACING}")
  iree_to_bool(_CONFIG_ENABLE_MLIR "${IREE_${CONFIG_NAME}_ENABLE_MLIR}")
  iree_to_bool(_CONFIG_ENABLE_EMITC "${IREE_${CONFIG_NAME}_ENABLE_EMITC}")

  iree_to_bool(_CONFIG_BUILD_COMPILER "${IREE_${CONFIG_NAME}_BUILD_COMPILER}")
  iree_to_bool(_CONFIG_BUILD_TESTS "${IREE_${CONFIG_NAME}_BUILD_TESTS}")
  iree_to_bool(_CONFIG_BUILD_DOCS "${IREE_${CONFIG_NAME}_BUILD_DOCS}")
  iree_to_bool(_CONFIG_BUILD_SAMPLES "${IREE_${CONFIG_NAME}_BUILD_SAMPLES}")
  iree_to_bool(_CONFIG_BUILD_DEBUGGER "${IREE_${CONFIG_NAME}_BUILD_DEBUGGER}")
  iree_to_bool(_CONFIG_BUILD_PYTHON_BINDINGS "${IREE_${CONFIG_NAME}_BUILD_PYTHON_BINDINGS}")
  iree_to_bool(_CONFIG_BUILD_EXPERIMENTAL "${IREE_${CONFIG_NAME}_BUILD_EXPERIMENTAL}")

  # Escape semicolons in the targets list so that CMake doesn't expand them to
  # spaces.
  string(REPLACE ";" "$<SEMICOLON>" _CONFIG_HAL_DRIVERS_TO_BUILD "${IREE_HAL_DRIVERS_TO_BUILD}")
  string(REPLACE ";" "$<SEMICOLON>" _CONFIG_TARGET_BACKENDS_TO_BUILD "${IREE_TARGET_BACKENDS_TO_BUILD}")
  # LINT.ThenChange(
  #   https://github.com/google/iree/tree/main/CMakeLists.txt:iree_options,
  #   https://github.com/google/iree/tree/main/build_tools/cmake/iree_cross_compile.cmake:iree_cross_compile_invoke
  # )

  message(STATUS "C compiler for ${CONFIG_NAME} build: ${_CONFIG_C_COMPILER}")
  message(STATUS "C++ compiler for ${CONFIG_NAME} build: ${_CONFIG_CXX_COMPILER}")

  add_custom_command(OUTPUT ${IREE_${CONFIG_NAME}_BINARY_ROOT}/CMakeCache.txt
    COMMAND "${CMAKE_COMMAND}" "${PROJECT_SOURCE_DIR}" -G "${CMAKE_GENERATOR}"
        -DCMAKE_MAKE_PROGRAM="${CMAKE_MAKE_PROGRAM}"
        -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
        -DCMAKE_C_COMPILER="${_CONFIG_C_COMPILER}"
        -DCMAKE_CXX_COMPILER="${_CONFIG_CXX_COMPILER}"
        # LINT.IfChange(iree_cross_compile_invoke)
        -DIREE_ENABLE_RUNTIME_TRACING=${_CONFIG_ENABLE_RUNTIME_TRACING}
        -DIREE_ENABLE_MLIR=${_CONFIG_ENABLE_MLIR}
        -DIREE_ENABLE_EMITC=${_CONFIG_ENABLE_EMITC}
        -DIREE_BUILD_COMPILER=${_CONFIG_BUILD_COMPILER}
        -DIREE_BUILD_TESTS=${_CONFIG_BUILD_TESTS}
        -DIREE_BUILD_DOCS=${_CONFIG_BUILD_DOCS}
        -DIREE_BUILD_SAMPLES=${_CONFIG_BUILD_SAMPLES}
        -DIREE_BUILD_DEBUGGER=${_CONFIG_BUILD_DEBUGGER}
        -DIREE_BUILD_PYTHON_BINDINGS=${_CONFIG_BUILD_PYTHON_BINDINGS}
        -DIREE_BUILD_EXPERIMENTAL=${_CONFIG_BUILD_EXPERIMENTAL}
        # LINT.ThenChange(
        #   https://github.com/google/iree/tree/main/CMakeLists.txt:iree_options,
        #   https://github.com/google/iree/tree/main/build_tools/cmake/iree_cross_compile.cmake:iree_cross_compile_options,
        # )
        -DIREE_HAL_DRIVERS_TO_BUILD="${_CONFIG_HAL_DRIVERS_TO_BUILD}"
        -DIREE_TARGET_BACKENDS_TO_BUILD="${_CONFIG_TARGET_BACKENDS_TO_BUILD}"
    WORKING_DIRECTORY ${_CONFIG_BINARY_ROOT}
    DEPENDS iree_prepare_${CONFIG_NAME}_dir
    COMMENT "Configuring IREE for ${CONFIG_NAME} build...")

  add_custom_target(iree_configure_${CONFIG_NAME} DEPENDS ${_CONFIG_BINARY_ROOT}/CMakeCache.txt)
endfunction()

# iree_get_build_command
#
# Gets the CMake build command for the given `EXECUTABLE`.
#
# Parameters:
# EXECUTABLE: the executable to build.
# BINDIR: root binary directory containing CMakeCache.txt.
# CMDVAR: variable name for receiving the build command.
function(iree_get_build_command EXECUTABLE)
  cmake_parse_arguments(_RULE "" "BINDIR;CMDVAR;CONFIG" "" ${ARGN})
  if(NOT _RULE_CONFIG)
    set(_RULE_CONFIG "$<CONFIG>")
  endif()
  if (CMAKE_GENERATOR MATCHES "Make")
    # Use special command for Makefiles to support parallelism.
    set(${_RULE_CMDVAR}
        "$(MAKE)" "-C" "${_RULE_BINDIR}" "${EXECUTABLE}" PARENT_SCOPE)
  else()
    set(${_RULE_CMDVAR}
        "${CMAKE_COMMAND}" --build ${_RULE_BINDIR}
                           --target ${EXECUTABLE}${IREE_HOST_EXECUTABLE_SUFFIX}
                           --config ${_RULE_CONFIG} PARENT_SCOPE)
  endif()
endfunction()

# iree_host_install
#
# Defines custom commands and targets for installing the given `EXECUTABLE`
# under host configuration. The custom target for install will be named as
# `iree_host_install_${EXECUTABLE}`.
#
# Precondition:
# iree_create_configuration(HOST) is invoked previously.
#
# Parameters:
# EXECUTABLE: the executable to install.
# COMPONENT: installation component; used for filtering installation targets.
# PREFIX: the root installation path prefix.
# DEPENDS: addtional dependencies for the installation.
function(iree_host_install EXECUTABLE)
  cmake_parse_arguments(_RULE "" "COMPONENT;PREFIX" "DEPENDS" ${ARGN})
  if(_RULE_COMPONENT)
    set(_COMPONENT_OPTION -DCMAKE_INSTALL_COMPONENT="${_RULE_COMPONENT}")
  endif()
  if(_RULE_PREFIX)
    set(_PREFIX_OPTION -DCMAKE_INSTALL_PREFIX="${_RULE_PREFIX}")
  endif()

  iree_get_executable_path(_OUTPUT_PATH ${EXECUTABLE})

  add_custom_command(
    OUTPUT ${_OUTPUT_PATH}
    DEPENDS ${_RULE_DEPENDS}
    COMMAND "${CMAKE_COMMAND}" ${_COMPONENT_OPTION} ${_PREFIX_OPTION}
            -P "${IREE_HOST_BINARY_ROOT}/cmake_install.cmake"
    USES_TERMINAL)

  # Give it a custom target so we can drive the generation manually
  # when useful.
  add_custom_target(iree_host_install_${EXECUTABLE} DEPENDS ${_OUTPUT_PATH})
endfunction()

# iree_declare_host_excutable
#
# Generates custom commands and targets for building and installing a tool on
# host for cross-compilation.
#
# Precondition:
# iree_create_configuration(HOST) is invoked previously.
#
# Parameters:
# EXECUTABLE: the executable to build on host.
# BUILDONLY: only generates commands for building the target.
# DEPENDS: any additional dependencies for the target.
function(iree_declare_host_excutable EXECUTABLE)
  cmake_parse_arguments(_RULE "BUILDONLY" "" "DEPENDS" ${ARGN})

  iree_get_executable_path(_OUTPUT_PATH ${EXECUTABLE})

  iree_get_build_command(${EXECUTABLE}
    BINDIR ${IREE_HOST_BINARY_ROOT}
    CMDVAR build_cmd)

  add_custom_target(iree_host_build_${EXECUTABLE}
                    COMMAND ${build_cmd}
                    DEPENDS iree_configure_HOST ${_RULE_DEPENDS}
                    WORKING_DIRECTORY "${IREE_HOST_BINARY_ROOT}"
                    COMMENT "Building host ${EXECUTABLE}..."
                    USES_TERMINAL)

  if(_RULE_BUILDONLY)
    return()
  endif()

  iree_host_install(${EXECUTABLE}
                    COMPONENT ${EXECUTABLE}
                    PREFIX ${IREE_HOST_BINARY_ROOT}
                    DEPENDS iree_host_build_${EXECUTABLE})

  # Note that this is not enabled when BUILDONLY so we can define
  # iree_host_${EXECUTABLE} to point to another installation path to
  # allow flexibility.
  add_custom_target(iree_host_${EXECUTABLE} DEPENDS "${_OUTPUT_PATH}")
endfunction()
