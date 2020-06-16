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
# This function defines two custom target, `iree_prepare_${config_name}_dir`
# and `iree_configure_${config_name}`, to drive the creation of a directory
# for hosting the new IREE configuration and its corresponding `CMakeCache.txt`
# file, respectively. Callers can then depend on either the file or the
# target to make sure the configuration is invoked as a dependency.
#
# This function is typically useful when cross-compiling towards another
# architecture. For example, when cross-compiling towards Android, we need
# to have certain tools first compiled on the host so that we can use them
# to programmatically generate some source code to be compiled together
# with other checked-in source code. Those host tools will be generated
# by another CMake invocation configured by this function.
#
# Supported CMake options:
# - IREE_<config_name>_BINARY_ROOT: the root directory for containing IREE build
#   artifacts for the given `config_name`. If not specified in caller, this is
#   set to a directory named as `config_name` under the current CMake binary
#   directory.
# - IREE_<config_name>_C_COMPILER: C compiler for the given `config_name`.
#   This must be defined by the caller.
# - IREE_<config_name>_CXX_COMPILER: C++ compiler for the given `config_name`.
#   This must be defined by the caller.
# - IREE_<config_name>_<option>: switch for the given `option` specifically for
#   `config_name`. If missing, default to OFF for bool options; default to
#   IREE_<option> for non-bool variables.
function(iree_create_configuration config_name)
  # Set IREE_${config_name}_BINARY_ROOT if missing.
  if(NOT DEFINED IREE_${config_name}_BINARY_ROOT)
    set(IREE_${config_name}_BINARY_ROOT "${CMAKE_CURRENT_BINARY_DIR}/${config_name}")
    set(IREE_${config_name}_BINARY_ROOT ${IREE_${config_name}_BINARY_ROOT} PARENT_SCOPE)
    message(STATUS "Setting ${config_name} build directory to ${IREE_${config_name}_BINARY_ROOT}")
  endif()

  set(_CONFIG_BINARY_ROOT ${IREE_${config_name}_BINARY_ROOT})

  set(_CONFIG_C_COMPILER ${IREE_${config_name}_C_COMPILER})
  set(_CONFIG_CXX_COMPILER ${IREE_${config_name}_CXX_COMPILER})

  # Check the compilers are specified in the caller.
  if("${_CONFIG_C_COMPILER}" STREQUAL "")
    message(FATAL_ERROR "Must define IREE_${config_name}_C_COMPILER for \"${config_name}\" configuration build")
  endif()
  if("${_CONFIG_CXX_COMPILER}" STREQUAL "")
    message(FATAL_ERROR "Must define IREE_${config_name}_CXX_COMPILER for \"${config_name}\" configuration build")
  endif()

  add_custom_command(OUTPUT ${_CONFIG_BINARY_ROOT}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_CONFIG_BINARY_ROOT}
    COMMENT "Creating ${_CONFIG_BINARY_ROOT}...")

  # Give it a custom target so we can drive the generation manually
  # when useful.
  add_custom_target(iree_prepare_${config_name}_dir DEPENDS ${_CONFIG_BINARY_ROOT})

  # LINT.IfChange(iree_cross_compile_options)
  iree_to_bool(_CONFIG_ENABLE_RUNTIME_TRACING "${IREE_${config_name}_ENABLE_RUNTIME_TRACING}")
  iree_to_bool(_CONFIG_ENABLE_MLIR "${IREE_${config_name}_ENABLE_MLIR}")
  iree_to_bool(_CONFIG_ENABLE_EMITC "${IREE_${config_name}_ENABLE_EMITC}")

  iree_to_bool(_CONFIG_BUILD_COMPILER "${IREE_${config_name}_BUILD_COMPILER}")
  iree_to_bool(_CONFIG_BUILD_TESTS "${IREE_${config_name}_BUILD_TESTS}")
  iree_to_bool(_CONFIG_BUILD_DOCS "${IREE_${config_name}_BUILD_DOCS}")
  iree_to_bool(_CONFIG_BUILD_SAMPLES "${IREE_${config_name}_BUILD_SAMPLES}")
  iree_to_bool(_CONFIG_BUILD_DEBUGGER "${IREE_${config_name}_BUILD_DEBUGGER}")
  iree_to_bool(_CONFIG_BUILD_PYTHON_BINDINGS "${IREE_${config_name}_BUILD_PYTHON_BINDINGS}")
  iree_to_bool(_CONFIG_BUILD_EXPERIMENTAL "${IREE_${config_name}_BUILD_EXPERIMENTAL}")

  # Escape semicolons in the targets list so that CMake doesn't expand them to
  # spaces.
  string(REPLACE ";" "$<SEMICOLON>" _CONFIG_HAL_DRIVERS_TO_BUILD "${IREE_HAL_DRIVERS_TO_BUILD}")
  string(REPLACE ";" "$<SEMICOLON>" _CONFIG_TARGET_BACKENDS_TO_BUILD "${IREE_TARGET_BACKENDS_TO_BUILD}")
  # LINT.ThenChange(https://github.com/google/iree/tree/master/CMakeLists.txt:iree_options)

  message(STATUS "C compiler for ${config_name} build: ${_CONFIG_C_COMPILER}")
  message(STATUS "C++ compiler for ${config_name} build: ${_CONFIG_CXX_COMPILER}")

  add_custom_command(OUTPUT ${IREE_${config_name}_BINARY_ROOT}/CMakeCache.txt
    COMMAND "${CMAKE_COMMAND}" "${PROJECT_SOURCE_DIR}" -G "${CMAKE_GENERATOR}"
        -DCMAKE_MAKE_PROGRAM="${CMAKE_MAKE_PROGRAM}"
        -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
        -DCMAKE_C_COMPILER="${_CONFIG_C_COMPILER}"
        -DCMAKE_CXX_COMPILER="${_CONFIG_CXX_COMPILER}"
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
        -DIREE_HAL_DRIVERS_TO_BUILD="${_CONFIG_HAL_DRIVERS_TO_BUILD}"
        -DIREE_TARGET_BACKENDS_TO_BUILD="${_CONFIG_TARGET_BACKENDS_TO_BUILD}"
    WORKING_DIRECTORY ${_CONFIG_BINARY_ROOT}
    DEPENDS iree_prepare_${config_name}_dir
    COMMENT "Configuring IREE for ${config_name} build...")

  add_custom_target(iree_configure_${config_name} DEPENDS ${_CONFIG_BINARY_ROOT}/CMakeCache.txt)
endfunction()

# iree_get_build_command
#
# Gets the CMake build command for the given `target`.
#
# Parameters:
# - target: the target to build.
# - bin_dir: root binary directory containing CMakeCache.txt.
# - cmd_var: variable name for receiving the build command.
function(iree_get_build_command target bin_dir cmd_var)
  cmake_parse_arguments(_RULE "" "CONFIG" "" ${ARGN})
  if(NOT _RULE_CONFIG)
    set(_RULE_CONFIG "$<CONFIG>")
  endif()
  if (CMAKE_GENERATOR MATCHES "Make")
    # Use special command for Makefiles to support parallelism.
    set(${cmd_var} "$(MAKE)" "-C" "${bin_dir}" "${target}" PARENT_SCOPE)
  else()
    set(${cmd_var} "${CMAKE_COMMAND}" --build ${bin_dir} --target ${target} --config ${_RULE_CONFIG} PARENT_SCOPE)
  endif()
endfunction()

# iree_host_install
#
# Defines custom commands and targets for installing the given `target` under
# host configuration. The custom target for install will be named as
# `iree_host_install_${target}`.
#
# Precondition:
# - iree_create_configuration(HOST) is invoked previously.
#
# Parameters:
# - COMPONENT: installation component; used for filtering installation targets.
# - PREFIX: the root installation path prefix.
# - DEPENDS: addtional dependencies for the installation.
function(iree_host_install target)
  cmake_parse_arguments(_RULE "" "COMPONENT;PREFIX" "DEPENDS" ${ARGN})
  if(_RULE_COMPONENT)
    set(component_option -DCMAKE_INSTALL_COMPONENT="${_RULE_COMPONENT}")
  endif()
  if(_RULE_PREFIX)
    set(prefix_option -DCMAKE_INSTALL_PREFIX="${_RULE_PREFIX}")
  endif()

  iree_get_executable_path(${target} output_path)

  add_custom_command(
    OUTPUT ${output_path}
    DEPENDS ${_RULE_DEPENDS}
    COMMAND "${CMAKE_COMMAND}" ${component_option} ${prefix_option}
            -P "${IREE_HOST_BINARY_ROOT}/cmake_install.cmake"
    USES_TERMINAL)

  # Give it a custom target so we can drive the generation manually
  # when useful.
  add_custom_target(iree_host_install_${target} DEPENDS ${output_path})
endfunction()

# iree_declare_host_excutable
#
# Generates custom commands and targets for building and installing a tool on
# host for cross-compilation.
#
# Precondition:
# - iree_create_configuration(HOST) is invoked previously.
#
# Parameters:
# - target: the target to build on host.
# - BUILDONLY: only generates commands for building the target.
# - DEPENDS: any additional dependencies for the target.
function(iree_declare_host_excutable target)
  cmake_parse_arguments(_RULE "BUILDONLY" "" "DEPENDS" ${ARGN})

  iree_get_executable_path(${target} output_path)

  iree_get_build_command(${target} ${IREE_HOST_BINARY_ROOT} build_cmd)

  add_custom_target(iree_host_build_${target}
                    COMMAND ${build_cmd}
                    DEPENDS iree_configure_HOST ${_RULE_DEPENDS}
                    WORKING_DIRECTORY "${IREE_HOST_BINARY_ROOT}"
                    COMMENT "Building host ${target}..."
                    USES_TERMINAL)

  if(_RULE_BUILDONLY)
    return()
  endif()

  iree_host_install(${target}
                    COMPONENT ${target}
                    PREFIX ${IREE_HOST_BINARY_ROOT}
                    DEPENDS iree_host_build_${target})

  # Give it a custom target so we can drive the generation manually
  # when useful.
  add_custom_target(iree_host_${target} DEPENDS "${output_path}")
endfunction()
