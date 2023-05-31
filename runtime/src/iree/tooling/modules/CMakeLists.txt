# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Link in externally defined modules.
# This allows users to conditionally enable modules that live outside of the
# IREE source tree by specifying a few cmake variables.
#
# Drivers are expected to have a CMakeLists.txt that is parsed when enabled.
# If a module is optional it may set an IREE_EXTERNAL_TOOLING_MODULE_{name}_FOUND
# variable to FALSE and be ignored, such as when dependencies are not found or
# other user configuration has disabled them.
#
# Each module provides a static library target name and a function that is
# called at runtime to register the module.
#
# Required variables:
#   IREE_EXTERNAL_TOOLING_MODULE_{name}_TARGET: static library target name.
#   IREE_EXTERNAL_TOOLING_MODULE_{name}_NAME: module name in MLIR.
#   IREE_EXTERNAL_TOOLING_MODULE_{name}_CREATE: module creation name:
#      iree_status_t {CREATE}(iree_vm_instance_t* instance,
#                             iree_allocator_t host_allocator,
#                             iree_vm_module_t** out_module)
# Optional variables:
#   IREE_EXTERNAL_TOOLING_MODULE_{name}_OPTIONAL: true if the module not being
#      found is not an error.
#   IREE_EXTERNAL_TOOLING_MODULE_{name}_SOURCE_DIR: source directory with a
#      CMakeLists.txt included when the module is enabled.
#   IREE_EXTERNAL_TOOLING_MODULE_{name}_BINARY_DIR: binary directory for cmake
#      outs.
#   IREE_EXTERNAL_TOOLING_MODULE_{name}_FOUND: bool to indicate whether the
#      module was found and valid for use.
#   IREE_EXTERNAL_TOOLING_MODULE_{name}_REGISTER_TYPES: optional type name:
#      iree_status_t {REGISTER_TYPES}(iree_vm_instance_t* instance)
set(IREE_EXTERNAL_TOOLING_MODULES_USED)
foreach(_MODULE_NAME ${IREE_EXTERNAL_TOOLING_MODULES})
  string(TOUPPER "IREE_EXTERNAL_TOOLING_MODULES_${_MODULE_NAME}" _MODULE_VAR)
  string(REGEX REPLACE "-" "_" _MODULE_VAR ${_MODULE_VAR})
  message(STATUS "Adding IREE external tooling module: ${_MODULE_NAME}")

  set(_MODULE_OPTIONAL ${${_MODULE_VAR}_OPTIONAL})
  set(_MODULE_SOURCE_DIR ${${_MODULE_VAR}_SOURCE_DIR})
  set(_MODULE_BINARY_DIR ${${_MODULE_VAR}_BINARY_DIR})

  # Default to found unless the user overrides it in the module source.
  # This allows the module to decide to disable itself even if the user
  # requested it.
  set(${_MODULE_VAR}_FOUND TRUE CACHE BOOL
      "Whether the external module is valid for use.")

  # Include the module source CMakeLists.txt if required.
  # Users may have already defined the targets and not need this.
  if(_MODULE_SOURCE_DIR)
    if(NOT EXISTS "${_MODULE_SOURCE_DIR}/CMakeLists.txt")
      message(FATAL_ERROR "External module CMakeLists.txt not found at "
                          "${_MODULE_SOURCE_DIR}")
    endif()
    add_subdirectory(${_MODULE_SOURCE_DIR} ${_MODULE_BINARY_DIR})
  endif()

  # If found then add to the list of valid modules.
  if(${${_MODULE_VAR}_FOUND})
    list(APPEND IREE_EXTERNAL_TOOLING_MODULES_USED ${_MODULE_NAME})
  else()
    if(${_MODULE_OPTIONAL})
      message(STATUS "Optional external module '${_MODULE_NAME}' requested "
                      "but not found; disabling and continuing")
    else()
      message(FATAL_ERROR "External module '${_MODULE_NAME}' not found; may "
                          "have unavailable dependencies")
    endif()
  endif()
endforeach()

# Produce resolve_external.c that contains a resolver switch for all modules.
# This will be called by resolver.c after internal modules are tried.
set(_RESOLVER_EXTERNAL_C_SRC)
set(_RESOLVER_EXTERNAL_COPTS)
set(_RESOLVER_EXTERNAL_DEPS)
if(IREE_EXTERNAL_TOOLING_MODULES_USED)
  message(STATUS "Registering external tooling modules: ${IREE_EXTERNAL_TOOLING_MODULES_USED}")

  set(_RESOLVER_EXTERNAL_COPTS "-DIREE_HAVE_EXTERNAL_TOOLING_MODULES=1")

  # Build the list of deps and our source code lines.
  set(_RESOLVER_EXTERNAL_DEPS)
  set(_RESOLVER_EXTERNAL_DECLS)
  set(_RESOLVER_EXTERNAL_TYPES)
  set(_RESOLVER_EXTERNAL_CHECKS)
  foreach(_MODULE_NAME ${IREE_EXTERNAL_TOOLING_MODULES_USED})
    string(TOUPPER "IREE_EXTERNAL_TOOLING_MODULE_${_MODULE_NAME}" _MODULE_VAR)
    string(REGEX REPLACE "-" "_" _MODULE_VAR ${_MODULE_VAR})
    set(_MODULE_TARGET ${${_MODULE_VAR}_TARGET})
    set(_MODULE_NAME ${${_MODULE_VAR}_NAME})
    set(_MODULE_REGISTER_TYPES ${${_MODULE_VAR}_REGISTER_TYPES})
    set(_MODULE_CREATE ${${_MODULE_VAR}_CREATE})
    list(APPEND _RESOLVER_EXTERNAL_DEPS ${_MODULE_TARGET})
    if(_MODULE_REGISTER_TYPES)
      list(APPEND _RESOLVER_EXTERNAL_DECLS
          "extern iree_status_t ${_MODULE_REGISTER_TYPES}(iree_vm_instance_t* instance);\n")
      list(APPEND _RESOLVER_EXTERNAL_REGISTER_TYPES
          "IREE_RETURN_IF_ERROR(${_MODULE_REGISTER_TYPES}(instance));\n")
    endif()
    list(APPEND _RESOLVER_EXTERNAL_DECLS
        "extern iree_status_t ${_MODULE_CREATE}(iree_vm_instance_t* instance, iree_allocator_t host_allocator, iree_vm_module_t** out_module);\n")
    list(APPEND _RESOLVER_EXTERNAL_TRY_CREATES
        "if (iree_string_view_equal(dependency->name, IREE_SV(\"${_MODULE_NAME}\"))) {\n    IREE_RETURN_IF_ERROR(${_MODULE_CREATE}(instance, host_allocator, out_module));\n  }\n")
  endforeach()

  # Read template file and substitute variables.
  set(_RESOLVER_EXTERNAL_C_TPL "${CMAKE_CURRENT_SOURCE_DIR}/resolve_external.c.in")
  set(_RESOLVER_EXTERNAL_C_SRC "${CMAKE_CURRENT_BINARY_DIR}/resolve_external.c")
  file(READ ${_RESOLVER_EXTERNAL_C_TPL} _RESOLVER_EXTERNAL_TEMPLATE)
  file(
    CONFIGURE OUTPUT ${_RESOLVER_EXTERNAL_C_SRC}
    CONTENT "${_RESOLVER_EXTERNAL_TEMPLATE}"
  )
endif()

set(_RESOLVER_INTERNAL_DEPS)
list(APPEND _RESOLVER_INTERNAL_DEPS iree::modules::vmvx)

iree_cc_library(
  NAME
    modules
  HDRS
    "resolver.h"
  SRCS
    "resolver.c"
    ${_RESOLVER_EXTERNAL_C_SRC}
  COPTS
    ${_RESOLVER_EXTERNAL_COPTS}
  DEPS
    iree::base
    iree::base::tracing
    iree::vm
    ${_RESOLVER_INTERNAL_DEPS}
    ${_RESOLVER_EXTERNAL_DEPS}
  PUBLIC
)
