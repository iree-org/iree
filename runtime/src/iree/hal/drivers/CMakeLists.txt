# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# TODO: we could make all of the internal drivers operate in the same way and
# have init.c generated too. That'd require bazel goo; today by having the hand-
# coded file we can use it in bazel as-is.

# Link in externally defined drivers.
# This allows users to conditionally enable drivers that live outside of the
# IREE source tree by specifying a few cmake variables.
#
# Drivers are expected to have a CMakeLists.txt that is parsed when enabled.
# If a driver is optional it may set an IREE_EXTERNAL_{name}_HAL_DRIVER_FOUND
# variable to FALSE and be ignored, such as when dependencies are not found or
# other user configuration has disabled them.
#
# Each driver provides a static library target name and a function that is
# called at runtime to register the driver.
#
# Required properties (typically set via iree_register_external_hal_driver()):
#   IREE_EXTERNAL_{name}_HAL_DRIVER_TARGET: static library target name.
#   IREE_EXTERNAL_{name}_HAL_DRIVER_REGISTER: registration function:
#      iree_status_t {name}_register(iree_hal_driver_registry_t* registry)
# Optional properties:
#   IREE_EXTERNAL_{name}_HAL_DRIVER_OPTIONAL: true if the driver not being found
#      is not an error.
#   IREE_EXTERNAL_{name}_HAL_DRIVER_SOURCE_DIR: source directory with a
#      CMakeLists.txt included when the driver is enabled.
#   IREE_EXTERNAL_{name}_HAL_DRIVER_BINARY_DIR: binary directory for cmake outs.
# Optional cache vars:
#   IREE_EXTERNAL_{name}_HAL_DRIVER_FOUND: bool to indicate whether the driver
#      was found and valid for use.
set(IREE_EXTERNAL_HAL_DRIVERS_USED)
get_property(IREE_EXTERNAL_HAL_DRIVERS_AVAILABLE GLOBAL PROPERTY IREE_EXTERNAL_HAL_DRIVERS_AVAILABLE)
foreach(_DRIVER_NAME ${IREE_EXTERNAL_HAL_DRIVERS})
  if(NOT _DRIVER_NAME IN_LIST IREE_EXTERNAL_HAL_DRIVERS_AVAILABLE)
    message(SEND_ERROR 
      "Driver passed to -DIREE_EXTERNAL_HAL_DRIVERS=${_DRIVER_NAME} is not found. Available: "
      "${IREE_EXTERNAL_HAL_DRIVERS_AVAILABLE}")
  endif()
  string(TOUPPER "IREE_EXTERNAL_${_DRIVER_NAME}_HAL_DRIVER" _DRIVER_VAR)
  string(REGEX REPLACE "-" "_" _DRIVER_VAR ${_DRIVER_VAR})
  message(STATUS "Adding IREE external HAL driver: ${_DRIVER_NAME}")

  get_property(_DRIVER_OPTIONAL GLOBAL PROPERTY ${_DRIVER_VAR}_OPTIONAL)
  get_property(_DRIVER_SOURCE_DIR GLOBAL PROPERTY ${_DRIVER_VAR}_SOURCE_DIR)
  get_property(_DRIVER_BINARY_DIR GLOBAL PROPERTY ${_DRIVER_VAR}_BINARY_DIR)

  # Default to found unless the user overrides it in the driver source.
  # This allows the driver to decide to disable itself even if the user
  # requested it.
  set(${_DRIVER_VAR}_FOUND TRUE CACHE BOOL
      "Whether the external driver is valid for use.")

  # Include the driver source CMakeLists.txt if required.
  # Users may have already defined the targets and not need this.
  if(_DRIVER_SOURCE_DIR)
    if(NOT EXISTS "${_DRIVER_SOURCE_DIR}/CMakeLists.txt")
      message(FATAL_ERROR "External driver CMakeLists.txt not found at "
                          "${_DRIVER_SOURCE_DIR}")
    endif()
    add_subdirectory(${_DRIVER_SOURCE_DIR} ${_DRIVER_BINARY_DIR})
  endif()

  # If found then add to the list of valid drivers.
  if(${${_DRIVER_VAR}_FOUND})
    list(APPEND IREE_EXTERNAL_HAL_DRIVERS_USED ${_DRIVER_NAME})
  else()
    if(${_DRIVER_OPTIONAL})
      message(STATUS "Optional external driver '${_DRIVER_NAME}' requested "
                      "but not found; disabling and continuing")
    else()
      message(FATAL_ERROR "External driver '${_DRIVER_NAME}' not found; may "
                          "have unavailable dependencies")
    endif()
  endif()
endforeach()

# Produce an init_external.c that contains all of the registration calls.
# This will be called by the init.c after internal drivers are registered.
set(_INIT_EXTERNAL_C_SRC)
set(_INIT_EXTERNAL_COPTS)
set(_INIT_EXTERNAL_DEPS)
if(IREE_EXTERNAL_HAL_DRIVERS_USED)
  message(STATUS "Registering external HAL drivers: ${IREE_EXTERNAL_HAL_DRIVERS_USED}")

  set(_INIT_EXTERNAL_COPTS "-DIREE_HAVE_HAL_EXTERNAL_DRIVERS=1")

  # Build the list of deps and our source code lines.
  set(_INIT_EXTERNAL_DEPS)
  set(_INIT_EXTERNAL_REGISTER_DECLS)
  set(_INIT_EXTERNAL_REGISTER_CALLS)
  foreach(_DRIVER_NAME ${IREE_EXTERNAL_HAL_DRIVERS_USED})
    string(TOUPPER "IREE_EXTERNAL_${_DRIVER_NAME}_HAL_DRIVER" _DRIVER_VAR)
    string(REGEX REPLACE "-" "_" _DRIVER_VAR ${_DRIVER_VAR})
    get_property(_DRIVER_TARGET GLOBAL PROPERTY ${_DRIVER_VAR}_TARGET)
    get_property(_DRIVER_REGISTER GLOBAL PROPERTY ${_DRIVER_VAR}_REGISTER)
    list(APPEND _INIT_EXTERNAL_DEPS ${_DRIVER_TARGET})

    if(NOT _DRIVER_TARGET OR NOT _DRIVER_REGISTER)
      message(SEND_ERROR 
        "Bad configuration for external HAL driver ${_DRIVER_VAR}: "
        "Missing ${_DRIVER_VAR}_TARGET or ${_DRIVER_VAR}_REGISTER vars. "
        "This usually indicates that the project was not included in the "
        "CMake build correctly: (got '${_DRIVER_TARGET}' '${_DRIVER_REGISTER}')")
    endif()
    list(APPEND _INIT_EXTERNAL_REGISTER_DECLS
        "extern iree_status_t ${_DRIVER_REGISTER}(iree_hal_driver_registry_t* registry);\n")
    list(APPEND _INIT_EXTERNAL_REGISTER_CALLS
        "IREE_RETURN_IF_ERROR(${_DRIVER_REGISTER}(registry));\n")
  endforeach()

  # Read template file and substitute variables.
  set(_INIT_EXTERNAL_C_TPL "${CMAKE_CURRENT_SOURCE_DIR}/init_external.c.in")
  set(_INIT_EXTERNAL_C_SRC "${CMAKE_CURRENT_BINARY_DIR}/init_external.c")
  file(READ ${_INIT_EXTERNAL_C_TPL} _INIT_EXTERNAL_TEMPLATE)
  file(
    CONFIGURE OUTPUT ${_INIT_EXTERNAL_C_SRC}
    CONTENT "${_INIT_EXTERNAL_TEMPLATE}"
  )
endif()

set(_INIT_INTERNAL_DEPS)
if(IREE_HAL_DRIVER_CUDA)
  add_subdirectory(cuda)
  list(APPEND _INIT_INTERNAL_DEPS iree::hal::drivers::cuda::registration)
endif()
if(IREE_HAL_DRIVER_HIP)
  add_subdirectory(hip)
  list(APPEND _INIT_INTERNAL_DEPS iree::hal::drivers::hip::registration)
endif()
if(IREE_HAL_DRIVER_LOCAL_SYNC)
  add_subdirectory(local_sync)
  list(APPEND _INIT_INTERNAL_DEPS iree::hal::drivers::local_sync::registration)
endif()
if(IREE_HAL_DRIVER_LOCAL_TASK)
  add_subdirectory(local_task)
  list(APPEND _INIT_INTERNAL_DEPS iree::hal::drivers::local_task::registration)
endif()
if(IREE_HAL_DRIVER_VULKAN)
  add_subdirectory(vulkan)
  list(APPEND _INIT_INTERNAL_DEPS iree::hal::drivers::vulkan::registration)
endif()
if(IREE_HAL_DRIVER_METAL)
  add_subdirectory(metal)
  list(APPEND _INIT_INTERNAL_DEPS iree::hal::drivers::metal::registration)
endif()

iree_cc_library(
  NAME
    drivers
  HDRS
    "init.h"
  SRCS
    "init.c"
    ${_INIT_EXTERNAL_C_SRC}
  COPTS
    ${_INIT_EXTERNAL_COPTS}
  DEPS
    iree::base
    ${_INIT_INTERNAL_DEPS}
    ${_INIT_EXTERNAL_DEPS}
  PUBLIC
)
