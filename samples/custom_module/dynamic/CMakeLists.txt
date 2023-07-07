# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_HAL_DRIVER_LOCAL_SYNC)
  return()
endif()

if(IREE_ENABLE_RUNTIME_TRACING)
  message(WARNING "IREE_ENABLE_RUNTIME_TRACING enabled but it currently has issues with dynamic libraries")
endif()

set(_NAME "iree_samples_custom_module_dynamic_module")
add_library(${_NAME} SHARED module.cc)
target_link_libraries(${_NAME}
  iree_base_base
  iree_hal_hal
  iree_modules_hal_types
  iree_vm_vm
  iree_vm_dynamic_api
)

# NOTE: this is only required because we want this sample to run on all
# platforms without needing to change the library name (libfoo.so/foo.dll).
set_target_properties(${_NAME}
  PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    PREFIX ""
    OUTPUT_NAME "module"
)

# TODO(benvanik): make iree_status_annotate_f always available as a function
# instead of defining it empty? otherwise optimized builds of the runtime won't
# export it but external libraries may pull it in.
target_compile_options(${_NAME} PRIVATE ${IREE_DEFAULT_COPTS})

add_dependencies(iree-sample-deps ${_NAME})

add_subdirectory(test)
