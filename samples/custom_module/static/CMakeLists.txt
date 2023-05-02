# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_HAL_DRIVER_LOCAL_SYNC)
  return()
endif()
if(NOT "static_sample" IN_LIST IREE_EXTERNAL_TOOLING_MODULES)
  return()
endif()

message(WARNING "IREE custom_module/static sample compiled into all tools")

set(_NAME "iree_samples_custom_module_static_module")
add_library(${_NAME} STATIC module.cc)
target_link_libraries(${_NAME}
  iree_base_base
  iree_hal_hal
  iree_modules_hal_types
  iree_vm_vm
)

# TODO(benvanik): make iree_status_annotate_f always available as a function
# instead of defining it empty? otherwise optimized builds of the runtime won't
# export it but external libraries may pull it in.
target_compile_options(${_NAME} PRIVATE ${IREE_DEFAULT_COPTS})

add_subdirectory(test)
