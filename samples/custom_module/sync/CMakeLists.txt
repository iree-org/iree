# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Sample requires the llvm-cpu compiler backend and the local-sync runtime
# driver. This could be made to work with other backends.
if(NOT IREE_TARGET_BACKEND_LLVM_CPU OR
   NOT IREE_HAL_DRIVER_LOCAL_SYNC)
  return()
endif()

set(_NAME "iree_samples_custom_module_sync_run")
add_executable(${_NAME} "")
target_sources(${_NAME}
  PRIVATE
    main.c
    module.cc
    module.h
)

set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "custom-module-sync-run")

# TODO(benvanik): make iree_status_annotate_f always available as a function
# instead of defining it empty? otherwise optimized builds of the runtime won't
# export it but external libraries may pull it in.
target_compile_options(${_NAME} PRIVATE ${IREE_DEFAULT_COPTS})

target_link_libraries(${_NAME}
  iree_runtime_runtime
)

add_subdirectory(test)
