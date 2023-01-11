# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(_NAME "iree_samples_custom_module_run")
add_executable(${_NAME} "")
target_sources(${_NAME}
  PRIVATE
    main.c
    module.cc
    module.h
)

set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "custom-module-run")

# TODO(benvanik): make iree_status_annotate_f always available as a function
# instead of defining it empty? otherwise optimized builds of the runtime won't
# export it but external libraries may pull it in.
target_compile_options(${_NAME} PRIVATE ${IREE_DEFAULT_COPTS})

target_link_libraries(${_NAME}
  iree_base_base
  iree_base_internal_file_io
  iree_vm_vm
  iree_vm_bytecode_module
)

add_subdirectory(test)
