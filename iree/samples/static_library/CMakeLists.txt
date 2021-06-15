# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(${IREE_BUILD_STATIC_LIBRARY_SAMPLES} AND NOT ${IREE_BUILD_SAMPLES})
  return()
endif()

iree_c_embed_data(
  NAME
    static_library_module
  GENERATED_SRCS
    ${CMAKE_CURRENT_BINARY_DIR}/static_library_module.vmfb
  C_FILE_OUTPUT
    "static_module.c"
  H_FILE_OUTPUT
    "static_module.h"
  IDENTIFIER
    "iree_samples_static_library_module"
  FLATTEN
  PUBLIC
)

iree_cc_binary(
  NAME
    static_library_demo
  SRCS
    "static_library_demo.c"
  DEPS
    ::static_library_module
    iree::base
    iree::hal
    iree::hal::local
    iree::hal::local::loaders::static_library_loader
    iree::hal::local::task_driver
    iree::modules::hal
    iree::task::api
    iree::vm
    iree::vm::bytecode_module
    ${CMAKE_CURRENT_BINARY_DIR}/static_library_module.o
)

### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###
