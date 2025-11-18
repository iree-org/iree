# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Sample demonstrating external transient buffer allocation using the C API.

if(IREE_HAL_DRIVER_LOCAL_SYNC AND IREE_HAL_EXECUTABLE_LOADER_VMVX_MODULE AND
   (IREE_TARGET_BACKEND_VMVX OR IREE_HOST_BIN_DIR))

iree_cc_binary(
  NAME
    external_transients_vmvx_sync
  SRCS
    "device_vmvx_sync.c"
    "main.c"
  DEPS
    ::external_transients_bytecode_module_vmvx_c
    iree::base
    iree::hal
    iree::hal::drivers::local_sync::sync_driver
    iree::hal::local
    iree::hal::local::loaders::vmvx_module_loader
    iree::modules::hal
    iree::vm
    iree::vm::bytecode::module
)

iree_bytecode_module(
  NAME
    external_transients_bytecode_module_vmvx
  SRC
    "in_place_computation.mlir"
  C_IDENTIFIER
    "iree_samples_external_transients_module_vmvx"
  FLAGS
    "--iree-hal-target-device=local"
    "--iree-hal-local-target-device-backends=vmvx"
  PUBLIC
)

iree_native_test(
  NAME
    "external_transients_vmvx_sync_test"
  SRC
    ::external_transients_vmvx_sync
)

endif()
