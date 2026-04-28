# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_bytecode_module(
  NAME
    hello_world_bytecode_module_webgpu
  SRC
    "hello_world.mlir"
  FLAGS
    "--iree-hal-target-device=webgpu"
  PUBLIC
)
