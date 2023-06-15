# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

iree_c_embed_data(
  NAME
    builtin
  SRCS
    "copy_buffer_generic.metal"
    "fill_buffer_generic.metal"
  C_FILE_OUTPUT
    "metal_buffer_kernels.c"
  H_FILE_OUTPUT
    "metal_buffer_kernels.h"
  IDENTIFIER
    "metal_buffer_kernels"
  FLATTEN
  PUBLIC
)
