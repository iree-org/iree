// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_ARCH_X86_64_QUERY_TILE_SIZES_X86_64_H_
#define IREE_BUILTINS_UKERNEL_ARCH_X86_64_QUERY_TILE_SIZES_X86_64_H_

#include "iree/builtins/ukernel/query_tile_sizes_internal.h"

bool iree_uk_query_matmul_tile_sizes_x86_64(
    const iree_uk_query_tile_sizes_2d_params_t* params,
    iree_uk_matmul_tile_sizes_t* out_matmul_tile_sizes);

#endif  // IREE_BUILTINS_UKERNEL_ARCH_X86_64_QUERY_TILE_SIZES_X86_64_H_
