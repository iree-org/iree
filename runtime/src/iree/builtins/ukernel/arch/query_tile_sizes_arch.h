// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_ARCH_QUERY_TILE_SIZES_ARCH_H_
#define IREE_BUILTINS_UKERNEL_ARCH_QUERY_TILE_SIZES_ARCH_H_

#include "iree/builtins/ukernel/query_tile_sizes_types.h"

bool iree_uk_query_matmul_tile_sizes_arch(
    const iree_uk_query_tile_sizes_2d_params_t* params,
    iree_uk_matmul_tile_sizes_t* out_matmul_tile_sizes);

#endif  // IREE_BUILTINS_UKERNEL_ARCH_QUERY_TILE_SIZES_ARCH_H_
