// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_QUERY_TILE_SIZES_INTERNAL_H_
#define IREE_BUILTINS_UKERNEL_QUERY_TILE_SIZES_INTERNAL_H_

#include "iree/builtins/ukernel/query_tile_sizes.h"

static inline iree_uk_uint32_t iree_uk_query_tile_sizes_operand_role(
    iree_uk_uint32_t flags) {
  return flags & IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_MASK;
}

static inline iree_uk_uint32_t iree_uk_query_tile_sizes_operation(
    iree_uk_uint32_t flags) {
  return flags & IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MASK;
}

// Holds matmul tile params as returned from architecture-specific backend code.
typedef struct iree_uk_matmul_tile_sizes_t {
  int M, K, N;
} iree_uk_matmul_tile_sizes_t;

// Architecture-specific implementation.
bool iree_uk_query_matmul_tile_sizes_arch(
    const iree_uk_query_tile_sizes_2d_params_t* params,
    iree_uk_matmul_tile_sizes_t* out_matmul_tile_sizes);

#endif  // IREE_BUILTINS_UKERNEL_QUERY_TILE_SIZES_INTERNAL_H_
