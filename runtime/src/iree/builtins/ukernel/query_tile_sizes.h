// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_QUERY_TILE_SIZES_H_
#define IREE_BUILTINS_UKERNEL_QUERY_TILE_SIZES_H_

#include "iree/builtins/ukernel/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Public entry points
//===----------------------------------------------------------------------===//

typedef struct iree_uk_query_tile_sizes_2d_params_t {
  iree_uk_uint32_t flags;
  iree_uk_ssize_t size0;
  iree_uk_ssize_t size1;
  const iree_uk_uint64_t* cpu_data;
} iree_uk_query_tile_sizes_2d_params_t;

typedef struct iree_uk_query_tile_sizes_2d_out_params_t {
  iree_uk_ssize_t tile_size0;
  iree_uk_ssize_t tile_size1;
} iree_uk_query_tile_sizes_2d_out_params_t;

IREE_UK_EXPORT void iree_uk_query_tile_sizes_2d(
    const iree_uk_query_tile_sizes_2d_params_t* params,
    iree_uk_query_tile_sizes_2d_out_params_t* out_params);

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static inline iree_uk_uint32_t iree_uk_query_tile_sizes_operand_role(
    iree_uk_uint32_t flags) {
  return flags & IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_MASK_INTERNAL;
}

static inline iree_uk_uint32_t iree_uk_query_tile_sizes_operation(
    iree_uk_uint32_t flags) {
  return flags & IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MASK_INTERNAL;
}

// Internal use only. Holds matmul tile params as returned from architecture
// specific backend code.
typedef struct iree_uk_matmul_tile_sizes_t {
  int M, K, N;
} iree_uk_matmul_tile_sizes_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BUILTINS_UKERNEL_QUERY_TILE_SIZES_H_
