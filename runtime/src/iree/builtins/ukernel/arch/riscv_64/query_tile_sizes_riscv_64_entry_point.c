// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/riscv_64/common_riscv_64.h"
#include "iree/builtins/ukernel/query_tile_sizes_internal.h"

static iree_uk_matmul_tile_sizes_t
iree_uk_query_matmul_tile_sizes_riscv_64_f32f32f32(
    const iree_uk_query_tile_sizes_2d_params_t* params) {
#if defined(IREE_UK_BUILD_RISCV_64_V)
  if (iree_uk_cpu_riscv_64_v(params->cpu_data)) {
    return (iree_uk_matmul_tile_sizes_t){.M = 7, .K = 1, .N = 16};
  }
#endif
  // generic fallback
  return (iree_uk_matmul_tile_sizes_t){.M = 8, .K = 1, .N = 8};
}

bool iree_uk_query_matmul_tile_sizes_arch(
    const iree_uk_query_tile_sizes_2d_params_t* params,
    iree_uk_matmul_tile_sizes_t* out_matmul_tile_sizes) {
  iree_uk_uint32_t op = iree_uk_query_tile_sizes_operation(params->flags);
  if (op == IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_F32F32F32) {
    *out_matmul_tile_sizes =
        iree_uk_query_matmul_tile_sizes_riscv_64_f32f32f32(params);
    return true;
  } else {
    // Shouldn't happen, validated earlier.
    return false;
  }
}
