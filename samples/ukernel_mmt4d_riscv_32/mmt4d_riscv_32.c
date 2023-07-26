// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mmt4d_riscv_32_internal.h"

void iree_uk_mmt4d_tile_i8i8i32_16x16x16_riscv_32(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel, iree_uk_int32_t K,
    iree_uk_uint32_t flags, const iree_uk_mmt4d_params_t* params) {
  (void)params;
  const iree_uk_int8_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;
  // 16x16 accumulator tile. In this dummy code, this is a stack array. In
  // a real implementation this would be a group of SIMD registers.
  iree_uk_int32_t acc_tile[16 * 16];
  if (flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    // This mmt4d operation is accumulating into the existing accumulator.
    // Load a 16x16 tile of Accumulator data. This happens in the general-case
    // lowering of a linalg.matmul. In a real impl this would be a few
    // SIMD register loads.
    iree_uk_memcpy(acc_tile, out_ptr, sizeof acc_tile);
  } else {
    // This mmt4d operation is overwriting the accumulator, i.e. doing as if
    // the existing accumulator is filled with zeros. This happens in the
    // lowering of folded linalg.fill->linalg.matmul. In a real impl this would`
    // be a few SIMD register zeroings.
    iree_uk_memset(acc_tile, 0, sizeof acc_tile);
  }
  IREE_UK_ASSUME(K >= 1);
  for (int k = 0; k < K; ++k) {
    // Load a 16x16 tile of LHS data. In a real impl this would be a few
    // SIMD register loads.
    iree_uk_int8_t lhs_tile[16 * 16];
    iree_uk_memcpy(lhs_tile, lhs_ptr, sizeof lhs_tile);
    lhs_ptr += sizeof lhs_tile;
    // Load a 16x16 tile of RHS data. In a real impl this would be a few
    // SIMD register loads.
    iree_uk_int8_t rhs_tile[16 * 16];
    iree_uk_memcpy(rhs_tile, rhs_ptr, sizeof rhs_tile);
    rhs_ptr += sizeof rhs_tile;
    // Arithmetic. In a real implementation this would be SIMD arithmetic
    // instructions on registers.
    for (int m0 = 0; m0 < 16; ++m0) {
      for (int n0 = 0; n0 < 16; ++n0) {
        for (int k0 = 0; k0 < 16; ++k0) {
          acc_tile[16 * m0 + n0] =
              lhs_tile[16 * m0 + k0] * rhs_tile[16 * n0 + k0];
        }
      }
    }
  }
  // Store acc_tile to destination. In a real impl this would be a few SIMD
  // register stores.
  iree_uk_memcpy(out_ptr, acc_tile, sizeof acc_tile);
}
