// Copyright 2025 10xEngineers
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/riscv_64/common_riscv_64.h"
#include "iree/builtins/ukernel/arch/riscv_64/mmt4d_riscv_64_internal.h"

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_f32f32f32_1x32x1_to_7x32x1_riscv_64(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 7);
  const float* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const float* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  float* IREE_UK_RESTRICT out_ptr = out_tile;

}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f32f32f32_1x32x1_to_7x32x1_riscv_64,
    iree_uk_mmt4d_tile_f32f32f32_1x32x1_riscv_64, 1)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f32f32f32_1x32x1_to_7x32x1_riscv_64,
    iree_uk_mmt4d_tile_f32f32f32_2x32x1_riscv_64, 2)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f32f32f32_1x32x1_to_7x32x1_riscv_64,
    iree_uk_mmt4d_tile_f32f32f32_4x32x1_riscv_64, 4)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f32f32f32_1x32x1_to_7x32x1_riscv_64,
    iree_uk_mmt4d_tile_f32f32f32_7x32x1_riscv_64, 7)
