// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/arm_64/common_arm_64.h"
#include "iree/builtins/ukernel/arch/arm_64/mmt4d_arm_64_internal.h"

void iree_uk_mmt4d_tile_f16f16f16_1x8x1_to_8x8x1_arm_64_fp16(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 8 && iree_uk_is_po2_u32(M0));
  float16_t* IREE_UK_RESTRICT out_ptr = out_tile;
  const float16_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const float16_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  float16x8_t acc[8];
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    for (int i = 0; i < M0; ++i) {
      acc[i] = vld1q_f16(out_ptr + 8 * i);
    }
  } else {
    for (int i = 0; i < M0; ++i) {
      acc[i] = vdupq_n_f16(0);
    }
  }
  IREE_UK_ASSUME(params->K >= 1);
  for (int k = 0; k < params->K; ++k) {
    float16x8_t rhs = vld1q_f16(rhs_ptr);
    rhs_ptr += 8;
    if (M0 <= 2) {
      for (int i = 0; i < M0; ++i) {
        acc[i] = vfmaq_n_f16(acc[i], rhs, *lhs_ptr++);
      }
    } else {
      for (int i = 0; i < M0; i += 4) {
        float16x4_t lhs = vld1_f16(lhs_ptr);
        lhs_ptr += 4;
        acc[i + 0] = vfmaq_lane_f16(acc[i + 0], rhs, lhs, 0);
        acc[i + 1] = vfmaq_lane_f16(acc[i + 1], rhs, lhs, 1);
        acc[i + 2] = vfmaq_lane_f16(acc[i + 2], rhs, lhs, 2);
        acc[i + 3] = vfmaq_lane_f16(acc[i + 3], rhs, lhs, 3);
      }
    }
  }
  for (int i = 0; i < M0; ++i) {
    vst1q_f16(out_ptr + 8 * i, acc[i]);
  }
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0_1_2_4_8(
    iree_uk_mmt4d_tile_f16f16f16_1x8x1_to_8x8x1_arm_64_fp16,
    iree_uk_mmt4d_tile_f16f16f16_1x8x1_arm_64_fp16,
    iree_uk_mmt4d_tile_f16f16f16_2x8x1_arm_64_fp16,
    iree_uk_mmt4d_tile_f16f16f16_4x8x1_arm_64_fp16,
    iree_uk_mmt4d_tile_f16f16f16_8x8x1_arm_64_fp16)
