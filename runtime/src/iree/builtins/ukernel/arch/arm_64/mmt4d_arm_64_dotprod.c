// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <arm_neon.h>

#include "iree/builtins/ukernel/arch/arm_64/mmt4d_arm_64.h"

void iree_uk_mmt4d_tile_i8i8i32_8x8x4_arm_64_dotprod(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel, iree_uk_int32_t K,
    iree_uk_uint32_t flags, const iree_uk_mmt4d_params_t* params) {
  (void)params;
  const iree_uk_int8_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;
  int32x4_t acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10,
      acc11, acc12, acc13, acc14, acc15;
  if (flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    acc0 = vld1q_s32(out_ptr + 4 * 0);
    acc1 = vld1q_s32(out_ptr + 4 * 1);
    acc2 = vld1q_s32(out_ptr + 4 * 2);
    acc3 = vld1q_s32(out_ptr + 4 * 3);
    acc4 = vld1q_s32(out_ptr + 4 * 4);
    acc5 = vld1q_s32(out_ptr + 4 * 5);
    acc6 = vld1q_s32(out_ptr + 4 * 6);
    acc7 = vld1q_s32(out_ptr + 4 * 7);
    acc8 = vld1q_s32(out_ptr + 4 * 8);
    acc9 = vld1q_s32(out_ptr + 4 * 9);
    acc10 = vld1q_s32(out_ptr + 4 * 10);
    acc11 = vld1q_s32(out_ptr + 4 * 11);
    acc12 = vld1q_s32(out_ptr + 4 * 12);
    acc13 = vld1q_s32(out_ptr + 4 * 13);
    acc14 = vld1q_s32(out_ptr + 4 * 14);
    acc15 = vld1q_s32(out_ptr + 4 * 15);
  } else {
    acc0 = vdupq_n_s32(0);
    acc1 = vdupq_n_s32(0);
    acc2 = vdupq_n_s32(0);
    acc3 = vdupq_n_s32(0);
    acc4 = vdupq_n_s32(0);
    acc5 = vdupq_n_s32(0);
    acc6 = vdupq_n_s32(0);
    acc7 = vdupq_n_s32(0);
    acc8 = vdupq_n_s32(0);
    acc9 = vdupq_n_s32(0);
    acc10 = vdupq_n_s32(0);
    acc11 = vdupq_n_s32(0);
    acc12 = vdupq_n_s32(0);
    acc13 = vdupq_n_s32(0);
    acc14 = vdupq_n_s32(0);
    acc15 = vdupq_n_s32(0);
  }
  IREE_UK_ASSUME(K >= 1);
  for (int k = 0; k < K; ++k) {
    int8x16_t lhs0 = vld1q_s8(lhs_ptr + 0);
    int8x16_t lhs1 = vld1q_s8(lhs_ptr + 16);
    lhs_ptr += 32;
    int8x16_t rhs0 = vld1q_s8(rhs_ptr + 0);
    int8x16_t rhs1 = vld1q_s8(rhs_ptr + 16);
    rhs_ptr += 32;
    acc0 = vdotq_lane_s32(acc0, rhs0, vget_low_s8(lhs0), 0);
    acc1 = vdotq_lane_s32(acc1, rhs1, vget_low_s8(lhs0), 0);
    acc2 = vdotq_lane_s32(acc2, rhs0, vget_low_s8(lhs0), 1);
    acc3 = vdotq_lane_s32(acc3, rhs1, vget_low_s8(lhs0), 1);
    acc4 = vdotq_lane_s32(acc4, rhs0, vget_high_s8(lhs0), 0);
    acc5 = vdotq_lane_s32(acc5, rhs1, vget_high_s8(lhs0), 0);
    acc6 = vdotq_lane_s32(acc6, rhs0, vget_high_s8(lhs0), 1);
    acc7 = vdotq_lane_s32(acc7, rhs1, vget_high_s8(lhs0), 1);
    acc8 = vdotq_lane_s32(acc8, rhs0, vget_low_s8(lhs1), 0);
    acc9 = vdotq_lane_s32(acc9, rhs1, vget_low_s8(lhs1), 0);
    acc10 = vdotq_lane_s32(acc10, rhs0, vget_low_s8(lhs1), 1);
    acc11 = vdotq_lane_s32(acc11, rhs1, vget_low_s8(lhs1), 1);
    acc12 = vdotq_lane_s32(acc12, rhs0, vget_high_s8(lhs1), 0);
    acc13 = vdotq_lane_s32(acc13, rhs1, vget_high_s8(lhs1), 0);
    acc14 = vdotq_lane_s32(acc14, rhs0, vget_high_s8(lhs1), 1);
    acc15 = vdotq_lane_s32(acc15, rhs1, vget_high_s8(lhs1), 1);
  }
  vst1q_s32(out_ptr + 4 * 0, acc0);
  vst1q_s32(out_ptr + 4 * 1, acc1);
  vst1q_s32(out_ptr + 4 * 2, acc2);
  vst1q_s32(out_ptr + 4 * 3, acc3);
  vst1q_s32(out_ptr + 4 * 4, acc4);
  vst1q_s32(out_ptr + 4 * 5, acc5);
  vst1q_s32(out_ptr + 4 * 6, acc6);
  vst1q_s32(out_ptr + 4 * 7, acc7);
  vst1q_s32(out_ptr + 4 * 8, acc8);
  vst1q_s32(out_ptr + 4 * 9, acc9);
  vst1q_s32(out_ptr + 4 * 10, acc10);
  vst1q_s32(out_ptr + 4 * 11, acc11);
  vst1q_s32(out_ptr + 4 * 12, acc12);
  vst1q_s32(out_ptr + 4 * 13, acc13);
  vst1q_s32(out_ptr + 4 * 14, acc14);
  vst1q_s32(out_ptr + 4 * 15, acc15);
}
