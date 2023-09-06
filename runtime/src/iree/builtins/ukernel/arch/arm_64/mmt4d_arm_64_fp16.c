// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/arm_64/common_arm_64.h"
#include "iree/builtins/ukernel/arch/arm_64/mmt4d_arm_64_internal.h"

void iree_uk_mmt4d_tile_f16f16f16_8x8x1_arm_64_fp16(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  float16_t* IREE_UK_RESTRICT out_ptr = out_tile;
  const float16_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const float16_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  float16x8_t acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    acc0 = vld1q_f16(out_ptr + 8 * 0);
    acc1 = vld1q_f16(out_ptr + 8 * 1);
    acc2 = vld1q_f16(out_ptr + 8 * 2);
    acc3 = vld1q_f16(out_ptr + 8 * 3);
    acc4 = vld1q_f16(out_ptr + 8 * 4);
    acc5 = vld1q_f16(out_ptr + 8 * 5);
    acc6 = vld1q_f16(out_ptr + 8 * 6);
    acc7 = vld1q_f16(out_ptr + 8 * 7);
  } else {
    acc0 = vdupq_n_f16(0);
    acc1 = vdupq_n_f16(0);
    acc2 = vdupq_n_f16(0);
    acc3 = vdupq_n_f16(0);
    acc4 = vdupq_n_f16(0);
    acc5 = vdupq_n_f16(0);
    acc6 = vdupq_n_f16(0);
    acc7 = vdupq_n_f16(0);
  }
  IREE_UK_ASSUME(params->K >= 1);
  for (int k = 0; k < params->K; ++k) {
    float16x8_t lhs = vld1q_f16(lhs_ptr);
    lhs_ptr += 8;
    float16x8_t rhs = vld1q_f16(rhs_ptr);
    rhs_ptr += 8;
    acc0 = vfmaq_lane_f16(acc0, rhs, vget_low_f16(lhs), 0);
    acc1 = vfmaq_lane_f16(acc1, rhs, vget_low_f16(lhs), 1);
    acc2 = vfmaq_lane_f16(acc2, rhs, vget_low_f16(lhs), 2);
    acc3 = vfmaq_lane_f16(acc3, rhs, vget_low_f16(lhs), 3);
    acc4 = vfmaq_lane_f16(acc4, rhs, vget_high_f16(lhs), 0);
    acc5 = vfmaq_lane_f16(acc5, rhs, vget_high_f16(lhs), 1);
    acc6 = vfmaq_lane_f16(acc6, rhs, vget_high_f16(lhs), 2);
    acc7 = vfmaq_lane_f16(acc7, rhs, vget_high_f16(lhs), 3);
  }
  vst1q_f16(out_ptr + 8 * 0, acc0);
  vst1q_f16(out_ptr + 8 * 1, acc1);
  vst1q_f16(out_ptr + 8 * 2, acc2);
  vst1q_f16(out_ptr + 8 * 3, acc3);
  vst1q_f16(out_ptr + 8 * 4, acc4);
  vst1q_f16(out_ptr + 8 * 5, acc5);
  vst1q_f16(out_ptr + 8 * 6, acc6);
  vst1q_f16(out_ptr + 8 * 7, acc7);
}
