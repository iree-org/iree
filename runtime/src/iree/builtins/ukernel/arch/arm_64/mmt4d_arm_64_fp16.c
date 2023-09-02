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
  float16x8_t acc[8];
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    for (int i = 0; i < 8; ++i) {
      acc[i] = vld1q_f16(out_ptr + 8 * i);
    }
  } else {
    for (int i = 0; i < 8; ++i) {
      acc[i] = vdupq_n_f16(0);
    }
  }
  IREE_UK_ASSUME(params->K >= 1);
  for (int k = 0; k < params->K; ++k) {
    float16x8_t lhs = vld1q_f16(lhs_ptr);
    lhs_ptr += 8;
    float16x8_t rhs = vld1q_f16(rhs_ptr);
    rhs_ptr += 8;
    acc[0] = vfmaq_lane_f16(acc[0], rhs, vget_low_f16(lhs), 0);
    acc[1] = vfmaq_lane_f16(acc[1], rhs, vget_low_f16(lhs), 1);
    acc[2] = vfmaq_lane_f16(acc[2], rhs, vget_low_f16(lhs), 2);
    acc[3] = vfmaq_lane_f16(acc[3], rhs, vget_low_f16(lhs), 3);
    acc[4] = vfmaq_lane_f16(acc[4], rhs, vget_high_f16(lhs), 0);
    acc[5] = vfmaq_lane_f16(acc[5], rhs, vget_high_f16(lhs), 1);
    acc[6] = vfmaq_lane_f16(acc[6], rhs, vget_high_f16(lhs), 2);
    acc[7] = vfmaq_lane_f16(acc[7], rhs, vget_high_f16(lhs), 3);
  }
  for (int i = 0; i < 8; ++i) {
    vst1q_f16(out_ptr + 8 * i, acc[i]);
  }
}
