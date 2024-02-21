// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/arm_64/common_arm_64.h"
#include "iree/builtins/ukernel/arch/arm_64/mmt4d_arm_64_internal.h"

static inline void iree_uk_mmt4d_tile_s8s8s32_1x8x4_to_8x8x4_arm_64_dotprod(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 8 && iree_uk_is_po2_u32(M0));
  const iree_uk_int8_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;
  int32x4_t acc[16];
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    for (int i = 0; i < 2 * M0; ++i) {
      acc[i] = vld1q_s32(out_ptr + 4 * i);
    }
  } else {
    for (int i = 0; i < 2 * M0; ++i) {
      acc[i] = vdupq_n_s32(0);
    }
  }
  for (int k = 0; k < params->K; ++k) {
    int8x16_t rhs[2];
    for (int i = 0; i < 2; ++i) {
      rhs[i] = vld1q_s8(rhs_ptr + 16 * i);
    }
    rhs_ptr += 32;
    int8x16_t lhs[2];
    if (M0 == 1) {
      lhs[0] = vdupq_n_s8(0);
      lhs[0] = vreinterpretq_s8_s32(vld1q_lane_s32(
          (const int32_t*)lhs_ptr, vreinterpretq_s32_s8(lhs[0]), 0));
    } else if (M0 == 2) {
      lhs[0] = vcombine_s8(vld1_s8(lhs_ptr), vdup_n_s8(0));
    } else
      for (int i = 0; i < 2; ++i) {
        lhs[i] = vld1q_s8(lhs_ptr + 16 * i);
      }
    lhs_ptr += 4 * M0;
    acc[0] = vdotq_lane_s32(acc[0], rhs[0], vget_low_s8(lhs[0]), 0);
    acc[1] = vdotq_lane_s32(acc[1], rhs[1], vget_low_s8(lhs[0]), 0);
    if (M0 == 1) continue;
    acc[2] = vdotq_lane_s32(acc[2], rhs[0], vget_low_s8(lhs[0]), 1);
    acc[3] = vdotq_lane_s32(acc[3], rhs[1], vget_low_s8(lhs[0]), 1);
    if (M0 == 2) continue;
    acc[4] = vdotq_lane_s32(acc[4], rhs[0], vget_high_s8(lhs[0]), 0);
    acc[5] = vdotq_lane_s32(acc[5], rhs[1], vget_high_s8(lhs[0]), 0);
    acc[6] = vdotq_lane_s32(acc[6], rhs[0], vget_high_s8(lhs[0]), 1);
    acc[7] = vdotq_lane_s32(acc[7], rhs[1], vget_high_s8(lhs[0]), 1);
    if (M0 == 4) continue;
    acc[8] = vdotq_lane_s32(acc[8], rhs[0], vget_low_s8(lhs[1]), 0);
    acc[9] = vdotq_lane_s32(acc[9], rhs[1], vget_low_s8(lhs[1]), 0);
    acc[10] = vdotq_lane_s32(acc[10], rhs[0], vget_low_s8(lhs[1]), 1);
    acc[11] = vdotq_lane_s32(acc[11], rhs[1], vget_low_s8(lhs[1]), 1);
    acc[12] = vdotq_lane_s32(acc[12], rhs[0], vget_high_s8(lhs[1]), 0);
    acc[13] = vdotq_lane_s32(acc[13], rhs[1], vget_high_s8(lhs[1]), 0);
    acc[14] = vdotq_lane_s32(acc[14], rhs[0], vget_high_s8(lhs[1]), 1);
    acc[15] = vdotq_lane_s32(acc[15], rhs[1], vget_high_s8(lhs[1]), 1);
  }

  for (int i = 0; i < 2 * M0; ++i) {
    vst1q_s32(out_ptr + 4 * i, acc[i]);
  }
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0_1_2_4_8(
    iree_uk_mmt4d_tile_s8s8s32_1x8x4_to_8x8x4_arm_64_dotprod,
    iree_uk_mmt4d_tile_s8s8s32_1x8x4_arm_64_dotprod,
    iree_uk_mmt4d_tile_s8s8s32_2x8x4_arm_64_dotprod,
    iree_uk_mmt4d_tile_s8s8s32_4x8x4_arm_64_dotprod,
    iree_uk_mmt4d_tile_s8s8s32_8x8x4_arm_64_dotprod)

static inline void iree_uk_mmt4d_tile_s8s4s32_1x8x8_to_8x8x8_arm_64_dotprod(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 8 && iree_uk_is_po2_u32(M0));
  const iree_uk_int8_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;

  const int8x16_t vmask = vmovq_n_s8(0xF0);
  const int8x8_t vzero = vmov_n_s8(0);

  int32x4_t acc[16];
  for (int i = 0; i < 16; i++) {
    // We start with zero accumulators and add the value of *out_ptr later.
    // This is required for the int4 left shift described later.
    acc[i] = vdupq_n_s32(0);
  }

  for (int k = 0; k < params->K; ++k) {
    int8x16_t rhs[4];
    for (int i = 0; i < 2; i++) {
      int8x16_t r = vld1q_s8(rhs_ptr);
      rhs_ptr += 16;
      // We unpack int4s into individual int8s. To preserve signedness,
      // int4s are moved to the upper 4-bits of each byte. This has the effect
      // of multiplying each int4 by 2^4 = 16. To compensate, we divide the
      // accumulator values by 16 before storing to memory.
      // This int4 conversion trick is borrowed from the `qd8-f32-qc4w-gemm*`
      // kernels in https://github.com/google/XNNPACK.
      rhs[i + 0] = vshlq_n_s8(r, 4);
      rhs[i + 2] = vandq_s8(r, vmask);
    }

    if (M0 == 8) {
      // 8x2 * 2x8 -> 8x8.
      int8x16x2_t lhs_uzp_0 = vld2q_s8(lhs_ptr);
      lhs_ptr += 32;
      int8x16x2_t lhs_uzp_1 = vld2q_s8(lhs_ptr);
      lhs_ptr += 32;

      int8x8_t lhs[8];
      lhs[0] = vget_low_s8(lhs_uzp_0.val[0]);
      lhs[1] = vget_high_s8(lhs_uzp_0.val[0]);
      lhs[2] = vget_low_s8(lhs_uzp_1.val[0]);
      lhs[3] = vget_high_s8(lhs_uzp_1.val[0]);
      lhs[4] = vget_low_s8(lhs_uzp_0.val[1]);
      lhs[5] = vget_high_s8(lhs_uzp_0.val[1]);
      lhs[6] = vget_low_s8(lhs_uzp_1.val[1]);
      lhs[7] = vget_high_s8(lhs_uzp_1.val[1]);

      for (int i = 0; i < 2; i++) {
        acc[0] = vdotq_lane_s32(acc[0], rhs[2 * i + 0], lhs[4 * i + 0], 0);
        acc[1] = vdotq_lane_s32(acc[1], rhs[2 * i + 1], lhs[4 * i + 0], 0);
        acc[2] = vdotq_lane_s32(acc[2], rhs[2 * i + 0], lhs[4 * i + 0], 1);
        acc[3] = vdotq_lane_s32(acc[3], rhs[2 * i + 1], lhs[4 * i + 0], 1);

        acc[4] = vdotq_lane_s32(acc[4], rhs[2 * i + 0], lhs[4 * i + 1], 0);
        acc[5] = vdotq_lane_s32(acc[5], rhs[2 * i + 1], lhs[4 * i + 1], 0);
        acc[6] = vdotq_lane_s32(acc[6], rhs[2 * i + 0], lhs[4 * i + 1], 1);
        acc[7] = vdotq_lane_s32(acc[7], rhs[2 * i + 1], lhs[4 * i + 1], 1);

        acc[8] = vdotq_lane_s32(acc[8], rhs[2 * i + 0], lhs[4 * i + 2], 0);
        acc[9] = vdotq_lane_s32(acc[9], rhs[2 * i + 1], lhs[4 * i + 2], 0);
        acc[10] = vdotq_lane_s32(acc[10], rhs[2 * i + 0], lhs[4 * i + 2], 1);
        acc[11] = vdotq_lane_s32(acc[11], rhs[2 * i + 1], lhs[4 * i + 2], 1);

        acc[12] = vdotq_lane_s32(acc[12], rhs[2 * i + 0], lhs[4 * i + 3], 0);
        acc[13] = vdotq_lane_s32(acc[13], rhs[2 * i + 1], lhs[4 * i + 3], 0);
        acc[14] = vdotq_lane_s32(acc[14], rhs[2 * i + 0], lhs[4 * i + 3], 1);
        acc[15] = vdotq_lane_s32(acc[15], rhs[2 * i + 1], lhs[4 * i + 3], 1);
      }
    } else if (M0 == 4) {
      // 4x2 * 2x8 -> 4x8.
      int8x16x2_t lhs_uzp = vld2q_s8(lhs_ptr);
      lhs_ptr += 32;

      int8x8_t lhs[4];
      lhs[0] = vget_low_s8(lhs_uzp.val[0]);
      lhs[1] = vget_high_s8(lhs_uzp.val[0]);
      lhs[2] = vget_low_s8(lhs_uzp.val[1]);
      lhs[3] = vget_high_s8(lhs_uzp.val[1]);

      for (int i = 0; i < 2; i++) {
        acc[0] = vdotq_lane_s32(acc[0], rhs[2 * i + 0], lhs[2 * i + 0], 0);
        acc[1] = vdotq_lane_s32(acc[1], rhs[2 * i + 1], lhs[2 * i + 0], 0);
        acc[2] = vdotq_lane_s32(acc[2], rhs[2 * i + 0], lhs[2 * i + 0], 1);
        acc[3] = vdotq_lane_s32(acc[3], rhs[2 * i + 1], lhs[2 * i + 0], 1);

        acc[4] = vdotq_lane_s32(acc[4], rhs[2 * i + 0], lhs[2 * i + 1], 0);
        acc[5] = vdotq_lane_s32(acc[5], rhs[2 * i + 1], lhs[2 * i + 1], 0);
        acc[6] = vdotq_lane_s32(acc[6], rhs[2 * i + 0], lhs[2 * i + 1], 1);
        acc[7] = vdotq_lane_s32(acc[7], rhs[2 * i + 1], lhs[2 * i + 1], 1);
      }
    } else if (M0 == 2) {
      // 2x2 * 2x8 -> 2x8.
      int8x8x2_t lhs_uzp = vld2_s8(lhs_ptr);
      lhs_ptr += 16;

      for (int i = 0; i < 2; i++) {
        acc[0] = vdotq_lane_s32(acc[0], rhs[2 * i + 0], lhs_uzp.val[i], 0);
        acc[1] = vdotq_lane_s32(acc[1], rhs[2 * i + 1], lhs_uzp.val[i], 0);
        acc[2] = vdotq_lane_s32(acc[2], rhs[2 * i + 0], lhs_uzp.val[i], 1);
        acc[3] = vdotq_lane_s32(acc[3], rhs[2 * i + 1], lhs_uzp.val[i], 1);
      }
    } else if (M0 == 1) {
      // 1x2 * 2x8 -> 1x8.
      int8x8_t lhs = vld1_s8(lhs_ptr);
      lhs_ptr += 8;
      int8x8x2_t lhs_uzp = vuzp_s8(lhs, vzero);

      acc[0] = vdotq_lane_s32(acc[0], rhs[0], lhs_uzp.val[0], 0);
      acc[1] = vdotq_lane_s32(acc[1], rhs[1], lhs_uzp.val[0], 0);
      acc[0] = vdotq_lane_s32(acc[0], rhs[2], lhs_uzp.val[1], 0);
      acc[1] = vdotq_lane_s32(acc[1], rhs[3], lhs_uzp.val[1], 0);
    }
  }

  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    for (int i = 0; i < 2 * M0; i++) {
      int32x4_t existing_acc = vld1q_s32(out_ptr);
      acc[i] = vsraq_n_s32(existing_acc, acc[i], 4);
      vst1q_s32(out_ptr, acc[i]);
      out_ptr += 4;
    }
  } else {
    for (int i = 0; i < 2 * M0; i++) {
      acc[i] = vshrq_n_s32(acc[i], 4);
      vst1q_s32(out_ptr, acc[i]);
      out_ptr += 4;
    }
  }
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0_1_2_4_8(
    iree_uk_mmt4d_tile_s8s4s32_1x8x8_to_8x8x8_arm_64_dotprod,
    iree_uk_mmt4d_tile_s8s4s32_1x8x8_arm_64_dotprod,
    iree_uk_mmt4d_tile_s8s4s32_2x8x8_arm_64_dotprod,
    iree_uk_mmt4d_tile_s8s4s32_4x8x8_arm_64_dotprod,
    iree_uk_mmt4d_tile_s8s4s32_8x8x8_arm_64_dotprod)
