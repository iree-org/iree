// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/arm_64/common_arm_64.h"
#include "iree/builtins/ukernel/arch/arm_64/mmt4d_arm_64_internal.h"

static inline float32x4_t iree_uk_neon_zip1_f32_as_s64(float32x4_t a,
                                                       float32x4_t b) {
  return vreinterpretq_f32_s64(
      vzip1q_s64(vreinterpretq_s64_f32(a), vreinterpretq_s64_f32(b)));
}

static inline float32x4_t iree_uk_neon_zip2_f32_as_s64(float32x4_t a,
                                                       float32x4_t b) {
  return vreinterpretq_f32_s64(
      vzip2q_s64(vreinterpretq_s64_f32(a), vreinterpretq_s64_f32(b)));
}

static inline float32x4_t iree_uk_neon_uzp1_f32_as_s64(float32x4_t a,
                                                       float32x4_t b) {
  return vreinterpretq_f32_s64(
      vuzp1q_s64(vreinterpretq_s64_f32(a), vreinterpretq_s64_f32(b)));
}

static inline float32x4_t iree_uk_neon_uzp2_f32_as_s64(float32x4_t a,
                                                       float32x4_t b) {
  return vreinterpretq_f32_s64(
      vuzp2q_s64(vreinterpretq_s64_f32(a), vreinterpretq_s64_f32(b)));
}

void iree_uk_mmt4d_tile_bf16bf16f32_8x8x4_arm_64_bf16(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  const bfloat16_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const bfloat16_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  float* IREE_UK_RESTRICT out_ptr = out_tile;
  // Accumulator 2x2 register tiles.
  float32x4_t acc[4][4];
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    // Load row-major accumulator and swizzle into 2x2 register tiles.
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 2; ++j) {
        float32x4_t acc_1x4_0 = vld1q_f32(out_ptr + 8 * (2 * i + 0) + 4 * j);
        float32x4_t acc_1x4_1 = vld1q_f32(out_ptr + 8 * (2 * i + 1) + 4 * j);
        acc[i][2 * j + 0] = iree_uk_neon_zip1_f32_as_s64(acc_1x4_0, acc_1x4_1);
        acc[i][2 * j + 1] = iree_uk_neon_zip2_f32_as_s64(acc_1x4_0, acc_1x4_1);
      }
    }
  } else {
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        acc[i][j] = vdupq_n_f32(0);
      }
    }
  }

  IREE_UK_ASSUME(params->K >= 1);
  for (int k = 0; k < params->K; ++k) {
    bfloat16x8_t lhs[4];
    bfloat16x8_t rhs[4];
    for (int i = 0; i < 4; ++i) {
      lhs[i] = vld1q_bf16(lhs_ptr + 8 * i);
      rhs[i] = vld1q_bf16(rhs_ptr + 8 * i);
    }
    lhs_ptr += 32;
    rhs_ptr += 32;
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        acc[i][j] = vbfmmlaq_f32(acc[i][j], lhs[i], rhs[j]);
      }
    }
  }

  // Swizzle accumulator 2x2 register tiles back to row-major and store.
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 2; ++j) {
      float32x4_t acc_1x4_0 =
          iree_uk_neon_uzp1_f32_as_s64(acc[i][2 * j + 0], acc[i][2 * j + 1]);
      float32x4_t acc_1x4_1 =
          iree_uk_neon_uzp2_f32_as_s64(acc[i][2 * j + 0], acc[i][2 * j + 1]);
      vst1q_f32(out_ptr + 8 * (2 * i + 0) + 4 * j, acc_1x4_0);
      vst1q_f32(out_ptr + 8 * (2 * i + 1) + 4 * j, acc_1x4_1);
    }
  }
}
