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

static inline void iree_uk_mmt4d_tile_bf16bf16fXX_1x8x4_to_8x8x4_arm_64_bf16(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, iree_uk_type_t acc_type, int M0) {
  IREE_UK_ASSERT(acc_type == IREE_UK_TYPE_FLOAT_32 ||
                 acc_type == IREE_UK_TYPE_BFLOAT_16);
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 8 && iree_uk_is_po2_u32(M0));
  const bfloat16_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const bfloat16_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;

  // Accumulator 2x2 register tiles.
  float32x4_t acc[4][4];
  const int mtiles = M0 == 1 ? 1 : M0 / 2;
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    // Load row-major accumulator and swizzle into 2x2 register tiles.
    for (int i = 0; i < mtiles; ++i) {
      for (int j = 0; j < 2; ++j) {
        float32x4_t acc_1x4_0, acc_1x4_1;
        if (acc_type == IREE_UK_TYPE_FLOAT_32) {
          const float* IREE_UK_RESTRICT out_ptr = out_tile;
          acc_1x4_0 = vld1q_f32(out_ptr + 8 * (2 * i + 0) + 4 * j);
          acc_1x4_1 = M0 == 1 ? vdupq_n_f32(0)
                              : vld1q_f32(out_ptr + 8 * (2 * i + 1) + 4 * j);
        } else {
          const bfloat16_t* IREE_UK_RESTRICT out_ptr = out_tile;
          acc_1x4_0 =
              vcvt_f32_bf16(vld1_bf16(out_ptr + 8 * (2 * i + 0) + 4 * j));
          acc_1x4_1 =
              M0 == 1
                  ? vdupq_n_f32(0)
                  : vcvt_f32_bf16(vld1_bf16(out_ptr + 8 * (2 * i + 1) + 4 * j));
        }
        acc[i][2 * j + 0] = iree_uk_neon_zip1_f32_as_s64(acc_1x4_0, acc_1x4_1);
        acc[i][2 * j + 1] = iree_uk_neon_zip2_f32_as_s64(acc_1x4_0, acc_1x4_1);
      }
    }
  } else {
    for (int i = 0; i < mtiles; ++i) {
      for (int j = 0; j < 4; ++j) {
        acc[i][j] = vdupq_n_f32(0);
      }
    }
  }

  for (int k = 0; k < params->K; ++k) {
    bfloat16x8_t rhs[4];
    for (int i = 0; i < 4; ++i) {
      rhs[i] = vld1q_bf16(rhs_ptr + 8 * i);
    }
    rhs_ptr += 32;
    bfloat16x8_t lhs[4];
    if (M0 == 1) {
      bfloat16x4_t lhs4 = vld1_bf16(lhs_ptr);
      lhs[0] = vcombine_bf16(lhs4, lhs4);
      lhs_ptr += 4;
    } else
      for (int i = 0; i < mtiles; ++i) {
        lhs[i] = vld1q_bf16(lhs_ptr);
        lhs_ptr += 8;
      }
    for (int i = 0; i < mtiles; ++i) {
      for (int j = 0; j < 4; ++j) {
        acc[i][j] = vbfmmlaq_f32(acc[i][j], lhs[i], rhs[j]);
      }
    }
  }

  // Swizzle accumulator 2x2 register tiles back to row-major and store.
  for (int i = 0; i < mtiles; ++i) {
    for (int j = 0; j < 2; ++j) {
      float32x4_t acc_1x4_0 =
          iree_uk_neon_uzp1_f32_as_s64(acc[i][2 * j + 0], acc[i][2 * j + 1]);
      float32x4_t acc_1x4_1 =
          iree_uk_neon_uzp2_f32_as_s64(acc[i][2 * j + 0], acc[i][2 * j + 1]);
      if (acc_type == IREE_UK_TYPE_FLOAT_32) {
        float* IREE_UK_RESTRICT out_ptr = out_tile;
        vst1q_f32(out_ptr + 8 * (2 * i + 0) + 4 * j, acc_1x4_0);
        if (M0 > 1) {
          vst1q_f32(out_ptr + 8 * (2 * i + 1) + 4 * j, acc_1x4_1);
        }
      } else {
        bfloat16_t* IREE_UK_RESTRICT out_ptr = out_tile;
        vst1_bf16(out_ptr + 8 * (2 * i + 0) + 4 * j, vcvt_bf16_f32(acc_1x4_0));
        if (M0 > 1) {
          vst1_bf16(out_ptr + 8 * (2 * i + 1) + 4 * j,
                    vcvt_bf16_f32(acc_1x4_1));
        }
      }
    }
  }
}

static inline void iree_uk_mmt4d_tile_bf16bf16f32_1x8x4_to_8x8x4_arm_64_bf16(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  iree_uk_mmt4d_tile_bf16bf16fXX_1x8x4_to_8x8x4_arm_64_bf16(
      out_tile, lhs_panel, rhs_panel, params, IREE_UK_TYPE_FLOAT_32, M0);
}

static inline void iree_uk_mmt4d_tile_bf16bf16bf16_1x8x4_to_8x8x4_arm_64_bf16(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  iree_uk_mmt4d_tile_bf16bf16fXX_1x8x4_to_8x8x4_arm_64_bf16(
      out_tile, lhs_panel, rhs_panel, params, IREE_UK_TYPE_BFLOAT_16, M0);
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0_1_2_4_8(
    iree_uk_mmt4d_tile_bf16bf16f32_1x8x4_to_8x8x4_arm_64_bf16,
    iree_uk_mmt4d_tile_bf16bf16f32_1x8x4_arm_64_bf16,
    iree_uk_mmt4d_tile_bf16bf16f32_2x8x4_arm_64_bf16,
    iree_uk_mmt4d_tile_bf16bf16f32_4x8x4_arm_64_bf16,
    iree_uk_mmt4d_tile_bf16bf16f32_8x8x4_arm_64_bf16)

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0_1_2_4_8(
    iree_uk_mmt4d_tile_bf16bf16bf16_1x8x4_to_8x8x4_arm_64_bf16,
    iree_uk_mmt4d_tile_bf16bf16bf16_1x8x4_arm_64_bf16,
    iree_uk_mmt4d_tile_bf16bf16bf16_2x8x4_arm_64_bf16,
    iree_uk_mmt4d_tile_bf16bf16bf16_4x8x4_arm_64_bf16,
    iree_uk_mmt4d_tile_bf16bf16bf16_8x8x4_arm_64_bf16)
