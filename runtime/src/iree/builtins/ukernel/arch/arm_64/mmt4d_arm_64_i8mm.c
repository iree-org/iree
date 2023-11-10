// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/arm_64/common_arm_64.h"
#include "iree/builtins/ukernel/arch/arm_64/mmt4d_arm_64_internal.h"

static inline int32x4_t iree_uk_neon_zip1_s32_as_s64(int32x4_t a, int32x4_t b) {
  return vreinterpretq_s32_s64(
      vzip1q_s64(vreinterpretq_s64_s32(a), vreinterpretq_s64_s32(b)));
}

static inline int32x4_t iree_uk_neon_zip2_s32_as_s64(int32x4_t a, int32x4_t b) {
  return vreinterpretq_s32_s64(
      vzip2q_s64(vreinterpretq_s64_s32(a), vreinterpretq_s64_s32(b)));
}

static inline int32x4_t iree_uk_neon_uzp1_s32_as_s64(int32x4_t a, int32x4_t b) {
  return vreinterpretq_s32_s64(
      vuzp1q_s64(vreinterpretq_s64_s32(a), vreinterpretq_s64_s32(b)));
}

static inline int32x4_t iree_uk_neon_uzp2_s32_as_s64(int32x4_t a, int32x4_t b) {
  return vreinterpretq_s32_s64(
      vuzp2q_s64(vreinterpretq_s64_s32(a), vreinterpretq_s64_s32(b)));
}

void iree_uk_mmt4d_tile_s8s8s32_1x8x8_to_8x8x8_arm_64_i8mm(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 8 && iree_uk_is_po2_u32(M0));
  const iree_uk_int8_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;

  // Accumulator 2x2 register tiles.
  int32x4_t acc[4][4];
  const int mtiles = M0 == 1 ? 1 : M0 / 2;
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    // Load row-major accumulator and swizzle into 2x2 register tiles.
    for (int i = 0; i < mtiles; ++i) {
      for (int j = 0; j < 2; ++j) {
        int32x4_t acc_1x4_0 = vld1q_s32(out_ptr + 8 * (2 * i + 0) + 4 * j);
        int32x4_t acc_1x4_1 =
            M0 == 1 ? vdupq_n_s32(0)
                    : vld1q_s32(out_ptr + 8 * (2 * i + 1) + 4 * j);
        acc[i][2 * j + 0] = iree_uk_neon_zip1_s32_as_s64(acc_1x4_0, acc_1x4_1);
        acc[i][2 * j + 1] = iree_uk_neon_zip2_s32_as_s64(acc_1x4_0, acc_1x4_1);
      }
    }
  } else {
    for (int i = 0; i < mtiles; ++i) {
      for (int j = 0; j < 4; ++j) {
        acc[i][j] = vdupq_n_s32(0);
      }
    }
  }

  IREE_UK_ASSUME(params->K >= 1);
  for (int k = 0; k < params->K; ++k) {
    int8x16_t rhs[4];
    for (int i = 0; i < 4; ++i) {
      rhs[i] = vld1q_s8(rhs_ptr + 16 * i);
    }
    rhs_ptr += 64;
    int8x16_t lhs[4];
    if (M0 == 1) {
      int8x8_t lhs8 = vld1_s8(lhs_ptr);
      lhs[0] = vcombine_s8(lhs8, lhs8);
      lhs_ptr += 8;
    } else
      for (int i = 0; i < mtiles; ++i) {
        lhs[i] = vld1q_s8(lhs_ptr);
        lhs_ptr += 16;
      }
    for (int i = 0; i < mtiles; ++i) {
      for (int j = 0; j < 4; ++j) {
        acc[i][j] = vmmlaq_s32(acc[i][j], lhs[i], rhs[j]);
      }
    }
  }

  // Swizzle accumulator 2x2 register tiles back to row-major and store.
  for (int i = 0; i < mtiles; ++i) {
    for (int j = 0; j < 2; ++j) {
      int32x4_t acc_1x4_0 =
          iree_uk_neon_uzp1_s32_as_s64(acc[i][2 * j + 0], acc[i][2 * j + 1]);
      vst1q_s32(out_ptr + 8 * (2 * i + 0) + 4 * j, acc_1x4_0);
      if (M0 > 1) {
        int32x4_t acc_1x4_1 =
            iree_uk_neon_uzp2_s32_as_s64(acc[i][2 * j + 0], acc[i][2 * j + 1]);
        vst1q_s32(out_ptr + 8 * (2 * i + 1) + 4 * j, acc_1x4_1);
      }
    }
  }
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0_1_2_4_8(
    iree_uk_mmt4d_tile_s8s8s32_1x8x8_to_8x8x8_arm_64_i8mm,
    iree_uk_mmt4d_tile_s8s8s32_1x8x8_arm_64_i8mm,
    iree_uk_mmt4d_tile_s8s8s32_2x8x8_arm_64_i8mm,
    iree_uk_mmt4d_tile_s8s8s32_4x8x8_arm_64_i8mm,
    iree_uk_mmt4d_tile_s8s8s32_8x8x8_arm_64_i8mm)
