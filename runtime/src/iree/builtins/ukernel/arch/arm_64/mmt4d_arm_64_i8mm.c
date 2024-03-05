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

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_s8s8s32_1x8x8_to_8x8x8_arm_64_i8mm(
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
    IREE_UK_UNROLL for (int i = 0; i < mtiles; ++i) {
      IREE_UK_UNROLL for (int j = 0; j < 2; ++j) {
        int32x4_t acc_1x4_0 = vld1q_s32(out_ptr + 8 * (2 * i + 0) + 4 * j);
        int32x4_t acc_1x4_1 =
            M0 == 1 ? vdupq_n_s32(0)
                    : vld1q_s32(out_ptr + 8 * (2 * i + 1) + 4 * j);
        acc[i][2 * j + 0] = iree_uk_neon_zip1_s32_as_s64(acc_1x4_0, acc_1x4_1);
        acc[i][2 * j + 1] = iree_uk_neon_zip2_s32_as_s64(acc_1x4_0, acc_1x4_1);
      }
    }
  } else {
    IREE_UK_UNROLL for (int i = 0; i < mtiles; ++i) {
      IREE_UK_UNROLL for (int j = 0; j < 4; ++j) { acc[i][j] = vdupq_n_s32(0); }
    }
  }
  for (int k = 0; k < params->K; ++k) {
    int8x16_t rhs[4];
    IREE_UK_UNROLL for (int i = 0; i < 4; ++i) {
      rhs[i] = vld1q_s8(rhs_ptr + 16 * i);
    }
    rhs_ptr += 64;
    int8x16_t lhs[4];
    if (M0 == 1) {
      int8x8_t lhs8 = vld1_s8(lhs_ptr);
      lhs[0] = vcombine_s8(lhs8, lhs8);
      lhs_ptr += 8;
    } else
      IREE_UK_UNROLL for (int i = 0; i < mtiles; ++i) {
        lhs[i] = vld1q_s8(lhs_ptr);
        lhs_ptr += 16;
      }
    IREE_UK_UNROLL for (int i = 0; i < mtiles; ++i) {
      IREE_UK_UNROLL for (int j = 0; j < 4; ++j) {
        acc[i][j] = vmmlaq_s32(acc[i][j], lhs[i], rhs[j]);
      }
    }
  }

  // Swizzle accumulator 2x2 register tiles back to row-major and store.
  IREE_UK_UNROLL for (int i = 0; i < mtiles; ++i) {
    IREE_UK_UNROLL for (int j = 0; j < 2; ++j) {
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

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1x8x8_to_8x8x8_arm_64_i8mm,
    iree_uk_mmt4d_tile_s8s8s32_1x8x8_arm_64_i8mm, 1)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1x8x8_to_8x8x8_arm_64_i8mm,
    iree_uk_mmt4d_tile_s8s8s32_2x8x8_arm_64_i8mm, 2)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1x8x8_to_8x8x8_arm_64_i8mm,
    iree_uk_mmt4d_tile_s8s8s32_4x8x8_arm_64_i8mm, 4)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1x8x8_to_8x8x8_arm_64_i8mm,
    iree_uk_mmt4d_tile_s8s8s32_8x8x8_arm_64_i8mm, 8)

// In the s8s4s32 kernels below, we unpack int4s into individual int8s.
// To preserve signedness, int4s are moved to the upper 4-bits of each byte.
// This has the effect of multiplying each int4 by 2^4 = 16. To compensate,
// we divide the accumulator values by 16 before storing to memory.
// This int4 conversion trick is borrowed from the `qd8-f32-qc4w-gemm*`
// kernels in https://github.com/google/XNNPACK.

IREE_UK_ATTRIBUTE_ALWAYS_INLINE inline void
iree_uk_mmt4d_tile_s8s4s32_1x8x16_arm_64_i8mm(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  IREE_UK_ASSERT(M0 == 1 && iree_uk_is_po2_u32(M0));
  IREE_UK_ASSERT(!(params->K0 % 16));
  const iree_uk_int8_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;

  const int8x16_t vmask = vmovq_n_s8(0xF0);
  const int8x8_t vzero = vmov_n_s8(0);

  int32x4_t acc[4];
  IREE_UK_UNROLL for (int i = 0; i < 4; i++) {
    // We start with zero accumulators and add the value of *out_ptr later.
    // This is required for the int4 left shift described above.
    acc[i] = vdupq_n_s32(0);
  }

  for (int k = 0; k < params->K; ++k) {
    int8x16_t rhs[2][4];
    IREE_UK_UNROLL for (int i = 0; i < 4; i++) {
      int8x16_t r = vld1q_s8(rhs_ptr + 16 * i);
      rhs[0][i] = vshlq_n_s8(r, 4);
      rhs[1][i] = vandq_s8(r, vmask);
    }
    rhs_ptr += 64;

    int8x16_t lhs[2];
    int8x8x2_t lhs_uzp = vld2_s8(lhs_ptr);
    lhs_ptr += 16;
    lhs[0] = vcombine_s8(lhs_uzp.val[0], vzero);
    lhs[1] = vcombine_s8(lhs_uzp.val[1], vzero);

    IREE_UK_UNROLL for (int i = 0; i < 4; i++) {
      acc[i] = vmmlaq_s32(acc[i], lhs[0], rhs[0][i]);
      acc[i] = vmmlaq_s32(acc[i], lhs[1], rhs[1][i]);
    }
  }

  IREE_UK_UNROLL for (int j = 0; j < 2; j++) {
    acc[2 * j + 0] = vshrq_n_s32(acc[2 * j + 0], 4);
    acc[2 * j + 1] = vshrq_n_s32(acc[2 * j + 1], 4);

    int32x4_t acc_1x4_0 =
        iree_uk_neon_uzp1_s32_as_s64(acc[2 * j + 0], acc[2 * j + 1]);
    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
      int32x4_t existing_acc = vld1q_s32(out_ptr + 4 * j);
      acc_1x4_0 = vaddq_s32(acc_1x4_0, existing_acc);
    }
    vst1q_s32(out_ptr + 4 * j, acc_1x4_0);
  }
}

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_s8s4s32_2x8x16_to_8x8x16_arm_64_i8mm(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  IREE_UK_ASSERT(M0 >= 2 && M0 <= 8 && iree_uk_is_po2_u32(M0));
  IREE_UK_ASSERT(!(params->K0 % 16));
  const iree_uk_int8_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;

  const int8x16_t vmask = vmovq_n_s8(0xF0);
  const int mtiles = M0 / 2;

  int32x4_t acc[4][4];
  for (int i = 0; i < mtiles; i++) {
    IREE_UK_UNROLL for (int j = 0; j < 4; j++) {
      // We start with zero accumulators and add the value of *out_ptr later.
      // This is required for the int4 left shift described above.
      acc[i][j] = vdupq_n_s32(0);
    }
  }

  for (int k = 0; k < params->K; ++k) {
    int8x16_t rhs[2][4];
    IREE_UK_UNROLL for (int i = 0; i < 4; i++) {
      int8x16_t r = vld1q_s8(rhs_ptr + 16 * i);
      rhs[0][i] = vshlq_n_s8(r, 4);
      rhs[1][i] = vandq_s8(r, vmask);
    }
    rhs_ptr += 64;

    int8x16_t lhs[2][4];
    if (M0 == 2) {
      int8x8x2_t lhs_uzp[2];
      IREE_UK_UNROLL for (int i = 0; i < 2; i++) {
        lhs_uzp[i] = vld2_s8(lhs_ptr + 16 * i);
      }
      lhs[0][0] = vcombine_s8(lhs_uzp[0].val[0], lhs_uzp[1].val[0]);
      lhs[1][0] = vcombine_s8(lhs_uzp[0].val[1], lhs_uzp[1].val[1]);
    } else {
      for (int i = 0; i < mtiles; i++) {
        int8x8x2_t lhs_0 = vld2_s8(lhs_ptr + 16 * 2 * i);
        int8x8x2_t lhs_1 = vld2_s8(lhs_ptr + 16 * (2 * i + 1));
        lhs[0][i] = vcombine_s8(lhs_0.val[0], lhs_1.val[0]);
        lhs[1][i] = vcombine_s8(lhs_0.val[1], lhs_1.val[1]);
      }
    }
    lhs_ptr += 32 * mtiles;

    for (int i = 0; i < mtiles; i++) {
      IREE_UK_UNROLL for (int j = 0; j < 4; j++) {
        IREE_UK_UNROLL for (int m = 0; m < 2; m++) {
          acc[i][j] = vmmlaq_s32(acc[i][j], lhs[m][i], rhs[m][j]);
        }
      }
    }
  }

  // Swizzle accumulator 2x2 register tiles back to row-major and store.
  for (int i = 0; i < mtiles; i++) {
    IREE_UK_UNROLL for (int j = 0; j < 2; j++) {
      acc[i][2 * j + 0] = vshrq_n_s32(acc[i][2 * j + 0], 4);
      acc[i][2 * j + 1] = vshrq_n_s32(acc[i][2 * j + 1], 4);

      int32x4_t acc_1x4_0 =
          iree_uk_neon_uzp1_s32_as_s64(acc[i][2 * j + 0], acc[i][2 * j + 1]);
      if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
        int32x4_t existing_acc = vld1q_s32(out_ptr + 8 * (2 * i + 0) + 4 * j);
        acc_1x4_0 = vaddq_s32(acc_1x4_0, existing_acc);
      }
      vst1q_s32(out_ptr + 8 * (2 * i + 0) + 4 * j, acc_1x4_0);

      int32x4_t acc_1x4_1 =
          iree_uk_neon_uzp2_s32_as_s64(acc[i][2 * j + 0], acc[i][2 * j + 1]);
      if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
        int32x4_t existing_acc = vld1q_s32(out_ptr + 8 * (2 * i + 1) + 4 * j);
        acc_1x4_1 = vaddq_s32(acc_1x4_1, existing_acc);
      }
      vst1q_s32(out_ptr + 8 * (2 * i + 1) + 4 * j, acc_1x4_1);
    }
  }
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s4s32_2x8x16_to_8x8x16_arm_64_i8mm,
    iree_uk_mmt4d_tile_s8s4s32_2x8x16_arm_64_i8mm, 2)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s4s32_2x8x16_to_8x8x16_arm_64_i8mm,
    iree_uk_mmt4d_tile_s8s4s32_4x8x16_arm_64_i8mm, 4)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s4s32_2x8x16_to_8x8x16_arm_64_i8mm,
    iree_uk_mmt4d_tile_s8s4s32_8x8x16_arm_64_i8mm, 8)
