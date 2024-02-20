// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/arm_64/common_arm_64.h"
#include "iree/builtins/ukernel/arch/arm_64/mmt4d_arm_64_internal.h"

#define IREE_DEVICE_STANDALONE
#ifndef IREE_DEVICE_STANDALONE
#include <stdio.h>

#define BYTE_TO_BINARY_PATTERN "%c%c%c%c %c%c%c%c"
#define BYTE_TO_BINARY(byte)  \
  ((byte) & 0x80 ? '1' : '0'), \
  ((byte) & 0x40 ? '1' : '0'), \
  ((byte) & 0x20 ? '1' : '0'), \
  ((byte) & 0x10 ? '1' : '0'), \
  ((byte) & 0x08 ? '1' : '0'), \
  ((byte) & 0x04 ? '1' : '0'), \
  ((byte) & 0x02 ? '1' : '0'), \
  ((byte) & 0x01 ? '1' : '0')

#define WORD_TO_BINARY_PATTERN "%c%c%c%c %c%c%c%c %c%c%c%c %c%c%c%c"
#define WORD_TO_BINARY(word)  \
  ((word) & 0x8000 ? '1' : '0'), \
  ((word) & 0x4000 ? '1' : '0'), \
  ((word) & 0x2000 ? '1' : '0'), \
  ((word) & 0x1000 ? '1' : '0'), \
  ((word) & 0x0800 ? '1' : '0'), \
  ((word) & 0x0400 ? '1' : '0'), \
  ((word) & 0x0200 ? '1' : '0'), \
  ((word) & 0x0100 ? '1' : '0'), \
  ((word) & 0x0080 ? '1' : '0'), \
  ((word) & 0x0040 ? '1' : '0'), \
  ((word) & 0x0020 ? '1' : '0'), \
  ((word) & 0x0010 ? '1' : '0'), \
  ((word) & 0x0008 ? '1' : '0'), \
  ((word) & 0x0004 ? '1' : '0'), \
  ((word) & 0x0002 ? '1' : '0'), \
  ((word) & 0x0001 ? '1' : '0')

#endif

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


static inline void iree_uk_mmt4d_tile_s8s4s32_1x8x16_to_8x8x16_arm_64_i8mm(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 8 && iree_uk_is_po2_u32(M0));
  IREE_UK_ASSERT(!(params->K0 % 16));
  const iree_uk_int8_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;

#ifndef IREE_DEVICE_STANDALONE
  printf("----- Start optimized M: %lld, N: %lld, K: %lld, M0: %d, N0: %d, K0: %d ------\n", params->M, params->N, params->K, M0, params->N0, params->K0);
#endif

  const int8x16_t vmask = vmovq_n_s8(0xF0);
  const int8x8_t vzero = vmov_n_s8(0);

  const int mtiles = M0 == 1 ? 1 : M0 / 2;

  int32x4_t acc[4][4];
  for (int i = 0; i < mtiles; ++i) {
    for (int j = 0; j < 4; ++j) {
      acc[i][j] = vdupq_n_s32(0);
    }
  }

  for (int k = 0; k < params->K; ++k) {
    int8x16_t rhs[2][4];
    for (int i = 0; i < 4; ++i) {
      int8x16_t r = vld1q_s8(rhs_ptr + 16 * i);

      // We unpack int4s into individual int8s. To preserve signedness,
      // int4s are moved to the upper 4-bits of each byte. This has the effect of multiplying each int4 by 2^4 = 16. To compensate, we divide the
      // accumulator values by 16 before storing to memory.
      // This int4 conversion trick is borrowed from the `qd8-f32-qc4w-gemm*`
      // kernels in https://github.com/google/XNNPACK.
      rhs[0][i] = vshlq_n_s8(r, 4);
      rhs[1][i] = vandq_s8(r, vmask);
    }
    rhs_ptr += 64;

    if (M0 == 8) {
      int8x16_t lhs[2][4];
      for (int i = 0; i < 4; ++i) {
        int8x8x2_t lhs_0 = vld2_s8(lhs_ptr + 16 * 2 * i);
        int8x8x2_t lhs_1 = vld2_s8(lhs_ptr + 16 * (2 * i + 1));

        lhs[0][i] = vcombine_s8(lhs_0.val[0], lhs_1.val[0]);
        lhs[1][i] = vcombine_s8(lhs_0.val[1], lhs_1.val[1]);
      }
      lhs_ptr += 128;

      // We unroll to optimize performance.
      acc[0][0] = vmmlaq_s32(acc[0][0], lhs[0][0], rhs[0][0]);
      acc[0][1] = vmmlaq_s32(acc[0][1], lhs[0][0], rhs[0][1]);
      acc[0][2] = vmmlaq_s32(acc[0][2], lhs[0][0], rhs[0][2]);
      acc[0][3] = vmmlaq_s32(acc[0][3], lhs[0][0], rhs[0][3]);

      acc[1][0] = vmmlaq_s32(acc[1][0], lhs[0][1], rhs[0][0]);
      acc[1][1] = vmmlaq_s32(acc[1][1], lhs[0][1], rhs[0][1]);
      acc[1][2] = vmmlaq_s32(acc[1][2], lhs[0][1], rhs[0][2]);
      acc[1][3] = vmmlaq_s32(acc[1][3], lhs[0][1], rhs[0][3]);

      acc[2][0] = vmmlaq_s32(acc[2][0], lhs[0][2], rhs[0][0]);
      acc[2][1] = vmmlaq_s32(acc[2][1], lhs[0][2], rhs[0][1]);
      acc[2][2] = vmmlaq_s32(acc[2][2], lhs[0][2], rhs[0][2]);
      acc[2][3] = vmmlaq_s32(acc[2][3], lhs[0][2], rhs[0][3]);

      acc[3][0] = vmmlaq_s32(acc[3][0], lhs[0][3], rhs[0][0]);
      acc[3][1] = vmmlaq_s32(acc[3][1], lhs[0][3], rhs[0][1]);
      acc[3][2] = vmmlaq_s32(acc[3][2], lhs[0][3], rhs[0][2]);
      acc[3][3] = vmmlaq_s32(acc[3][3], lhs[0][3], rhs[0][3]);

      acc[0][0] = vmmlaq_s32(acc[0][0], lhs[1][0], rhs[1][0]);
      acc[0][1] = vmmlaq_s32(acc[0][1], lhs[1][0], rhs[1][1]);
      acc[0][2] = vmmlaq_s32(acc[0][2], lhs[1][0], rhs[1][2]);
      acc[0][3] = vmmlaq_s32(acc[0][3], lhs[1][0], rhs[1][3]);

      acc[1][0] = vmmlaq_s32(acc[1][0], lhs[1][1], rhs[1][0]);
      acc[1][1] = vmmlaq_s32(acc[1][1], lhs[1][1], rhs[1][1]);
      acc[1][2] = vmmlaq_s32(acc[1][2], lhs[1][1], rhs[1][2]);
      acc[1][3] = vmmlaq_s32(acc[1][3], lhs[1][1], rhs[1][3]);

      acc[2][0] = vmmlaq_s32(acc[2][0], lhs[1][2], rhs[1][0]);
      acc[2][1] = vmmlaq_s32(acc[2][1], lhs[1][2], rhs[1][1]);
      acc[2][2] = vmmlaq_s32(acc[2][2], lhs[1][2], rhs[1][2]);
      acc[2][3] = vmmlaq_s32(acc[2][3], lhs[1][2], rhs[1][3]);

      acc[3][0] = vmmlaq_s32(acc[3][0], lhs[1][3], rhs[1][0]);
      acc[3][1] = vmmlaq_s32(acc[3][1], lhs[1][3], rhs[1][1]);
      acc[3][2] = vmmlaq_s32(acc[3][2], lhs[1][3], rhs[1][2]);
      acc[3][3] = vmmlaq_s32(acc[3][3], lhs[1][3], rhs[1][3]);
    } else if (M0 == 4) {
      int8x16_t lhs[2][2];
      for (int i = 0; i < 2; ++i) {
        int8x8x2_t lhs_0 = vld2_s8(lhs_ptr + 16 * 2 * i);
        int8x8x2_t lhs_1 = vld2_s8(lhs_ptr + 16 * (2 * i + 1));

        lhs[0][i] = vcombine_s8(lhs_0.val[0], lhs_1.val[0]);
        lhs[1][i] = vcombine_s8(lhs_0.val[1], lhs_1.val[1]);
      }
      lhs_ptr += 64;

      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
          acc[i][j] = vmmlaq_s32(acc[i][j], lhs[0][i], rhs[0][j]);
          acc[i][j] = vmmlaq_s32(acc[i][j], lhs[1][i], rhs[1][j]);
        }
      }
    } else {  // M0 = 2, M0 = 1.
      int8x16_t lhs[2];
      if (M0 == 2) {
        int8x8x2_t lhs_uzp[2];
        for (int i = 0; i < 2; ++i) {
          lhs_uzp[i] = vld2_s8(lhs_ptr + 16 * i);
        }
        lhs_ptr += 32;
        lhs[0] = vcombine_s8(lhs_uzp[0].val[0], lhs_uzp[1].val[0]);
        lhs[1] = vcombine_s8(lhs_uzp[0].val[1], lhs_uzp[1].val[1]);
      } else {
        // M0 == 1.
        int8x8x2_t lhs_uzp = vld2_s8(lhs_ptr);
        lhs_ptr += 16;
        lhs[0] = vcombine_s8(lhs_uzp.val[0], vzero);
        lhs[1] = vcombine_s8(lhs_uzp.val[1], vzero);
      }

      for (int i = 0; i < 4; ++i) {
        acc[0][i] = vmmlaq_s32(acc[0][i], lhs[0], rhs[0][i]);
        acc[0][i] = vmmlaq_s32(acc[0][i], lhs[1], rhs[1][i]);
      }
    }
  }

  // Swizzle accumulator 2x2 register tiles back to row-major and store.
  for (int i = 0; i < mtiles; ++i) {
    for (int j = 0; j < 2; ++j) {
      acc[i][2 * j + 0] = vshrq_n_s32(acc[i][2 * j + 0], 4);
      acc[i][2 * j + 1] = vshrq_n_s32(acc[i][2 * j + 1], 4);

      int32x4_t acc_1x4_0 =
          iree_uk_neon_uzp1_s32_as_s64(acc[i][2 * j + 0], acc[i][2 * j + 1]);
      if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
        int32x4_t existing_acc = vld1q_s32(out_ptr + 8 * (2 * i + 0) + 4 * j);
        acc_1x4_0 = vaddq_s32(acc_1x4_0, existing_acc);
      }
      vst1q_s32(out_ptr + 8 * (2 * i + 0) + 4 * j, acc_1x4_0);

#ifndef IREE_DEVICE_STANDALONE
      int acc_0 = 2 * j;
      int acc_1 = 2 * j + 1;
      for (int k = 0; k < 4; ++k) {
        printf("acc[%d][%d][%d]: %d\n:", i, acc_0, k, acc[i][acc_0][k]);
      }
      for (int k = 0; k < 4; ++k) {
        printf("acc[%d][%d][%d]: %d\n:", i, acc_1, k, acc[i][acc_1][k]);
      }
      for (int k = 0; k < 4; ++k) {
        printf("acc_1x4_0[%d]: %d\n", k, acc_1x4_0[k]);
      }
#endif

      if (M0 > 1) {
        int32x4_t acc_1x4_1 =
            iree_uk_neon_uzp2_s32_as_s64(acc[i][2 * j + 0], acc[i][2 * j + 1]);
        if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
          int32x4_t existing_acc = vld1q_s32(out_ptr + 8 * (2 * i + 1) + 4 * j);
          acc_1x4_1 = vaddq_s32(acc_1x4_1, existing_acc);
        }
#ifndef IREE_DEVICE_STANDALONE
        for (int k = 0; k < 4; ++k) {
          printf("acc_1x4_1[%d]: %d\n", k, acc_1x4_1[k]);
        }
#endif
        vst1q_s32(out_ptr + 8 * (2 * i + 1) + 4 * j, acc_1x4_1);
      }
    }
  }
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0_1_2_4_8(
    iree_uk_mmt4d_tile_s8s4s32_1x8x16_to_8x8x16_arm_64_i8mm,
    iree_uk_mmt4d_tile_s8s4s32_1x8x16_arm_64_i8mm,
    iree_uk_mmt4d_tile_s8s4s32_2x8x16_arm_64_i8mm,
    iree_uk_mmt4d_tile_s8s4s32_4x8x16_arm_64_i8mm,
    iree_uk_mmt4d_tile_s8s4s32_8x8x16_arm_64_i8mm)
