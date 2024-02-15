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
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 4 && iree_uk_is_po2_u32(M0));
  IREE_UK_ASSERT(!(params->K0 % 4));
  const iree_uk_int8_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;

#ifndef IREE_DEVICE_STANDALONE
  printf("----- Start optimized M: %lld, N: %lld, K: %lld, M0: %d, N0: %d, K0: %d ------\n", params->M, params->N, params->K, M0, params->N0, params->K0);
#endif

#ifndef IREE_DEVICE_STANDALONE
//  for (int i = 0; i < 4; ++i) {
//    for (int j = 0; j < 4; j++) {
//      int32_t val = acc[i][j];
//      printf("init acc[%d][%d]: %d\n", i, j, val);
//    }
//  }
#endif

  const int8x16_t vmask = vmovq_n_s8(0xF0);
  const int8x8_t vzero = vmov_n_s8(0);

  if (M0 == 8) {
    int32x4_t acc[8];
    for (int i = 0; i < 8; i++) {
      acc[i] = vdupq_n_s32(0);
    }

    for (int k = 0; k < params->K; ++k) {
#ifndef IREE_DEVICE_STANDALONE
      for(int i = 0; i < 32; i++) {
        printf("lhs[%d]: %d\n:", i, *(lhs_ptr + i));
      }
#endif
      int8x16x2_t lhs_uzp[2];
      for (int i = 0; i < 2; ++i) {
        lhs_uzp[i] = vld2q_s8(lhs_ptr);
        lhs_ptr += 32;
      }

#ifndef IREE_DEVICE_STANDALONE
//      for (int j = 0; j < 2; j++) {
//        for (int i = 0; i < 16; i++) {
//          printf("lhs_uzp[%d].val[0][%d]: %d\n", j, i, lhs_uzp[j].val[0][i]);
//        }
//        for (int i = 0; i < 16; i++) {
//          printf("lhs_uzp[%d].val[1][%d]: %d\n", j, i, lhs_uzp[j].val[1][i]);
//        }
//      }
#endif

      int8x16_t rhs = vld1q_s8(rhs_ptr);
      rhs_ptr += 16;

      int8x16_t rhs_0 = vshlq_n_s8(rhs, 4);
      int8x16_t rhs_1 = vandq_s8(rhs, vmask);

#ifndef IREE_DEVICE_STANDALONE
      for (int m = 0; m < 16; m++) {
          int8_t rhs_byte = rhs_0[m];
          rhs_byte = (rhs_byte >> 4) & 0x0F;
          if (rhs_byte & 0x08) {
            rhs_byte |= 0xF0;
          }
          printf("rhs_0[%d]: %d, rhs_byte " BYTE_TO_BINARY_PATTERN "\n", m,
                 rhs_byte, BYTE_TO_BINARY(rhs_byte));
        }

        for (int m = 0; m < 16; m++) {
          int8_t rhs_byte = rhs_1[m];
          rhs_byte = (rhs_byte >> 4) & 0x0F;
          if (rhs_byte & 0x08) {
            rhs_byte |= 0xF0;
          }
          printf("rhs_1[%d]: %d, rhs_byte " BYTE_TO_BINARY_PATTERN "\n", m,
                 rhs_byte, BYTE_TO_BINARY(rhs_byte));
        }
#endif

#ifndef IREE_DEVICE_STANDALONE
        for (int j = 0; j < 8; j++) {
          printf("vget_low_s8(lhs_uzp[%d].val[0][%d]: %d\n", i, j, vget_low_s8(lhs_uzp[i].val[0])[j]);
        }
        for (int j = 0; j < 8; j++) {
          printf("vget_low_s8(lhs_uzp[%d].val[1][%d]: %d\n", i, j, vget_low_s8(lhs_uzp[i].val[1])[j]);
        }
        for (int j = 0; j < 8; j++) {
          printf("vget_high_s8(lhs_uzp[%d].val[0][%d]: %d\n", i, j, vget_high_s8(lhs_uzp[i].val[0])[j]);
        }
        for (int j = 0; j < 8; j++) {
          printf("vget_high_s8(lhs_uzp[%d].val[1][%d]: %d\n", i, j, vget_high_s8(lhs_uzp[i].val[1])[j]);
        }
#endif

        int8x8_t lhs_0 = vget_low_s8(lhs_uzp[0].val[0]);
        int8x8_t lhs_1 = vget_low_s8(lhs_uzp[0].val[1]);
        int8x8_t lhs_2 = vget_high_s8(lhs_uzp[0].val[0]);
        int8x8_t lhs_3 = vget_high_s8(lhs_uzp[0].val[1]);
        int8x8_t lhs_4 = vget_low_s8(lhs_uzp[1].val[0]);
        int8x8_t lhs_5 = vget_low_s8(lhs_uzp[1].val[1]);
        int8x8_t lhs_6 = vget_high_s8(lhs_uzp[1].val[0]);
        int8x8_t lhs_7 = vget_high_s8(lhs_uzp[1].val[1]);

        acc[0] = vdotq_lane_s32(acc[0], rhs_0, lhs_0, 0);
        acc[1] = vdotq_lane_s32(acc[1], rhs_0, lhs_0, 1);
        acc[2] = vdotq_lane_s32(acc[2], rhs_0, lhs_2, 0);
        acc[3] = vdotq_lane_s32(acc[3], rhs_0, lhs_2, 1);
        acc[4] = vdotq_lane_s32(acc[4], rhs_0, lhs_4, 0);
        acc[5] = vdotq_lane_s32(acc[5], rhs_0, lhs_4, 1);
        acc[6] = vdotq_lane_s32(acc[6], rhs_0, lhs_6, 0);
        acc[7] = vdotq_lane_s32(acc[7], rhs_0, lhs_6, 1);

        acc[0] = vdotq_lane_s32(acc[0], rhs_1, lhs_1, 0);
        acc[1] = vdotq_lane_s32(acc[1], rhs_1, lhs_1, 1);
        acc[2] = vdotq_lane_s32(acc[2], rhs_1, lhs_3, 0);
        acc[3] = vdotq_lane_s32(acc[3], rhs_1, lhs_3, 1);
        acc[4] = vdotq_lane_s32(acc[4], rhs_1, lhs_5, 0);
        acc[5] = vdotq_lane_s32(acc[5], rhs_1, lhs_5, 1);
        acc[6] = vdotq_lane_s32(acc[6], rhs_1, lhs_7, 0);
        acc[7] = vdotq_lane_s32(acc[7], rhs_1, lhs_7, 1);
    }

    for (int i = 0; i < 8; i++) {
      acc[i] = vshrq_n_s32(acc[i], 4);
      if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
        int32x4_t existing_acc = vld1q_s32(out_ptr);
        acc[i] = vaddq_s32(acc[i], existing_acc);
      }
      vst1q_s32(out_ptr, acc[i]);
#ifndef IREE_DEVICE_STANDALONE
      for (int j = 0; j < 4; j++) {
        printf("acc[%d][%d]: %d\n", i, j, acc[i][j]);
      }
#endif
      out_ptr += 4;
    }
  } else if (M0 == 4) {
    int32x4_t acc[4];
    for (int i = 0; i < 4; i++) {
      acc[i] = vdupq_n_s32(0);
    }

    for (int k = 0; k < params->K; ++k) {
      int8x16x2_t lhs_uzp = vld2q_s8(lhs_ptr);
      lhs_ptr += 32;

      int8x16_t rhs = vld1q_s8(rhs_ptr);
      rhs_ptr += 16;

      int8x16_t rhs_0 = vshlq_n_s8(rhs, 4);
      int8x16_t rhs_1 = vandq_s8(rhs, vmask);

      acc[0] = vdotq_lane_s32(acc[0], rhs_0, vget_low_s8(lhs_uzp.val[0]), 0);
      acc[0] = vdotq_lane_s32(acc[0], rhs_1, vget_low_s8(lhs_uzp.val[1]), 0);
      acc[1] = vdotq_lane_s32(acc[1], rhs_0, vget_low_s8(lhs_uzp.val[0]), 1);
      acc[1] = vdotq_lane_s32(acc[1], rhs_1, vget_low_s8(lhs_uzp.val[1]), 1);

      acc[2] = vdotq_lane_s32(acc[2], rhs_0, vget_high_s8(lhs_uzp.val[0]), 0);
      acc[2] = vdotq_lane_s32(acc[2], rhs_1, vget_high_s8(lhs_uzp.val[1]), 0);
      acc[3] = vdotq_lane_s32(acc[3], rhs_0, vget_high_s8(lhs_uzp.val[0]), 1);
      acc[3] = vdotq_lane_s32(acc[3], rhs_1, vget_high_s8(lhs_uzp.val[1]), 1);
    }

    for (int i = 0; i < 4; i++) {
      acc[i] = vshrq_n_s32(acc[i], 4);
      if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
        int32x4_t existing_acc = vld1q_s32(out_ptr);
        acc[i] = vaddq_s32(acc[i], existing_acc);
      }
      vst1q_s32(out_ptr, acc[i]);
      out_ptr += 4;
    }
  } else if (M0 == 2) {
    int32x4_t acc[2];
    for (int i = 0; i < 2; i++) {
      acc[i] = vdupq_n_s32(0);
    }

    for (int k = 0; k < params->K; ++k) {
      int8x8x2_t lhs_uzp = vld2_s8(lhs_ptr);
      lhs_ptr += 16;

      int8x16_t rhs = vld1q_s8(rhs_ptr);
      rhs_ptr += 16;

      int8x16_t rhs_0 = vshlq_n_s8(rhs, 4);
      int8x16_t rhs_1 = vandq_s8(rhs, vmask);

      acc[0] = vdotq_lane_s32(acc[0], rhs_0, lhs_uzp.val[0], 0);
      acc[0] = vdotq_lane_s32(acc[0], rhs_1, lhs_uzp.val[1], 0);
      acc[1] = vdotq_lane_s32(acc[1], rhs_0, lhs_uzp.val[0], 1);
      acc[1] = vdotq_lane_s32(acc[1], rhs_1, lhs_uzp.val[1], 1);
    }

    for (int i = 0; i < 2; i++) {
      acc[i] = vshrq_n_s32(acc[i], 4);
      if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
        int32x4_t existing_acc = vld1q_s32(out_ptr);
        acc[i] = vaddq_s32(acc[i], existing_acc);
      }
      vst1q_s32(out_ptr, acc[i]);
      out_ptr += 4;
    }
  } else if (M0 == 1) {
    int32x4_t acc = vdupq_n_s32(0);
    for (int k = 0; k < params->K; ++k) {
      int8x8_t lhs = vld1_s8(lhs_ptr);
      lhs_ptr += 8;
      int8x8x2_t lhs_uzp = vuzp_s8(lhs, vzero);

      int8x16_t rhs = vld1q_s8(rhs_ptr);
      rhs_ptr += 16;

      // We unpack int4s into individual int8s. To preserve signedness, int4s are moved to the upper 4-bits of each byte. This has the effect of
      // multiplying each int4 by 2^4 = 16. To compensate, we divide the
      // accumulator values by 16 before storing to memory.
      int8x16_t rhs_0 = vshlq_n_s8(rhs, 4);
      int8x16_t rhs_1 = vandq_s8(rhs, vmask);

      acc = vdotq_lane_s32(acc, rhs_0, lhs_uzp.val[0], 0);
      acc = vdotq_lane_s32(acc, rhs_1, lhs_uzp.val[1], 0);
    }
    acc = vshrq_n_s32(acc, 4);
    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
     int32x4_t existing_acc = vld1q_s32(out_ptr);
     acc = vaddq_s32(acc, existing_acc);
    }
    vst1q_s32(out_ptr, acc);
    out_ptr += 4;
  }
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0_1_2_4_8(
    iree_uk_mmt4d_tile_s8s4s32_1x8x8_to_8x8x8_arm_64_dotprod,
    iree_uk_mmt4d_tile_s8s4s32_1x8x8_arm_64_dotprod,
    iree_uk_mmt4d_tile_s8s4s32_2x8x8_arm_64_dotprod,
    iree_uk_mmt4d_tile_s8s4s32_4x8x8_arm_64_dotprod,
    iree_uk_mmt4d_tile_s8s4s32_8x8x8_arm_64_dotprod)
