// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/arm_64/common_arm_64.h"
#include "iree/builtins/ukernel/arch/arm_64/mmt4d_arm_64_internal.h"

static inline void iree_uk_mmt4d_tile_f32f32f32_1x8x1_to_8x8x1_arm_64(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 8 && iree_uk_is_po2_u32(M0));
  const float* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const float* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  float* IREE_UK_RESTRICT out_ptr = out_tile;
  float32x4_t acc[16];
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    for (int i = 0; i < 2 * M0; ++i) {
      acc[i] = vld1q_f32(out_ptr + 4 * i);
    }
  } else {
    for (int i = 0; i < 2 * M0; ++i) {
      acc[i] = vdupq_n_f32(0);
    }
  }
  IREE_UK_ASSUME(params->K >= 1);
  for (int k = 0; k < params->K; ++k) {
    float32x4_t rhs[2];
    for (int i = 0; i < 2; ++i) {
      rhs[i] = vld1q_f32(rhs_ptr + 4 * i);
    }
    rhs_ptr += 8;

    if (M0 == 1) {
      float lhs = *lhs_ptr++;
      acc[0] = vfmaq_n_f32(acc[0], rhs[0], lhs);
      acc[1] = vfmaq_n_f32(acc[1], rhs[1], lhs);
    } else if (M0 == 2) {
      float32x2_t lhs = vld1_f32(lhs_ptr);
      lhs_ptr += 2;
      acc[0] = vfmaq_lane_f32(acc[0], rhs[0], lhs, 0);
      acc[1] = vfmaq_lane_f32(acc[1], rhs[1], lhs, 0);
      acc[2] = vfmaq_lane_f32(acc[2], rhs[0], lhs, 1);
      acc[3] = vfmaq_lane_f32(acc[3], rhs[1], lhs, 1);
    } else {
      float32x4_t lhs[2];
      for (int i = 0; i < M0 / 4; ++i) {
        lhs[i] = vld1q_f32(lhs_ptr + 4 * i);
      }
      lhs_ptr += M0;
      acc[0] = vfmaq_lane_f32(acc[0], rhs[0], vget_low_f32(lhs[0]), 0);
      acc[1] = vfmaq_lane_f32(acc[1], rhs[1], vget_low_f32(lhs[0]), 0);
      acc[2] = vfmaq_lane_f32(acc[2], rhs[0], vget_low_f32(lhs[0]), 1);
      acc[3] = vfmaq_lane_f32(acc[3], rhs[1], vget_low_f32(lhs[0]), 1);
      acc[4] = vfmaq_lane_f32(acc[4], rhs[0], vget_high_f32(lhs[0]), 0);
      acc[5] = vfmaq_lane_f32(acc[5], rhs[1], vget_high_f32(lhs[0]), 0);
      acc[6] = vfmaq_lane_f32(acc[6], rhs[0], vget_high_f32(lhs[0]), 1);
      acc[7] = vfmaq_lane_f32(acc[7], rhs[1], vget_high_f32(lhs[0]), 1);
      if (M0 == 8) {
        acc[8] = vfmaq_lane_f32(acc[8], rhs[0], vget_low_f32(lhs[1]), 0);
        acc[9] = vfmaq_lane_f32(acc[9], rhs[1], vget_low_f32(lhs[1]), 0);
        acc[10] = vfmaq_lane_f32(acc[10], rhs[0], vget_low_f32(lhs[1]), 1);
        acc[11] = vfmaq_lane_f32(acc[11], rhs[1], vget_low_f32(lhs[1]), 1);
        acc[12] = vfmaq_lane_f32(acc[12], rhs[0], vget_high_f32(lhs[1]), 0);
        acc[13] = vfmaq_lane_f32(acc[13], rhs[1], vget_high_f32(lhs[1]), 0);
        acc[14] = vfmaq_lane_f32(acc[14], rhs[0], vget_high_f32(lhs[1]), 1);
        acc[15] = vfmaq_lane_f32(acc[15], rhs[1], vget_high_f32(lhs[1]), 1);
      }
    }
  }
  for (int i = 0; i < 2 * M0; ++i) {
    vst1q_f32(out_ptr + 4 * i, acc[i]);
  }
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0_1_2_4_8(
    iree_uk_mmt4d_tile_f32f32f32_1x8x1_to_8x8x1_arm_64,
    iree_uk_mmt4d_tile_f32f32f32_1x8x1_arm_64,
    iree_uk_mmt4d_tile_f32f32f32_2x8x1_arm_64,
    iree_uk_mmt4d_tile_f32f32f32_4x8x1_arm_64,
    iree_uk_mmt4d_tile_f32f32f32_8x8x1_arm_64)

// Shared implementation for f16f16f16 and f16f16f32.
// In the f16f16f16 case, intermediate roundings are skipped. This function
// should only be used if IREE_UK_FLAG_MMT4D_SKIP_INTERMEDIATE_ROUNDINGS is set.
static inline void iree_uk_mmt4d_tile_f16f16fXX_1x8x1_to_8x8x1_arm_64(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, iree_uk_type_t acc_type, int M0) {
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 8 && iree_uk_is_po2_u32(M0));
  const float16_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const float16_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  float32x4_t acc[16];
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    if (acc_type == IREE_UK_TYPE_FLOAT_32) {
      float* IREE_UK_RESTRICT out_ptr = out_tile;
      for (int i = 0; i < 2 * M0; ++i) {
        acc[i] = vld1q_f32(out_ptr + 4 * i);
      }
    } else {
      float16_t* IREE_UK_RESTRICT out_ptr = out_tile;
      for (int i = 0; i < 2 * M0; ++i) {
        acc[i] = vcvt_f32_f16(vld1_f16(out_ptr + 4 * i));
      }
    }
  } else {
    for (int i = 0; i < 2 * M0; ++i) {
      acc[i] = vdupq_n_f32(0);
    }
  }
  IREE_UK_ASSUME(params->K >= 1);
  for (int k = 0; k < params->K; ++k) {
    float32x4_t rhs[2];
    for (int i = 0; i < 2; ++i) {
      rhs[i] = vcvt_f32_f16(vld1_f16(rhs_ptr + 4 * i));
    }
    rhs_ptr += 8;

    if (M0 == 1) {
      float lhs = (float)*lhs_ptr++;
      acc[0] = vfmaq_n_f32(acc[0], rhs[0], lhs);
      acc[1] = vfmaq_n_f32(acc[1], rhs[1], lhs);
    } else if (M0 == 2) {
      float16x4_t lhs_f16 = vld1_dup_f16(lhs_ptr);
      lhs_f16 = vld1_lane_f16(lhs_ptr + 1, lhs_f16, 1);
      lhs_ptr += 2;
      float32x2_t lhs = vget_low_f32(vcvt_f32_f16(lhs_f16));
      acc[0] = vfmaq_lane_f32(acc[0], rhs[0], lhs, 0);
      acc[1] = vfmaq_lane_f32(acc[1], rhs[1], lhs, 0);
      acc[2] = vfmaq_lane_f32(acc[2], rhs[0], lhs, 1);
      acc[3] = vfmaq_lane_f32(acc[3], rhs[1], lhs, 1);
    } else {
      float32x4_t lhs[2];
      for (int i = 0; i < M0 / 4; ++i) {
        lhs[i] = vcvt_f32_f16(vld1_f16(lhs_ptr + 4 * i));
      }
      lhs_ptr += M0;
      acc[0] = vfmaq_lane_f32(acc[0], rhs[0], vget_low_f32(lhs[0]), 0);
      acc[1] = vfmaq_lane_f32(acc[1], rhs[1], vget_low_f32(lhs[0]), 0);
      acc[2] = vfmaq_lane_f32(acc[2], rhs[0], vget_low_f32(lhs[0]), 1);
      acc[3] = vfmaq_lane_f32(acc[3], rhs[1], vget_low_f32(lhs[0]), 1);
      acc[4] = vfmaq_lane_f32(acc[4], rhs[0], vget_high_f32(lhs[0]), 0);
      acc[5] = vfmaq_lane_f32(acc[5], rhs[1], vget_high_f32(lhs[0]), 0);
      acc[6] = vfmaq_lane_f32(acc[6], rhs[0], vget_high_f32(lhs[0]), 1);
      acc[7] = vfmaq_lane_f32(acc[7], rhs[1], vget_high_f32(lhs[0]), 1);
      if (M0 == 8) {
        acc[8] = vfmaq_lane_f32(acc[8], rhs[0], vget_low_f32(lhs[1]), 0);
        acc[9] = vfmaq_lane_f32(acc[9], rhs[1], vget_low_f32(lhs[1]), 0);
        acc[10] = vfmaq_lane_f32(acc[10], rhs[0], vget_low_f32(lhs[1]), 1);
        acc[11] = vfmaq_lane_f32(acc[11], rhs[1], vget_low_f32(lhs[1]), 1);
        acc[12] = vfmaq_lane_f32(acc[12], rhs[0], vget_high_f32(lhs[1]), 0);
        acc[13] = vfmaq_lane_f32(acc[13], rhs[1], vget_high_f32(lhs[1]), 0);
        acc[14] = vfmaq_lane_f32(acc[14], rhs[0], vget_high_f32(lhs[1]), 1);
        acc[15] = vfmaq_lane_f32(acc[15], rhs[1], vget_high_f32(lhs[1]), 1);
      }
    }
  }
  if (acc_type == IREE_UK_TYPE_FLOAT_32) {
    float* IREE_UK_RESTRICT out_ptr = out_tile;
    for (int i = 0; i < 2 * M0; ++i) {
      vst1q_f32(out_ptr + 4 * i, acc[i]);
    }
  } else {
    float16_t* IREE_UK_RESTRICT out_ptr = out_tile;
    for (int i = 0; i < 2 * M0; ++i) {
      vst1_f16(out_ptr + 4 * i, vcvt_f16_f32(acc[i]));
    }
  }
}

static inline void iree_uk_mmt4d_tile_f16f16f16_1x8x1_to_8x8x1_arm_64(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  iree_uk_mmt4d_tile_f16f16fXX_1x8x1_to_8x8x1_arm_64(
      out_tile, lhs_panel, rhs_panel, params, IREE_UK_TYPE_FLOAT_16, M0);
}

static inline void iree_uk_mmt4d_tile_f16f16f32_1x8x1_to_8x8x1_arm_64(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  iree_uk_mmt4d_tile_f16f16fXX_1x8x1_to_8x8x1_arm_64(
      out_tile, lhs_panel, rhs_panel, params, IREE_UK_TYPE_FLOAT_32, M0);
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0_1_2_4_8(
    iree_uk_mmt4d_tile_f16f16f32_1x8x1_to_8x8x1_arm_64,
    iree_uk_mmt4d_tile_f16f16f32_1x8x1_arm_64,
    iree_uk_mmt4d_tile_f16f16f32_2x8x1_arm_64,
    iree_uk_mmt4d_tile_f16f16f32_4x8x1_arm_64,
    iree_uk_mmt4d_tile_f16f16f32_8x8x1_arm_64)

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0_1_2_4_8(
    iree_uk_mmt4d_tile_f16f16f16_1x8x1_to_8x8x1_arm_64,
    iree_uk_mmt4d_tile_f16f16f16_1x8x1_arm_64,
    iree_uk_mmt4d_tile_f16f16f16_2x8x1_arm_64,
    iree_uk_mmt4d_tile_f16f16f16_4x8x1_arm_64,
    iree_uk_mmt4d_tile_f16f16f16_8x8x1_arm_64)

static inline void iree_uk_mmt4d_tile_i8i8i32_1x8x1_to_8x8x1_arm_64(
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
  IREE_UK_ASSUME(params->K >= 1);
  for (int k = 0; k < params->K; ++k) {
    int16x8_t rhs = vmovl_s8(vld1_s8(rhs_ptr));
    rhs_ptr += 8;
    if (M0 <= 4) {
      for (int i = 0; i < M0; ++i) {
        int16_t lhs = *lhs_ptr++;
        acc[2 * i + 0] = vmlal_n_s16(acc[2 * i + 0], vget_low_s16(rhs), lhs);
        acc[2 * i + 1] = vmlal_n_s16(acc[2 * i + 1], vget_high_s16(rhs), lhs);
      }
    } else {
      int16x8_t lhs = vmovl_s8(vld1_s8(lhs_ptr));
      lhs_ptr += 8;
      acc[0] = vmlal_lane_s16(acc[0], vget_low_s16(rhs), vget_low_s16(lhs), 0);
      acc[1] = vmlal_lane_s16(acc[1], vget_high_s16(rhs), vget_low_s16(lhs), 0);
      acc[2] = vmlal_lane_s16(acc[2], vget_low_s16(rhs), vget_low_s16(lhs), 1);
      acc[3] = vmlal_lane_s16(acc[3], vget_high_s16(rhs), vget_low_s16(lhs), 1);
      acc[4] = vmlal_lane_s16(acc[4], vget_low_s16(rhs), vget_low_s16(lhs), 2);
      acc[5] = vmlal_lane_s16(acc[5], vget_high_s16(rhs), vget_low_s16(lhs), 2);
      acc[6] = vmlal_lane_s16(acc[6], vget_low_s16(rhs), vget_low_s16(lhs), 3);
      acc[7] = vmlal_lane_s16(acc[7], vget_high_s16(rhs), vget_low_s16(lhs), 3);
      acc[8] = vmlal_lane_s16(acc[8], vget_low_s16(rhs), vget_high_s16(lhs), 0);
      acc[9] =
          vmlal_lane_s16(acc[9], vget_high_s16(rhs), vget_high_s16(lhs), 0);
      acc[10] =
          vmlal_lane_s16(acc[10], vget_low_s16(rhs), vget_high_s16(lhs), 1);
      acc[11] =
          vmlal_lane_s16(acc[11], vget_high_s16(rhs), vget_high_s16(lhs), 1);
      acc[12] =
          vmlal_lane_s16(acc[12], vget_low_s16(rhs), vget_high_s16(lhs), 2);
      acc[13] =
          vmlal_lane_s16(acc[13], vget_high_s16(rhs), vget_high_s16(lhs), 2);
      acc[14] =
          vmlal_lane_s16(acc[14], vget_low_s16(rhs), vget_high_s16(lhs), 3);
      acc[15] =
          vmlal_lane_s16(acc[15], vget_high_s16(rhs), vget_high_s16(lhs), 3);
    }
  }
  for (int i = 0; i < 2 * M0; ++i) {
    vst1q_s32(out_ptr + 4 * i, acc[i]);
  }
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0_1_2_4_8(
    iree_uk_mmt4d_tile_i8i8i32_1x8x1_to_8x8x1_arm_64,
    iree_uk_mmt4d_tile_i8i8i32_1x8x1_arm_64,
    iree_uk_mmt4d_tile_i8i8i32_2x8x1_arm_64,
    iree_uk_mmt4d_tile_i8i8i32_4x8x1_arm_64,
    iree_uk_mmt4d_tile_i8i8i32_8x8x1_arm_64)
