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
  float32x4_t acc_01_01, acc_01_23, acc_01_45, acc_01_67;
  float32x4_t acc_23_01, acc_23_23, acc_23_45, acc_23_67;
  float32x4_t acc_45_01, acc_45_23, acc_45_45, acc_45_67;
  float32x4_t acc_67_01, acc_67_23, acc_67_45, acc_67_67;
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    float32x4_t acc_0_0123 = vld1q_f32(out_ptr + 8 * 0 + 0);
    float32x4_t acc_0_4567 = vld1q_f32(out_ptr + 8 * 0 + 4);
    float32x4_t acc_1_0123 = vld1q_f32(out_ptr + 8 * 1 + 0);
    float32x4_t acc_1_4567 = vld1q_f32(out_ptr + 8 * 1 + 4);
    float32x4_t acc_2_0123 = vld1q_f32(out_ptr + 8 * 2 + 0);
    float32x4_t acc_2_4567 = vld1q_f32(out_ptr + 8 * 2 + 4);
    float32x4_t acc_3_0123 = vld1q_f32(out_ptr + 8 * 3 + 0);
    float32x4_t acc_3_4567 = vld1q_f32(out_ptr + 8 * 3 + 4);
    float32x4_t acc_4_0123 = vld1q_f32(out_ptr + 8 * 4 + 0);
    float32x4_t acc_4_4567 = vld1q_f32(out_ptr + 8 * 4 + 4);
    float32x4_t acc_5_0123 = vld1q_f32(out_ptr + 8 * 5 + 0);
    float32x4_t acc_5_4567 = vld1q_f32(out_ptr + 8 * 5 + 4);
    float32x4_t acc_6_0123 = vld1q_f32(out_ptr + 8 * 6 + 0);
    float32x4_t acc_6_4567 = vld1q_f32(out_ptr + 8 * 6 + 4);
    float32x4_t acc_7_0123 = vld1q_f32(out_ptr + 8 * 7 + 0);
    float32x4_t acc_7_4567 = vld1q_f32(out_ptr + 8 * 7 + 4);
    acc_01_01 = iree_uk_neon_zip1_f32_as_s64(acc_0_0123, acc_1_0123);
    acc_01_23 = iree_uk_neon_zip2_f32_as_s64(acc_0_0123, acc_1_0123);
    acc_01_45 = iree_uk_neon_zip1_f32_as_s64(acc_0_4567, acc_1_4567);
    acc_01_67 = iree_uk_neon_zip2_f32_as_s64(acc_0_4567, acc_1_4567);
    acc_23_01 = iree_uk_neon_zip1_f32_as_s64(acc_2_0123, acc_3_0123);
    acc_23_23 = iree_uk_neon_zip2_f32_as_s64(acc_2_0123, acc_3_0123);
    acc_23_45 = iree_uk_neon_zip1_f32_as_s64(acc_2_4567, acc_3_4567);
    acc_23_67 = iree_uk_neon_zip2_f32_as_s64(acc_2_4567, acc_3_4567);
    acc_45_01 = iree_uk_neon_zip1_f32_as_s64(acc_4_0123, acc_5_0123);
    acc_45_23 = iree_uk_neon_zip2_f32_as_s64(acc_4_0123, acc_5_0123);
    acc_45_45 = iree_uk_neon_zip1_f32_as_s64(acc_4_4567, acc_5_4567);
    acc_45_67 = iree_uk_neon_zip2_f32_as_s64(acc_4_4567, acc_5_4567);
    acc_67_01 = iree_uk_neon_zip1_f32_as_s64(acc_6_0123, acc_7_0123);
    acc_67_23 = iree_uk_neon_zip2_f32_as_s64(acc_6_0123, acc_7_0123);
    acc_67_45 = iree_uk_neon_zip1_f32_as_s64(acc_6_4567, acc_7_4567);
    acc_67_67 = iree_uk_neon_zip2_f32_as_s64(acc_6_4567, acc_7_4567);
  } else {
    acc_01_01 = vdupq_n_f32(0);
    acc_01_23 = vdupq_n_f32(0);
    acc_01_45 = vdupq_n_f32(0);
    acc_01_67 = vdupq_n_f32(0);
    acc_23_01 = vdupq_n_f32(0);
    acc_23_23 = vdupq_n_f32(0);
    acc_23_45 = vdupq_n_f32(0);
    acc_23_67 = vdupq_n_f32(0);
    acc_45_01 = vdupq_n_f32(0);
    acc_45_23 = vdupq_n_f32(0);
    acc_45_45 = vdupq_n_f32(0);
    acc_45_67 = vdupq_n_f32(0);
    acc_67_01 = vdupq_n_f32(0);
    acc_67_23 = vdupq_n_f32(0);
    acc_67_45 = vdupq_n_f32(0);
    acc_67_67 = vdupq_n_f32(0);
  }

  IREE_UK_ASSUME(params->K >= 1);
  for (int k = 0; k < params->K; ++k) {
    bfloat16x8_t lhs01 = vld1q_bf16(lhs_ptr + 0);
    bfloat16x8_t lhs23 = vld1q_bf16(lhs_ptr + 8);
    bfloat16x8_t lhs45 = vld1q_bf16(lhs_ptr + 16);
    bfloat16x8_t lhs67 = vld1q_bf16(lhs_ptr + 24);
    lhs_ptr += 32;
    bfloat16x8_t rhs01 = vld1q_bf16(rhs_ptr + 0);
    bfloat16x8_t rhs23 = vld1q_bf16(rhs_ptr + 8);
    bfloat16x8_t rhs45 = vld1q_bf16(rhs_ptr + 16);
    bfloat16x8_t rhs67 = vld1q_bf16(rhs_ptr + 24);
    rhs_ptr += 32;
    acc_01_01 = vbfmmlaq_f32(acc_01_01, lhs01, rhs01);
    acc_01_23 = vbfmmlaq_f32(acc_01_23, lhs01, rhs23);
    acc_01_45 = vbfmmlaq_f32(acc_01_45, lhs01, rhs45);
    acc_01_67 = vbfmmlaq_f32(acc_01_67, lhs01, rhs67);
    acc_23_01 = vbfmmlaq_f32(acc_23_01, lhs23, rhs01);
    acc_23_23 = vbfmmlaq_f32(acc_23_23, lhs23, rhs23);
    acc_23_45 = vbfmmlaq_f32(acc_23_45, lhs23, rhs45);
    acc_23_67 = vbfmmlaq_f32(acc_23_67, lhs23, rhs67);
    acc_45_01 = vbfmmlaq_f32(acc_45_01, lhs45, rhs01);
    acc_45_23 = vbfmmlaq_f32(acc_45_23, lhs45, rhs23);
    acc_45_45 = vbfmmlaq_f32(acc_45_45, lhs45, rhs45);
    acc_45_67 = vbfmmlaq_f32(acc_45_67, lhs45, rhs67);
    acc_67_01 = vbfmmlaq_f32(acc_67_01, lhs67, rhs01);
    acc_67_23 = vbfmmlaq_f32(acc_67_23, lhs67, rhs23);
    acc_67_45 = vbfmmlaq_f32(acc_67_45, lhs67, rhs45);
    acc_67_67 = vbfmmlaq_f32(acc_67_67, lhs67, rhs67);
  }

  float32x4_t acc_0_0123 = iree_uk_neon_uzp1_f32_as_s64(acc_01_01, acc_01_23);
  float32x4_t acc_0_4567 = iree_uk_neon_uzp1_f32_as_s64(acc_01_45, acc_01_67);
  float32x4_t acc_1_0123 = iree_uk_neon_uzp2_f32_as_s64(acc_01_01, acc_01_23);
  float32x4_t acc_1_4567 = iree_uk_neon_uzp2_f32_as_s64(acc_01_45, acc_01_67);
  float32x4_t acc_2_0123 = iree_uk_neon_uzp1_f32_as_s64(acc_23_01, acc_23_23);
  float32x4_t acc_2_4567 = iree_uk_neon_uzp1_f32_as_s64(acc_23_45, acc_23_67);
  float32x4_t acc_3_0123 = iree_uk_neon_uzp2_f32_as_s64(acc_23_01, acc_23_23);
  float32x4_t acc_3_4567 = iree_uk_neon_uzp2_f32_as_s64(acc_23_45, acc_23_67);
  float32x4_t acc_4_0123 = iree_uk_neon_uzp1_f32_as_s64(acc_45_01, acc_45_23);
  float32x4_t acc_4_4567 = iree_uk_neon_uzp1_f32_as_s64(acc_45_45, acc_45_67);
  float32x4_t acc_5_0123 = iree_uk_neon_uzp2_f32_as_s64(acc_45_01, acc_45_23);
  float32x4_t acc_5_4567 = iree_uk_neon_uzp2_f32_as_s64(acc_45_45, acc_45_67);
  float32x4_t acc_6_0123 = iree_uk_neon_uzp1_f32_as_s64(acc_67_01, acc_67_23);
  float32x4_t acc_6_4567 = iree_uk_neon_uzp1_f32_as_s64(acc_67_45, acc_67_67);
  float32x4_t acc_7_0123 = iree_uk_neon_uzp2_f32_as_s64(acc_67_01, acc_67_23);
  float32x4_t acc_7_4567 = iree_uk_neon_uzp2_f32_as_s64(acc_67_45, acc_67_67);
  vst1q_f32(out_ptr + 8 * 0 + 0, acc_0_0123);
  vst1q_f32(out_ptr + 8 * 0 + 4, acc_0_4567);
  vst1q_f32(out_ptr + 8 * 1 + 0, acc_1_0123);
  vst1q_f32(out_ptr + 8 * 1 + 4, acc_1_4567);
  vst1q_f32(out_ptr + 8 * 2 + 0, acc_2_0123);
  vst1q_f32(out_ptr + 8 * 2 + 4, acc_2_4567);
  vst1q_f32(out_ptr + 8 * 3 + 0, acc_3_0123);
  vst1q_f32(out_ptr + 8 * 3 + 4, acc_3_4567);
  vst1q_f32(out_ptr + 8 * 4 + 0, acc_4_0123);
  vst1q_f32(out_ptr + 8 * 4 + 4, acc_4_4567);
  vst1q_f32(out_ptr + 8 * 5 + 0, acc_5_0123);
  vst1q_f32(out_ptr + 8 * 5 + 4, acc_5_4567);
  vst1q_f32(out_ptr + 8 * 6 + 0, acc_6_0123);
  vst1q_f32(out_ptr + 8 * 6 + 4, acc_6_4567);
  vst1q_f32(out_ptr + 8 * 7 + 0, acc_7_0123);
  vst1q_f32(out_ptr + 8 * 7 + 4, acc_7_4567);
}
