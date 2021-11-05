// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mmt4d.h"

// This file is used on any aarch64 platform; specializations for an optional
// instruction must be checked with feature test macros:
// https://developer.arm.com/documentation/101028/0010/Feature-test-macros
#if defined(__aarch64__)

#include <arm_neon.h>

#if defined(__ARM_FEATURE_DOTPROD)

MMT4D_EXPORT void mmt4d_8x4x8_i8i8i32(int k_size, const int8_t* lhs,
                                      const int8_t* rhs,
                                      int32_t* MMT4D_RESTRICT dst) {
  int32x4_t acc0 = vdupq_n_s32(0);
  int32x4_t acc1 = vdupq_n_s32(0);
  int32x4_t acc2 = vdupq_n_s32(0);
  int32x4_t acc3 = vdupq_n_s32(0);
  int32x4_t acc4 = vdupq_n_s32(0);
  int32x4_t acc5 = vdupq_n_s32(0);
  int32x4_t acc6 = vdupq_n_s32(0);
  int32x4_t acc7 = vdupq_n_s32(0);
  int32x4_t acc8 = vdupq_n_s32(0);
  int32x4_t acc9 = vdupq_n_s32(0);
  int32x4_t acc10 = vdupq_n_s32(0);
  int32x4_t acc11 = vdupq_n_s32(0);
  int32x4_t acc12 = vdupq_n_s32(0);
  int32x4_t acc13 = vdupq_n_s32(0);
  int32x4_t acc14 = vdupq_n_s32(0);
  int32x4_t acc15 = vdupq_n_s32(0);
  for (int k = 0; k < k_size; k += 4) {
    int8x16_t lhs0 = vld1q_s8(lhs + 0);
    int8x16_t lhs4 = vld1q_s8(lhs + 16);
    int8x16_t rhs0 = vld1q_s8(rhs + 0);
    int8x16_t rhs4 = vld1q_s8(rhs + 16);
    acc0 = vdotq_lane_s32(acc0, rhs0, vget_low_s8(lhs0), 0);
    acc1 = vdotq_lane_s32(acc1, rhs4, vget_low_s8(lhs0), 0);
    acc2 = vdotq_lane_s32(acc2, rhs0, vget_low_s8(lhs0), 1);
    acc3 = vdotq_lane_s32(acc3, rhs4, vget_low_s8(lhs0), 1);
    acc4 = vdotq_lane_s32(acc4, rhs0, vget_high_s8(lhs0), 0);
    acc5 = vdotq_lane_s32(acc5, rhs4, vget_high_s8(lhs0), 0);
    acc6 = vdotq_lane_s32(acc6, rhs0, vget_high_s8(lhs0), 1);
    acc7 = vdotq_lane_s32(acc7, rhs4, vget_high_s8(lhs0), 1);
    acc8 = vdotq_lane_s32(acc8, rhs0, vget_low_s8(lhs4), 0);
    acc9 = vdotq_lane_s32(acc9, rhs4, vget_low_s8(lhs4), 0);
    acc10 = vdotq_lane_s32(acc10, rhs0, vget_low_s8(lhs4), 1);
    acc11 = vdotq_lane_s32(acc11, rhs4, vget_low_s8(lhs4), 1);
    acc12 = vdotq_lane_s32(acc12, rhs0, vget_high_s8(lhs4), 0);
    acc13 = vdotq_lane_s32(acc13, rhs4, vget_high_s8(lhs4), 0);
    acc14 = vdotq_lane_s32(acc14, rhs0, vget_high_s8(lhs4), 1);
    acc15 = vdotq_lane_s32(acc15, rhs4, vget_high_s8(lhs4), 1);
    lhs += 8 * 4;
    rhs += 8 * 4;
  }
  vst1q_s32(dst + 0, acc0);
  vst1q_s32(dst + 4, acc1);
  vst1q_s32(dst + 8, acc2);
  vst1q_s32(dst + 12, acc3);
  vst1q_s32(dst + 16, acc4);
  vst1q_s32(dst + 20, acc5);
  vst1q_s32(dst + 24, acc6);
  vst1q_s32(dst + 28, acc7);
  vst1q_s32(dst + 32, acc8);
  vst1q_s32(dst + 36, acc9);
  vst1q_s32(dst + 40, acc10);
  vst1q_s32(dst + 44, acc11);
  vst1q_s32(dst + 48, acc12);
  vst1q_s32(dst + 52, acc13);
  vst1q_s32(dst + 56, acc14);
  vst1q_s32(dst + 60, acc15);
}

#else

MMT4D_GENERIC(8, 4, 8, int8_t, int8_t, int32_t);

#endif  // __ARM_FEATURE_DOTPROD

#endif  // __aarch64__
