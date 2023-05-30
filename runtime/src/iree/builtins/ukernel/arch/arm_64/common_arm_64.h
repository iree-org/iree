// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_ARCH_ARM_64_COMMON_ARM_64_H_
#define IREE_BUILTINS_UKERNEL_ARCH_ARM_64_COMMON_ARM_64_H_

#include <arm_neon.h>

#include "iree/builtins/ukernel/common.h"
#include "iree/schemas/cpu_data.h"

#if IREE_UK_COMPILER_CLANG_VERSION_AT_LEAST(7, 0) || \
    IREE_UK_COMPILER_GCC_VERSION_AT_LEAST(8, 0)
#define IREE_UK_BUILD_ARM_64_DOTPROD
static inline bool iree_uk_cpu_supports_dotprod(
    const iree_uk_uint64_t* cpu_data) {
  return iree_uk_all_bits_set(cpu_data[0], IREE_CPU_DATA0_ARM_64_DOTPROD);
}
#endif

#if IREE_UK_COMPILER_CLANG_VERSION_AT_LEAST(10, 0) || \
    IREE_UK_COMPILER_GCC_VERSION_AT_LEAST(10, 0)
#define IREE_UK_BUILD_ARM_64_I8MM
static inline bool iree_uk_cpu_supports_i8mm(const iree_uk_uint64_t* cpu_data) {
  return iree_uk_all_bits_set(cpu_data[0], IREE_CPU_DATA0_ARM_64_I8MM);
}
#endif

static inline int8x16x2_t iree_uk_neon_load_8x4xi8_strided(
    const iree_uk_int8_t* src, iree_uk_ssize_t stride) {
  int32x4_t v0_i32 = vdupq_n_s32(0);
  int32x4_t v1_i32 = vdupq_n_s32(0);
  v0_i32 =
      vld1q_lane_s32((const iree_uk_int32_t*)(src + 0 * stride), v0_i32, 0);
  v0_i32 =
      vld1q_lane_s32((const iree_uk_int32_t*)(src + 1 * stride), v0_i32, 1);
  v0_i32 =
      vld1q_lane_s32((const iree_uk_int32_t*)(src + 2 * stride), v0_i32, 2);
  v0_i32 =
      vld1q_lane_s32((const iree_uk_int32_t*)(src + 3 * stride), v0_i32, 3);
  v1_i32 =
      vld1q_lane_s32((const iree_uk_int32_t*)(src + 4 * stride), v1_i32, 0);
  v1_i32 =
      vld1q_lane_s32((const iree_uk_int32_t*)(src + 5 * stride), v1_i32, 1);
  v1_i32 =
      vld1q_lane_s32((const iree_uk_int32_t*)(src + 6 * stride), v1_i32, 2);
  v1_i32 =
      vld1q_lane_s32((const iree_uk_int32_t*)(src + 7 * stride), v1_i32, 3);
  int8x16x2_t v;
  v.val[0] = vreinterpretq_s8_s32(v0_i32);
  v.val[1] = vreinterpretq_s8_s32(v1_i32);
  return v;
}

static inline int8x16x4_t iree_uk_neon_load_8x8xi8_strided_permute(
    const iree_uk_int8_t* src, iree_uk_ssize_t stride, int p0, int p1, int p2,
    int p3, int p4, int p5, int p6, int p7) {
  int8x8_t row0 = vld1_s8(src + p0 * stride);
  int8x8_t row1 = vld1_s8(src + p1 * stride);
  int8x8_t row2 = vld1_s8(src + p2 * stride);
  int8x8_t row3 = vld1_s8(src + p3 * stride);
  int8x8_t row4 = vld1_s8(src + p4 * stride);
  int8x8_t row5 = vld1_s8(src + p5 * stride);
  int8x8_t row6 = vld1_s8(src + p6 * stride);
  int8x8_t row7 = vld1_s8(src + p7 * stride);
  int8x16x4_t v;
  v.val[0] = vcombine_s8(row0, row1);
  v.val[1] = vcombine_s8(row2, row3);
  v.val[2] = vcombine_s8(row4, row5);
  v.val[3] = vcombine_s8(row6, row7);
  return v;
}

static inline int8x16x4_t iree_uk_neon_load_8x8xi8_strided(
    const iree_uk_int8_t* src, iree_uk_ssize_t stride) {
  return iree_uk_neon_load_8x8xi8_strided_permute(src, stride, 0, 1, 2, 3, 4, 5,
                                                  6, 7);
}

static inline int16x8x2_t iree_uk_neon_zip_16xi8_as_8xi16(int8x16_t a,
                                                          int8x16_t b) {
  int8x16x2_t z = vzipq_s8(a, b);
  int16x8x2_t r;
  r.val[0] = vreinterpretq_s16_s8(z.val[0]);
  r.val[1] = vreinterpretq_s16_s8(z.val[1]);
  return r;
}

static inline int32x4x2_t iree_uk_neon_zip_8xi16_as_4xi32(int16x8_t a,
                                                          int16x8_t b) {
  int16x8x2_t z = vzipq_s16(a, b);
  int32x4x2_t r;
  r.val[0] = vreinterpretq_s32_s16(z.val[0]);
  r.val[1] = vreinterpretq_s32_s16(z.val[1]);
  return r;
}

static inline int64x2x2_t iree_uk_neon_zip_4xi32_as_2xi64(int32x4_t a,
                                                          int32x4_t b) {
  int32x4x2_t z = vzipq_s32(a, b);
  int64x2x2_t r;
  r.val[0] = vreinterpretq_s64_s32(z.val[0]);
  r.val[1] = vreinterpretq_s64_s32(z.val[1]);
  return r;
}

static inline void iree_uk_neon_copy_8x1xi8_strided_to_unstrided(
    iree_uk_int8_t* IREE_UK_RESTRICT out_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr, iree_uk_ssize_t in_stride) {
  int8x8_t v = vdup_n_s8(0);
  v = vld1_lane_s8(in_ptr + 0 * in_stride, v, 0);
  v = vld1_lane_s8(in_ptr + 1 * in_stride, v, 1);
  v = vld1_lane_s8(in_ptr + 2 * in_stride, v, 2);
  v = vld1_lane_s8(in_ptr + 3 * in_stride, v, 3);
  v = vld1_lane_s8(in_ptr + 4 * in_stride, v, 4);
  v = vld1_lane_s8(in_ptr + 5 * in_stride, v, 5);
  v = vld1_lane_s8(in_ptr + 6 * in_stride, v, 6);
  v = vld1_lane_s8(in_ptr + 7 * in_stride, v, 7);
  vst1_s8(out_ptr, v);
}

static inline void iree_uk_neon_copy_8x4xi8_strided_to_unstrided(
    iree_uk_int8_t* IREE_UK_RESTRICT out_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr, iree_uk_ssize_t in_stride) {
  int8x16x2_t in = iree_uk_neon_load_8x4xi8_strided(in_ptr, in_stride);
  vst1q_s8(out_ptr + 0, in.val[0]);
  vst1q_s8(out_ptr + 16, in.val[1]);
}

static inline void iree_uk_neon_copy_8x8xi8_strided_to_unstrided(
    iree_uk_int8_t* IREE_UK_RESTRICT out_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr, iree_uk_ssize_t in_stride) {
  int8x16x4_t in = iree_uk_neon_load_8x8xi8_strided(in_ptr, in_stride);
  vst1q_s8(out_ptr + 0, in.val[0]);
  vst1q_s8(out_ptr + 16, in.val[1]);
  vst1q_s8(out_ptr + 32, in.val[2]);
  vst1q_s8(out_ptr + 48, in.val[3]);
}

static inline void
iree_uk_neon_copy_8x8xi8_tiled_1x4_transpose_strided_to_strided(
    iree_uk_int8_t* IREE_UK_RESTRICT out_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr, iree_uk_ssize_t out_stride,
    iree_uk_ssize_t in_stride) {
  int8x16x4_t in = iree_uk_neon_load_8x8xi8_strided_permute(
      in_ptr, in_stride, 0, 2, 1, 3, 4, 6, 5, 7);
  int32x4x2_t c0 = vtrnq_s32(vreinterpretq_s32_s8(in.val[0]),
                             vreinterpretq_s32_s8(in.val[1]));
  int32x4x2_t c1 = vtrnq_s32(vreinterpretq_s32_s8(in.val[2]),
                             vreinterpretq_s32_s8(in.val[3]));
  vst1q_s8(out_ptr + 0 + 0 * out_stride, vreinterpretq_s8_s32(c0.val[0]));
  vst1q_s8(out_ptr + 16 + 0 * out_stride, vreinterpretq_s8_s32(c1.val[0]));
  vst1q_s8(out_ptr + 0 + 1 * out_stride, vreinterpretq_s8_s32(c0.val[1]));
  vst1q_s8(out_ptr + 16 + 1 * out_stride, vreinterpretq_s8_s32(c1.val[1]));
}

static inline void iree_uk_neon_copy_8x32xi8_strided_to_strided(
    iree_uk_int8_t* IREE_UK_RESTRICT out_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr, iree_uk_ssize_t out_stride,
    iree_uk_ssize_t in_stride) {
  for (int i = 0; i < 8; ++i) {
    iree_uk_memcpy(out_ptr + i * out_stride, in_ptr + i * in_stride, 32);
  }
}

static inline void iree_uk_neon_copy_8x8xi8_transpose_strided_to_strided(
    iree_uk_int8_t* IREE_UK_RESTRICT out_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr, iree_uk_ssize_t out_stride,
    iree_uk_ssize_t in_stride) {
  int8x16x4_t in = iree_uk_neon_load_8x8xi8_strided_permute(
      in_ptr, in_stride, 0, 4, 1, 5, 2, 6, 3, 7);
  int16x8x2_t zip_i16_0 = iree_uk_neon_zip_16xi8_as_8xi16(in.val[0], in.val[1]);
  int16x8x2_t zip_i16_1 = iree_uk_neon_zip_16xi8_as_8xi16(in.val[2], in.val[3]);
  int32x4x2_t zip_i32_0 =
      iree_uk_neon_zip_8xi16_as_4xi32(zip_i16_0.val[0], zip_i16_1.val[0]);
  int32x4x2_t zip_i32_1 =
      iree_uk_neon_zip_8xi16_as_4xi32(zip_i16_0.val[1], zip_i16_1.val[1]);
  int64x2x2_t zip_i64_0 =
      iree_uk_neon_zip_4xi32_as_2xi64(zip_i32_0.val[0], zip_i32_1.val[0]);
  int64x2x2_t zip_i64_1 =
      iree_uk_neon_zip_4xi32_as_2xi64(zip_i32_0.val[1], zip_i32_1.val[1]);
  int8x16x4_t out;
  out.val[0] = vreinterpretq_s8_s64(zip_i64_0.val[0]);
  out.val[1] = vreinterpretq_s8_s64(zip_i64_0.val[1]);
  out.val[2] = vreinterpretq_s8_s64(zip_i64_1.val[0]);
  out.val[3] = vreinterpretq_s8_s64(zip_i64_1.val[1]);
  vst1_s8(out_ptr + 0 * out_stride, vget_low_s8(out.val[0]));
  vst1_s8(out_ptr + 1 * out_stride, vget_high_s8(out.val[0]));
  vst1_s8(out_ptr + 2 * out_stride, vget_low_s8(out.val[1]));
  vst1_s8(out_ptr + 3 * out_stride, vget_high_s8(out.val[1]));
  vst1_s8(out_ptr + 4 * out_stride, vget_low_s8(out.val[2]));
  vst1_s8(out_ptr + 5 * out_stride, vget_high_s8(out.val[2]));
  vst1_s8(out_ptr + 6 * out_stride, vget_low_s8(out.val[3]));
  vst1_s8(out_ptr + 7 * out_stride, vget_high_s8(out.val[3]));
}

static inline void iree_uk_neon_copy_8x8xi8_transpose_strided_to_unstrided(
    iree_uk_int8_t* IREE_UK_RESTRICT out_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr, iree_uk_ssize_t in_stride) {
  // Clang (Android NDK r25) actually produces worse code when this code is
  // specialized for out_stride==8 using longer contiguous stores!
  iree_uk_neon_copy_8x8xi8_transpose_strided_to_strided(out_ptr, in_ptr, 8,
                                                        in_stride);
}

#endif  // IREE_BUILTINS_UKERNEL_ARCH_ARM_64_COMMON_ARM_64_H_
