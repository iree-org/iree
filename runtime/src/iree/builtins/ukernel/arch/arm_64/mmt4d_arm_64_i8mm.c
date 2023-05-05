// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <arm_neon.h>

#include "iree/builtins/ukernel/arch/arm_64/mmt4d_arm_64.h"

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

void iree_uk_mmt4d_tile_i8i8i32_8x8x8_arm_64_i8mm_intrinsics(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel, iree_uk_int32_t K,
    iree_uk_uint32_t flags, const iree_uk_mmt4d_params_t* params) {
  (void)params;
  const iree_uk_int8_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;
  int32x4_t acc_01_01, acc_01_23, acc_01_45, acc_01_67;
  int32x4_t acc_23_01, acc_23_23, acc_23_45, acc_23_67;
  int32x4_t acc_45_01, acc_45_23, acc_45_45, acc_45_67;
  int32x4_t acc_67_01, acc_67_23, acc_67_45, acc_67_67;
  if (flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    int32x4_t acc_0_0123 = vld1q_s32(out_ptr + 8 * 0 + 0);
    int32x4_t acc_0_4567 = vld1q_s32(out_ptr + 8 * 0 + 4);
    int32x4_t acc_1_0123 = vld1q_s32(out_ptr + 8 * 1 + 0);
    int32x4_t acc_1_4567 = vld1q_s32(out_ptr + 8 * 1 + 4);
    int32x4_t acc_2_0123 = vld1q_s32(out_ptr + 8 * 2 + 0);
    int32x4_t acc_2_4567 = vld1q_s32(out_ptr + 8 * 2 + 4);
    int32x4_t acc_3_0123 = vld1q_s32(out_ptr + 8 * 3 + 0);
    int32x4_t acc_3_4567 = vld1q_s32(out_ptr + 8 * 3 + 4);
    int32x4_t acc_4_0123 = vld1q_s32(out_ptr + 8 * 4 + 0);
    int32x4_t acc_4_4567 = vld1q_s32(out_ptr + 8 * 4 + 4);
    int32x4_t acc_5_0123 = vld1q_s32(out_ptr + 8 * 5 + 0);
    int32x4_t acc_5_4567 = vld1q_s32(out_ptr + 8 * 5 + 4);
    int32x4_t acc_6_0123 = vld1q_s32(out_ptr + 8 * 6 + 0);
    int32x4_t acc_6_4567 = vld1q_s32(out_ptr + 8 * 6 + 4);
    int32x4_t acc_7_0123 = vld1q_s32(out_ptr + 8 * 7 + 0);
    int32x4_t acc_7_4567 = vld1q_s32(out_ptr + 8 * 7 + 4);
    acc_01_01 = iree_uk_neon_zip1_s32_as_s64(acc_0_0123, acc_1_0123);
    acc_01_23 = iree_uk_neon_zip2_s32_as_s64(acc_0_0123, acc_1_0123);
    acc_01_45 = iree_uk_neon_zip1_s32_as_s64(acc_0_4567, acc_1_4567);
    acc_01_67 = iree_uk_neon_zip2_s32_as_s64(acc_0_4567, acc_1_4567);
    acc_23_01 = iree_uk_neon_zip1_s32_as_s64(acc_2_0123, acc_3_0123);
    acc_23_23 = iree_uk_neon_zip2_s32_as_s64(acc_2_0123, acc_3_0123);
    acc_23_45 = iree_uk_neon_zip1_s32_as_s64(acc_2_4567, acc_3_4567);
    acc_23_67 = iree_uk_neon_zip2_s32_as_s64(acc_2_4567, acc_3_4567);
    acc_45_01 = iree_uk_neon_zip1_s32_as_s64(acc_4_0123, acc_5_0123);
    acc_45_23 = iree_uk_neon_zip2_s32_as_s64(acc_4_0123, acc_5_0123);
    acc_45_45 = iree_uk_neon_zip1_s32_as_s64(acc_4_4567, acc_5_4567);
    acc_45_67 = iree_uk_neon_zip2_s32_as_s64(acc_4_4567, acc_5_4567);
    acc_67_01 = iree_uk_neon_zip1_s32_as_s64(acc_6_0123, acc_7_0123);
    acc_67_23 = iree_uk_neon_zip2_s32_as_s64(acc_6_0123, acc_7_0123);
    acc_67_45 = iree_uk_neon_zip1_s32_as_s64(acc_6_4567, acc_7_4567);
    acc_67_67 = iree_uk_neon_zip2_s32_as_s64(acc_6_4567, acc_7_4567);
  } else {
    acc_01_01 = vdupq_n_s32(0);
    acc_01_23 = vdupq_n_s32(0);
    acc_01_45 = vdupq_n_s32(0);
    acc_01_67 = vdupq_n_s32(0);
    acc_23_01 = vdupq_n_s32(0);
    acc_23_23 = vdupq_n_s32(0);
    acc_23_45 = vdupq_n_s32(0);
    acc_23_67 = vdupq_n_s32(0);
    acc_45_01 = vdupq_n_s32(0);
    acc_45_23 = vdupq_n_s32(0);
    acc_45_45 = vdupq_n_s32(0);
    acc_45_67 = vdupq_n_s32(0);
    acc_67_01 = vdupq_n_s32(0);
    acc_67_23 = vdupq_n_s32(0);
    acc_67_45 = vdupq_n_s32(0);
    acc_67_67 = vdupq_n_s32(0);
  }
  IREE_UK_ASSUME(K >= 1);
  for (int k = 0; k < K; ++k) {
    int8x16_t lhs01 = vld1q_s8(lhs_ptr + 0);
    int8x16_t lhs23 = vld1q_s8(lhs_ptr + 16);
    int8x16_t lhs45 = vld1q_s8(lhs_ptr + 32);
    int8x16_t lhs67 = vld1q_s8(lhs_ptr + 48);
    lhs_ptr += 64;
    int8x16_t rhs01 = vld1q_s8(rhs_ptr + 0);
    int8x16_t rhs23 = vld1q_s8(rhs_ptr + 16);
    int8x16_t rhs45 = vld1q_s8(rhs_ptr + 32);
    int8x16_t rhs67 = vld1q_s8(rhs_ptr + 48);
    rhs_ptr += 64;
    acc_01_01 = vmmlaq_s32(acc_01_01, lhs01, rhs01);
    acc_01_23 = vmmlaq_s32(acc_01_23, lhs01, rhs23);
    acc_01_45 = vmmlaq_s32(acc_01_45, lhs01, rhs45);
    acc_01_67 = vmmlaq_s32(acc_01_67, lhs01, rhs67);
    acc_23_01 = vmmlaq_s32(acc_23_01, lhs23, rhs01);
    acc_23_23 = vmmlaq_s32(acc_23_23, lhs23, rhs23);
    acc_23_45 = vmmlaq_s32(acc_23_45, lhs23, rhs45);
    acc_23_67 = vmmlaq_s32(acc_23_67, lhs23, rhs67);
    acc_45_01 = vmmlaq_s32(acc_45_01, lhs45, rhs01);
    acc_45_23 = vmmlaq_s32(acc_45_23, lhs45, rhs23);
    acc_45_45 = vmmlaq_s32(acc_45_45, lhs45, rhs45);
    acc_45_67 = vmmlaq_s32(acc_45_67, lhs45, rhs67);
    acc_67_01 = vmmlaq_s32(acc_67_01, lhs67, rhs01);
    acc_67_23 = vmmlaq_s32(acc_67_23, lhs67, rhs23);
    acc_67_45 = vmmlaq_s32(acc_67_45, lhs67, rhs45);
    acc_67_67 = vmmlaq_s32(acc_67_67, lhs67, rhs67);
  }

  int32x4_t acc_0_0123 = iree_uk_neon_uzp1_s32_as_s64(acc_01_01, acc_01_23);
  int32x4_t acc_0_4567 = iree_uk_neon_uzp1_s32_as_s64(acc_01_45, acc_01_67);
  int32x4_t acc_1_0123 = iree_uk_neon_uzp2_s32_as_s64(acc_01_01, acc_01_23);
  int32x4_t acc_1_4567 = iree_uk_neon_uzp2_s32_as_s64(acc_01_45, acc_01_67);
  int32x4_t acc_2_0123 = iree_uk_neon_uzp1_s32_as_s64(acc_23_01, acc_23_23);
  int32x4_t acc_2_4567 = iree_uk_neon_uzp1_s32_as_s64(acc_23_45, acc_23_67);
  int32x4_t acc_3_0123 = iree_uk_neon_uzp2_s32_as_s64(acc_23_01, acc_23_23);
  int32x4_t acc_3_4567 = iree_uk_neon_uzp2_s32_as_s64(acc_23_45, acc_23_67);
  int32x4_t acc_4_0123 = iree_uk_neon_uzp1_s32_as_s64(acc_45_01, acc_45_23);
  int32x4_t acc_4_4567 = iree_uk_neon_uzp1_s32_as_s64(acc_45_45, acc_45_67);
  int32x4_t acc_5_0123 = iree_uk_neon_uzp2_s32_as_s64(acc_45_01, acc_45_23);
  int32x4_t acc_5_4567 = iree_uk_neon_uzp2_s32_as_s64(acc_45_45, acc_45_67);
  int32x4_t acc_6_0123 = iree_uk_neon_uzp1_s32_as_s64(acc_67_01, acc_67_23);
  int32x4_t acc_6_4567 = iree_uk_neon_uzp1_s32_as_s64(acc_67_45, acc_67_67);
  int32x4_t acc_7_0123 = iree_uk_neon_uzp2_s32_as_s64(acc_67_01, acc_67_23);
  int32x4_t acc_7_4567 = iree_uk_neon_uzp2_s32_as_s64(acc_67_45, acc_67_67);
  vst1q_s32(out_ptr + 8 * 0 + 0, acc_0_0123);
  vst1q_s32(out_ptr + 8 * 0 + 4, acc_0_4567);
  vst1q_s32(out_ptr + 8 * 1 + 0, acc_1_0123);
  vst1q_s32(out_ptr + 8 * 1 + 4, acc_1_4567);
  vst1q_s32(out_ptr + 8 * 2 + 0, acc_2_0123);
  vst1q_s32(out_ptr + 8 * 2 + 4, acc_2_4567);
  vst1q_s32(out_ptr + 8 * 3 + 0, acc_3_0123);
  vst1q_s32(out_ptr + 8 * 3 + 4, acc_3_4567);
  vst1q_s32(out_ptr + 8 * 4 + 0, acc_4_0123);
  vst1q_s32(out_ptr + 8 * 4 + 4, acc_4_4567);
  vst1q_s32(out_ptr + 8 * 5 + 0, acc_5_0123);
  vst1q_s32(out_ptr + 8 * 5 + 4, acc_5_4567);
  vst1q_s32(out_ptr + 8 * 6 + 0, acc_6_0123);
  vst1q_s32(out_ptr + 8 * 6 + 4, acc_6_4567);
  vst1q_s32(out_ptr + 8 * 7 + 0, acc_7_0123);
  vst1q_s32(out_ptr + 8 * 7 + 4, acc_7_4567);
}

#if defined(IREE_UK_ENABLE_INLINE_ASM)
// Compared to the intrinsics code path, this asm code has optimizations (loop
// pipelining, 2x partial unrolling) that were introduced in #10552. An attempt
// to bring these optimizations to the intrinsics code path ran into LLVM ARM
// backend issues: https://github.com/llvm/llvm-project/issues/62527, so for now
// we retain this asm code path.
void iree_uk_mmt4d_tile_i8i8i32_8x8x8_arm_64_i8mm_inline_asm(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel, iree_uk_int32_t K,
    iree_uk_uint32_t flags, const iree_uk_mmt4d_params_t* params) {
  (void)params;
  asm("    // Do we accumulate into or clear the accumulator tile?         \n"
      "    tbnz w4, %[accumulate_bit_pos], 1f                              \n"
      "                                                                    \n"
      "0:                                                                  \n"
      "    // No-accumulate case. Clear the 8x8 accumulator tile.          \n"
      "    movi v16.16b, 0                                                 \n"
      "    movi v17.16b, 0                                                 \n"
      "    movi v18.16b, 0                                                 \n"
      "    movi v19.16b, 0                                                 \n"
      "    movi v20.16b, 0                                                 \n"
      "    movi v21.16b, 0                                                 \n"
      "    movi v22.16b, 0                                                 \n"
      "    movi v23.16b, 0                                                 \n"
      "    movi v24.16b, 0                                                 \n"
      "    movi v25.16b, 0                                                 \n"
      "    movi v26.16b, 0                                                 \n"
      "    movi v27.16b, 0                                                 \n"
      "    movi v28.16b, 0                                                 \n"
      "    movi v29.16b, 0                                                 \n"
      "    movi v30.16b, 0                                                 \n"
      "    movi v31.16b, 0                                                 \n"
      "    b 2f                                                            \n"
      "                                                                    \n"
      "1:                                                                  \n"
      "    // Accumulate case. Load the 8x8 accumulator tile from          \n"
      "    // row-major out_tile and swizzle it into 2x2 tiled layout.     \n"
      "    //                                                              \n"
      "    // Load rows 0--3.                                              \n"
      "    ldp q0, q1, [x0, 0]                                             \n"
      "    ldp q2, q3, [x0, 32]                                            \n"
      "    ldp q4, q5, [x0, 64]                                            \n"
      "    ldp q6, q7, [x0, 96]                                            \n"
      "    // Load rows 4--7.                                              \n"
      "    ldp q8, q9, [x0, 128]                                           \n"
      "    ldp q10, q11, [x0, 160]                                         \n"
      "    ldp q12, q13, [x0, 192]                                         \n"
      "    ldp q14, q15, [x0, 224]                                         \n"
      "    // Swizzle in 2x2 tiles for smmla, rows 0--1.                   \n"
      "    zip1 v16.2d, v0.2d, v2.2d                                       \n"
      "    zip2 v17.2d, v0.2d, v2.2d                                       \n"
      "    zip1 v18.2d, v1.2d, v3.2d                                       \n"
      "    zip2 v19.2d, v1.2d, v3.2d                                       \n"
      "    // Swizzle in 2x2 tiles for smmla, rows 2--3.                   \n"
      "    zip1 v20.2d, v4.2d, v6.2d                                       \n"
      "    zip2 v21.2d, v4.2d, v6.2d                                       \n"
      "    zip1 v22.2d, v5.2d, v7.2d                                       \n"
      "    zip2 v23.2d, v5.2d, v7.2d                                       \n"
      "    // Swizzle in 2x2 tiles for smmla, rows 4--5.                   \n"
      "    zip1 v24.2d, v8.2d, v10.2d                                      \n"
      "    zip2 v25.2d, v8.2d, v10.2d                                      \n"
      "    zip1 v26.2d, v9.2d, v11.2d                                      \n"
      "    zip2 v27.2d, v9.2d, v11.2d                                      \n"
      "    // Swizzle in 2x2 tiles for smmla, rows 6--7.                   \n"
      "    zip1 v28.2d, v12.2d, v14.2d                                     \n"
      "    zip2 v29.2d, v12.2d, v14.2d                                     \n"
      "    zip1 v30.2d, v13.2d, v15.2d                                     \n"
      "    zip2 v31.2d, v13.2d, v15.2d                                     \n"
      "                                                                    \n"
      "  2:                                                                \n"
      "                                                                    \n"
      "    // Start of math work. If K==1, jump over the whole main loop.  \n"
      "    subs w3, w3, 1                                                  \n"
      "    b.eq 6f                                                         \n"
      "                                                                    \n"
      "  3:                                                                \n"
      "    // Prologue of main loop, 2x partially unrolled, for when K>=2. \n"
      "    //                                                              \n"
      "    // Decrement the loop counter K.                                \n"
      "    subs w3, w3, 2                                                  \n"
      "    // Pre-load data for first loop iteration                       \n"
      "    //                                                              \n"
      "    // Load 8x8 LHS tile                                            \n"
      "    ldp q0, q1, [x1], 32                                            \n"
      "    ldp q2, q3, [x1], 32                                            \n"
      "    // Load 8x8 RHS tile                                            \n"
      "    ldp q4, q5, [x2], 32                                            \n"
      "    ldp q6, q7, [x2], 32                                            \n"
      "    // Load 8x8 LHS tile                                            \n"
      "    ldp q8, q9, [x1], 32                                            \n"
      "    ldp q10, q11, [x1], 32                                          \n"
      "    // Load 8x8 RHS tile...                                         \n"
      "    ldp q12, q13, [x2], 32                                          \n"
      "    // ...second half loads is kept inside the loop below.          \n"
      "    //                                                              \n"
      "    // Multiply-accumulate, rows 0--1.                              \n"
      "    smmla v16.4s, v0.16b, v4.16b                                    \n"
      "    smmla v17.4s, v0.16b, v5.16b                                    \n"
      "    smmla v18.4s, v0.16b, v6.16b                                    \n"
      "    smmla v19.4s, v0.16b, v7.16b                                    \n"
      "                                                                    \n"
      "    // If K==2, jump to the epilogue.                               \n"
      "    b.le 5f                                                         \n"
      "                                                                    \n"
      "  4:                                                                \n"
      "    // Body of main loop, 2x partially unrolled, for when K>2.      \n"
      "    //                                                              \n"
      "    // Multiply-accumulate, rows 2--3.                              \n"
      "    smmla v20.4s, v1.16b, v4.16b                                    \n"
      "    smmla v21.4s, v1.16b, v5.16b                                    \n"
      "    smmla v22.4s, v1.16b, v6.16b                                    \n"
      "    smmla v23.4s, v1.16b, v7.16b                                    \n"
      "    ldp q14, q15, [x2], 32                                          \n"
      "    // Multiply-accumulate, rows 4--5.                              \n"
      "    smmla v24.4s, v2.16b, v4.16b                                    \n"
      "    smmla v25.4s, v2.16b, v5.16b                                    \n"
      "    smmla v26.4s, v2.16b, v6.16b                                    \n"
      "    smmla v27.4s, v2.16b, v7.16b                                    \n"
      "    ldp q0, q1, [x1], 32                                            \n"
      "    // Multiply-accumulate, rows 6--7.                              \n"
      "    smmla v28.4s, v3.16b, v4.16b                                    \n"
      "    smmla v29.4s, v3.16b, v5.16b                                    \n"
      "    smmla v30.4s, v3.16b, v6.16b                                    \n"
      "    smmla v31.4s, v3.16b, v7.16b                                    \n"
      "    ldp q2, q3, [x1], 32                                            \n"
      "    // Multiply-accumulate, rows 0--1.                              \n"
      "    smmla v16.4s, v8.16b, v12.16b                                   \n"
      "    smmla v17.4s, v8.16b, v13.16b                                   \n"
      "    smmla v18.4s, v8.16b, v14.16b                                   \n"
      "    smmla v19.4s, v8.16b, v15.16b                                   \n"
      "    ldp q4, q5, [x2], 32                                            \n"
      "    // Multiply-accumulate, rows 2--3.                              \n"
      "    smmla v20.4s, v9.16b, v12.16b                                   \n"
      "    smmla v21.4s, v9.16b, v13.16b                                   \n"
      "    smmla v22.4s, v9.16b, v14.16b                                   \n"
      "    smmla v23.4s, v9.16b, v15.16b                                   \n"
      "    ldp q6, q7, [x2], 32                                            \n"
      "    // Multiply-accumulate, rows 4--5.                              \n"
      "    smmla v24.4s, v10.16b, v12.16b                                  \n"
      "    smmla v25.4s, v10.16b, v13.16b                                  \n"
      "    smmla v26.4s, v10.16b, v14.16b                                  \n"
      "    smmla v27.4s, v10.16b, v15.16b                                  \n"
      "    ldp q8, q9, [x1], 32                                            \n"
      "    // Multiply-accumulate, rows 6--7.                              \n"
      "    smmla v28.4s, v11.16b, v12.16b                                  \n"
      "    smmla v29.4s, v11.16b, v13.16b                                  \n"
      "    smmla v30.4s, v11.16b, v14.16b                                  \n"
      "    smmla v31.4s, v11.16b, v15.16b                                  \n"
      "    ldp q10, q11, [x1], 32                                          \n"
      "    // Multiply-accumulate, rows 0--1.                              \n"
      "    smmla v16.4s, v0.16b, v4.16b                                    \n"
      "    smmla v17.4s, v0.16b, v5.16b                                    \n"
      "    ldp q12, q13, [x2], 32                                          \n"
      "    smmla v18.4s, v0.16b, v6.16b                                    \n"
      "    subs w3, w3, 2                                                  \n"
      "    smmla v19.4s, v0.16b, v7.16b                                    \n"
      "    b.gt 4b                                                         \n"
      "                                                                    \n"
      "  5:                                                                \n"
      "    // Epilogue of main loop, 2x partially unrolled, for when K>2.  \n"
      "    //                                                              \n"
      "    // Load last chunk of last RHS tile.                            \n"
      "    ldp q14, q15, [x2], 32                                          \n"
      "                                                                    \n"
      "    // Multiply-accumulate, rows 2--3.                              \n"
      "    smmla v20.4s, v1.16b, v4.16b                                    \n"
      "    smmla v21.4s, v1.16b, v5.16b                                    \n"
      "    smmla v22.4s, v1.16b, v6.16b                                    \n"
      "    smmla v23.4s, v1.16b, v7.16b                                    \n"
      "    // Multiply-accumulate, rows 4--5.                              \n"
      "    smmla v24.4s, v2.16b, v4.16b                                    \n"
      "    smmla v25.4s, v2.16b, v5.16b                                    \n"
      "    smmla v26.4s, v2.16b, v6.16b                                    \n"
      "    smmla v27.4s, v2.16b, v7.16b                                    \n"
      "    // Multiply-accumulate, rows 6--7.                              \n"
      "    smmla v28.4s, v3.16b, v4.16b                                    \n"
      "    smmla v29.4s, v3.16b, v5.16b                                    \n"
      "    smmla v30.4s, v3.16b, v6.16b                                    \n"
      "    smmla v31.4s, v3.16b, v7.16b                                    \n"
      "                                                                    \n"
      "    // Multiply-accumulate, rows 0--1.                              \n"
      "    smmla v16.4s, v8.16b, v12.16b                                   \n"
      "    smmla v17.4s, v8.16b, v13.16b                                   \n"
      "    smmla v18.4s, v8.16b, v14.16b                                   \n"
      "    smmla v19.4s, v8.16b, v15.16b                                   \n"
      "    // Multiply-accumulate, rows 2--3.                              \n"
      "    smmla v20.4s, v9.16b, v12.16b                                   \n"
      "    smmla v21.4s, v9.16b, v13.16b                                   \n"
      "    smmla v22.4s, v9.16b, v14.16b                                   \n"
      "    smmla v23.4s, v9.16b, v15.16b                                   \n"
      "    // Multiply-accumulate, rows 4--5.                              \n"
      "    smmla v24.4s, v10.16b, v12.16b                                  \n"
      "    smmla v25.4s, v10.16b, v13.16b                                  \n"
      "    smmla v26.4s, v10.16b, v14.16b                                  \n"
      "    smmla v27.4s, v10.16b, v15.16b                                  \n"
      "    // Multiply-accumulate, rows 6--7.                              \n"
      "    smmla v28.4s, v11.16b, v12.16b                                  \n"
      "    smmla v29.4s, v11.16b, v13.16b                                  \n"
      "    smmla v30.4s, v11.16b, v14.16b                                  \n"
      "    smmla v31.4s, v11.16b, v15.16b                                  \n"
      "                                                                    \n"
      "    // Finished accumulating? Then jump to final store.             \n"
      "    b.lt 7f                                                         \n"
      "    // Fall through.                                                \n"
      "                                                                    \n"
      "  6:                                                                \n"
      "    // Accumulate for a single K-value - used for either the        \n"
      "    // K==1 case or the final value of K for odd K>1.               \n"
      "                                                                    \n"
      "    // Load 8x8 LHS tile                                            \n"
      "    ldp q0, q1, [x1, 0]                                             \n"
      "    ldp q2, q3, [x1, 32]                                            \n"
      "    add x1, x1, 64                                                  \n"
      "    // Load 8x8 RHS tile                                            \n"
      "    ldp q4, q5, [x2, 0]                                             \n"
      "    ldp q6, q7, [x2, 32]                                            \n"
      "    add x2, x2, 64                                                  \n"
      "    // Multiply-accumulate, rows 0--1.                              \n"
      "    smmla v16.4s, v0.16b, v4.16b                                    \n"
      "    smmla v17.4s, v0.16b, v5.16b                                    \n"
      "    smmla v18.4s, v0.16b, v6.16b                                    \n"
      "    smmla v19.4s, v0.16b, v7.16b                                    \n"
      "    // Multiply-accumulate, rows 2--3.                              \n"
      "    smmla v20.4s, v1.16b, v4.16b                                    \n"
      "    smmla v21.4s, v1.16b, v5.16b                                    \n"
      "    smmla v22.4s, v1.16b, v6.16b                                    \n"
      "    smmla v23.4s, v1.16b, v7.16b                                    \n"
      "    // Multiply-accumulate, rows 4--5.                              \n"
      "    smmla v24.4s, v2.16b, v4.16b                                    \n"
      "    smmla v25.4s, v2.16b, v5.16b                                    \n"
      "    smmla v26.4s, v2.16b, v6.16b                                    \n"
      "    smmla v27.4s, v2.16b, v7.16b                                    \n"
      "    // Multiply-accumulate, rows 6--7.                              \n"
      "    smmla v28.4s, v3.16b, v4.16b                                    \n"
      "    smmla v29.4s, v3.16b, v5.16b                                    \n"
      "    smmla v30.4s, v3.16b, v6.16b                                    \n"
      "    smmla v31.4s, v3.16b, v7.16b                                    \n"
      "                                                                    \n"
      "  7:                                                                \n"
      "    // Done accumulating.                                           \n"
      "    //                                                              \n"
      "    // Swizzle back to row-major and store to destination.          \n"
      "    //                                                              \n"
      "    // Swizzle back to row-major, rows 0--1.                        \n"
      "    uzp1 v0.2d, v16.2d, v17.2d                                      \n"
      "    uzp1 v1.2d, v18.2d, v19.2d                                      \n"
      "    uzp2 v2.2d, v16.2d, v17.2d                                      \n"
      "    uzp2 v3.2d, v18.2d, v19.2d                                      \n"
      "    // Swizzle back to row-major, rows 2--3.                        \n"
      "    uzp1 v4.2d, v20.2d, v21.2d                                      \n"
      "    uzp1 v5.2d, v22.2d, v23.2d                                      \n"
      "    uzp2 v6.2d, v20.2d, v21.2d                                      \n"
      "    uzp2 v7.2d, v22.2d, v23.2d                                      \n"
      "    // Swizzle back to row-major, rows 4--5.                        \n"
      "    uzp1 v8.2d, v24.2d, v25.2d                                      \n"
      "    uzp1 v9.2d, v26.2d, v27.2d                                      \n"
      "    uzp2 v10.2d, v24.2d, v25.2d                                     \n"
      "    uzp2 v11.2d, v26.2d, v27.2d                                     \n"
      "    // Swizzle back to row-major, rows 6--7.                        \n"
      "    uzp1 v12.2d, v28.2d, v29.2d                                     \n"
      "    uzp1 v13.2d, v30.2d, v31.2d                                     \n"
      "    uzp2 v14.2d, v28.2d, v29.2d                                     \n"
      "    uzp2 v15.2d, v30.2d, v31.2d                                     \n"
      "    // Store rows 0--3 to destination.                              \n"
      "    stp q0, q1, [x0, 0]                                             \n"
      "    stp q2, q3, [x0, 32]                                            \n"
      "    stp q4, q5, [x0, 64]                                            \n"
      "    stp q6, q7, [x0, 96]                                            \n"
      "    stp q8, q9, [x0, 128]                                           \n"
      "    stp q10, q11, [x0, 160]                                         \n"
      "    stp q12, q13, [x0, 192]                                         \n"
      "    stp q14, q15, [x0, 224]                                         \n"
      : /*modified*/[K] "+r"(K), [lhs_ptr] "+r"(lhs_panel),
        [rhs_ptr] "+r"(rhs_panel)
      : /*unmodified*/[flags] "r"(flags), [out_ptr] "r"(out_tile),
        [accumulate_bit_pos] "i"(IREE_UK_FLAG_MMT4D_ACCUMULATE_BIT_POS)
      : /*clobbers*/ "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
        "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
        "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
        "v27", "v28", "v29", "v30", "v31");
}
#endif  // defined(IREE_UK_ENABLE_INLINE_ASM)
