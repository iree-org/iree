// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/arm_64/common_arm_64.h"
#include "iree/builtins/ukernel/arch/arm_64/mmt4d_arm_64_internal.h"

#ifdef __clang__
// Work around https://github.com/llvm/llvm-project/issues/64104
// Notes:
// 1. The key to this work-around is the "x" constraint on the C operand,
//    which restricts it to registers v0 .. v15, as opposed to the "w"
//    constraint used here for the A and B operands, allowing v0.. v31. See:
//      https://llvm.org/docs/LangRef.html#supported-constraint-code-list
// 2. The ({...}) syntax is GCC-compatible "statement expressions". See:
//      https://gcc.gnu.org/onlinedocs/gcc/Statement-Exprs.html
#define iree_vfmlalq_laneq_x_f16(INSTR, A, B, C, L) \
  ({                                                \
    asm(INSTR " %[a].4s, %[b].4h, %[c].h[" #L "]"   \
        : [a] "+w"(A)                               \
        : [b] "w"(B), [c] "x"(C)                    \
        :);                                         \
    A;                                              \
  })
#define iree_vfmlalq_laneq_low_f16(A, B, C, L) \
  iree_vfmlalq_laneq_x_f16("fmlal", A, B, C, L)
#define iree_vfmlalq_laneq_high_f16(A, B, C, L) \
  iree_vfmlalq_laneq_x_f16("fmlal2", A, B, C, L)
#else
#define iree_vfmlalq_laneq_low_f16(A, X, Y, L) vfmlalq_laneq_low_f16(A, X, Y, L)
#define iree_vfmlalq_laneq_high_f16(A, X, Y, L) \
  vfmlalq_laneq_high_f16(A, X, Y, L)
#endif

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_f16f16f32_1x8x1_to_8x8x1_arm_64_fp16fml(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  float* IREE_UK_RESTRICT out_ptr = out_tile;
  const float16_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const float16_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  float32x4_t acc[16];
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    IREE_UK_UNROLL for (int i = 0; i < 2 * M0; ++i) {
      acc[i] = vld1q_f32(out_ptr + 4 * i);
    }
  } else {
    IREE_UK_UNROLL for (int i = 0; i < 2 * M0; ++i) { acc[i] = vdupq_n_f32(0); }
  }
  for (int k = 0; k < params->K; ++k) {
    float16x8_t rhs = vld1q_f16(rhs_ptr);
    rhs_ptr += 8;
    float16x8_t lhs;
    if (M0 <= 2) {
      lhs = vld1q_dup_f16(lhs_ptr);
      if (M0 == 2) {
        lhs = vld1q_lane_f16(lhs_ptr + 1, lhs, 1);
      }
    } else if (M0 == 4) {
      lhs = vcombine_f16(vld1_f16(lhs_ptr), vdup_n_f16(0));
    } else {
      lhs = vld1q_f16(lhs_ptr);
    }
    lhs_ptr += M0;
    acc[0] = iree_vfmlalq_laneq_low_f16(acc[0], rhs, lhs, 0);
    acc[1] = iree_vfmlalq_laneq_high_f16(acc[1], rhs, lhs, 0);
    if (M0 == 1) continue;
    acc[2] = iree_vfmlalq_laneq_low_f16(acc[2], rhs, lhs, 1);
    acc[3] = iree_vfmlalq_laneq_high_f16(acc[3], rhs, lhs, 1);
    if (M0 == 2) continue;
    acc[4] = iree_vfmlalq_laneq_low_f16(acc[4], rhs, lhs, 2);
    acc[5] = iree_vfmlalq_laneq_high_f16(acc[5], rhs, lhs, 2);
    acc[6] = iree_vfmlalq_laneq_low_f16(acc[6], rhs, lhs, 3);
    acc[7] = iree_vfmlalq_laneq_high_f16(acc[7], rhs, lhs, 3);
    if (M0 == 4) continue;
    acc[8] = iree_vfmlalq_laneq_low_f16(acc[8], rhs, lhs, 4);
    acc[9] = iree_vfmlalq_laneq_high_f16(acc[9], rhs, lhs, 4);
    acc[10] = iree_vfmlalq_laneq_low_f16(acc[10], rhs, lhs, 5);
    acc[11] = iree_vfmlalq_laneq_high_f16(acc[11], rhs, lhs, 5);
    acc[12] = iree_vfmlalq_laneq_low_f16(acc[12], rhs, lhs, 6);
    acc[13] = iree_vfmlalq_laneq_high_f16(acc[13], rhs, lhs, 6);
    acc[14] = iree_vfmlalq_laneq_low_f16(acc[14], rhs, lhs, 7);
    acc[15] = iree_vfmlalq_laneq_high_f16(acc[15], rhs, lhs, 7);
  }
  IREE_UK_UNROLL for (int i = 0; i < 2 * M0; ++i) {
    vst1q_f32(out_ptr + 4 * i, acc[i]);
  }
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0_1_2_4_8(
    iree_uk_mmt4d_tile_f16f16f32_1x8x1_to_8x8x1_arm_64_fp16fml,
    iree_uk_mmt4d_tile_f16f16f32_1x8x1_arm_64_fp16fml,
    iree_uk_mmt4d_tile_f16f16f32_2x8x1_arm_64_fp16fml,
    iree_uk_mmt4d_tile_f16f16f32_4x8x1_arm_64_fp16fml,
    iree_uk_mmt4d_tile_f16f16f32_8x8x1_arm_64_fp16fml)
