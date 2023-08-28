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
//    constraint used here for the A and B operands, allowing v0 .. v31. See:
//      https://llvm.org/docs/LangRef.html#supported-constraint-code-list
// 2. The ({...}) syntax is GCC-compatible "statement expressions". See:
//      https://gcc.gnu.org/onlinedocs/gcc/Statement-Exprs.html
#define iree_workaround_vfmlalq_laneq_x_f16(INSTR, A, B, C, L) \
  ({                                                           \
    asm(INSTR " %[a].4s, %[b].4h, %[c].h[" #L "]"              \
        : [a] "+w"(A)                                          \
        : [b] "w"(B), [c] "x"(C)                               \
        :);                                                    \
    A;                                                         \
  })
#define iree_workaround_vfmlalq_laneq_low_f16(A, B, C, L) \
  iree_workaround_vfmlalq_laneq_x_f16("fmlal", A, B, C, L)
#define iree_workaround_vfmlalq_laneq_high_f16(A, B, C, L) \
  iree_workaround_vfmlalq_laneq_x_f16("fmlal2", A, B, C, L)
#else
#define iree_workaround_vfmlalq_laneq_low_f16(A, X, Y, L) \
  vfmlalq_laneq_low_f16(A, X, Y, L)
#define iree_workaround_vfmlalq_laneq_high_f16(A, X, Y, L) \
  vfmlalq_laneq_high_f16(A, X, Y, L)
#endif

void iree_uk_mmt4d_tile_f16f16f32_8x8x1_arm_64_fp16fml(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel, iree_uk_int32_t K,
    iree_uk_uint32_t flags, const iree_uk_mmt4d_params_t* params) {
  (void)params;
  float* IREE_UK_RESTRICT out_ptr = out_tile;
  const float16_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const float16_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  float32x4_t acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10,
      acc11, acc12, acc13, acc14, acc15;
  if (flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    acc0 = vld1q_f32(out_ptr + 4 * 0);
    acc1 = vld1q_f32(out_ptr + 4 * 1);
    acc2 = vld1q_f32(out_ptr + 4 * 2);
    acc3 = vld1q_f32(out_ptr + 4 * 3);
    acc4 = vld1q_f32(out_ptr + 4 * 4);
    acc5 = vld1q_f32(out_ptr + 4 * 5);
    acc6 = vld1q_f32(out_ptr + 4 * 6);
    acc7 = vld1q_f32(out_ptr + 4 * 7);
    acc8 = vld1q_f32(out_ptr + 4 * 8);
    acc9 = vld1q_f32(out_ptr + 4 * 9);
    acc10 = vld1q_f32(out_ptr + 4 * 10);
    acc11 = vld1q_f32(out_ptr + 4 * 11);
    acc12 = vld1q_f32(out_ptr + 4 * 12);
    acc13 = vld1q_f32(out_ptr + 4 * 13);
    acc14 = vld1q_f32(out_ptr + 4 * 14);
    acc15 = vld1q_f32(out_ptr + 4 * 15);
  } else {
    acc0 = vdupq_n_f32(0);
    acc1 = vdupq_n_f32(0);
    acc2 = vdupq_n_f32(0);
    acc3 = vdupq_n_f32(0);
    acc4 = vdupq_n_f32(0);
    acc5 = vdupq_n_f32(0);
    acc6 = vdupq_n_f32(0);
    acc7 = vdupq_n_f32(0);
    acc8 = vdupq_n_f32(0);
    acc9 = vdupq_n_f32(0);
    acc10 = vdupq_n_f32(0);
    acc11 = vdupq_n_f32(0);
    acc12 = vdupq_n_f32(0);
    acc13 = vdupq_n_f32(0);
    acc14 = vdupq_n_f32(0);
    acc15 = vdupq_n_f32(0);
  }
  IREE_UK_ASSUME(K >= 1);
  for (int k = 0; k < K; ++k) {
    float16x8_t lhs = vld1q_f16(lhs_ptr);
    lhs_ptr += 8;
    float16x8_t rhs = vld1q_f16(rhs_ptr);
    rhs_ptr += 8;
    acc0 = iree_workaround_vfmlalq_laneq_low_f16(acc0, rhs, lhs, 0);
    acc1 = iree_workaround_vfmlalq_laneq_high_f16(acc1, rhs, lhs, 0);
    acc2 = iree_workaround_vfmlalq_laneq_low_f16(acc2, rhs, lhs, 1);
    acc3 = iree_workaround_vfmlalq_laneq_high_f16(acc3, rhs, lhs, 1);
    acc4 = iree_workaround_vfmlalq_laneq_low_f16(acc4, rhs, lhs, 2);
    acc5 = iree_workaround_vfmlalq_laneq_high_f16(acc5, rhs, lhs, 2);
    acc6 = iree_workaround_vfmlalq_laneq_low_f16(acc6, rhs, lhs, 3);
    acc7 = iree_workaround_vfmlalq_laneq_high_f16(acc7, rhs, lhs, 3);
    acc8 = iree_workaround_vfmlalq_laneq_low_f16(acc8, rhs, lhs, 4);
    acc9 = iree_workaround_vfmlalq_laneq_high_f16(acc9, rhs, lhs, 4);
    acc10 = iree_workaround_vfmlalq_laneq_low_f16(acc10, rhs, lhs, 5);
    acc11 = iree_workaround_vfmlalq_laneq_high_f16(acc11, rhs, lhs, 5);
    acc12 = iree_workaround_vfmlalq_laneq_low_f16(acc12, rhs, lhs, 6);
    acc13 = iree_workaround_vfmlalq_laneq_high_f16(acc13, rhs, lhs, 6);
    acc14 = iree_workaround_vfmlalq_laneq_low_f16(acc14, rhs, lhs, 7);
    acc15 = iree_workaround_vfmlalq_laneq_high_f16(acc15, rhs, lhs, 7);
  }
  vst1q_f32(out_ptr + 4 * 0, acc0);
  vst1q_f32(out_ptr + 4 * 1, acc1);
  vst1q_f32(out_ptr + 4 * 2, acc2);
  vst1q_f32(out_ptr + 4 * 3, acc3);
  vst1q_f32(out_ptr + 4 * 4, acc4);
  vst1q_f32(out_ptr + 4 * 5, acc5);
  vst1q_f32(out_ptr + 4 * 6, acc6);
  vst1q_f32(out_ptr + 4 * 7, acc7);
  vst1q_f32(out_ptr + 4 * 8, acc8);
  vst1q_f32(out_ptr + 4 * 9, acc9);
  vst1q_f32(out_ptr + 4 * 10, acc10);
  vst1q_f32(out_ptr + 4 * 11, acc11);
  vst1q_f32(out_ptr + 4 * 12, acc12);
  vst1q_f32(out_ptr + 4 * 13, acc13);
  vst1q_f32(out_ptr + 4 * 14, acc14);
  vst1q_f32(out_ptr + 4 * 15, acc15);
}
