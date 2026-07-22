// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <riscv_vector.h>

#include "iree/builtins/ukernel/arch/riscv_64/common_riscv_64.h"
#include "iree/builtins/ukernel/arch/riscv_64/mmt4d_riscv_64_internal.h"

// bf16bf16f32 tile using the Zvfbfwma bf16 widening multiply-accumulate
// (`vfwmaccbf16`). Structurally identical to the f16f16f32 path in
// mmt4d_riscv_64_zvfh.c, with the f16 vector/scalar types swapped for bf16 and
// the widening MAC swapped for its bf16 counterpart. The accumulator is f32 in
// both, so there is no accumulator-type branching and no narrowing store:
// unlike f16 (native f16 arithmetic via Zvfh), bf16 has no non-widening vector
// FMA, so only the bf16bf16f32 case is provided here.
IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_bf16bf16f32_1xXXx1_to_7xXXx1_riscv_64_zvfbfwma(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 7);
  const __bf16* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const __bf16* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  float* IREE_UK_RESTRICT out_ptr = out_tile;

  vfloat32m4_t acc0, acc1, acc2, acc3, acc4, acc5, acc6;
  int N0 = params->N0;
  size_t vl = N0;

  if (M0 == 1) {
    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
      acc0 = __riscv_vle32_v_f32m4(out_ptr, vl);
    } else {
      acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    }
    for (int k = 0; k < params->K; ++k) {
      vbfloat16m2_t rhs = __riscv_vle16_v_bf16m2(rhs_ptr, vl);
      rhs_ptr += N0;
      __bf16 lhs0 = *lhs_ptr++;

      acc0 = __riscv_vfwmaccbf16_vf_f32m4(acc0, lhs0, rhs, vl);
    }
    __riscv_vse32_v_f32m4(out_ptr, acc0, vl);
  } else if (M0 == 2) {
    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
      acc0 = __riscv_vle32_v_f32m4(out_ptr, vl);
      acc1 = __riscv_vle32_v_f32m4(out_ptr + N0, vl);
    } else {
      acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    }
    for (int k = 0; k < params->K; ++k) {
      vbfloat16m2_t rhs = __riscv_vle16_v_bf16m2(rhs_ptr, vl);
      rhs_ptr += N0;
      __bf16 lhs0 = *lhs_ptr++;
      __bf16 lhs1 = *lhs_ptr++;

      acc0 = __riscv_vfwmaccbf16_vf_f32m4(acc0, lhs0, rhs, vl);
      acc1 = __riscv_vfwmaccbf16_vf_f32m4(acc1, lhs1, rhs, vl);
    }
    __riscv_vse32_v_f32m4(out_ptr, acc0, vl);
    __riscv_vse32_v_f32m4(out_ptr + N0, acc1, vl);
  } else if (M0 == 4) {
    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
      acc0 = __riscv_vle32_v_f32m4(out_ptr, vl);
      acc1 = __riscv_vle32_v_f32m4(out_ptr + N0, vl);
      acc2 = __riscv_vle32_v_f32m4(out_ptr + N0 * 2, vl);
      acc3 = __riscv_vle32_v_f32m4(out_ptr + N0 * 3, vl);
    } else {
      acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      acc2 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      acc3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    }
    for (int k = 0; k < params->K; ++k) {
      vbfloat16m2_t rhs = __riscv_vle16_v_bf16m2(rhs_ptr, vl);
      rhs_ptr += N0;
      __bf16 lhs0 = *lhs_ptr++;
      __bf16 lhs1 = *lhs_ptr++;
      __bf16 lhs2 = *lhs_ptr++;
      __bf16 lhs3 = *lhs_ptr++;

      acc0 = __riscv_vfwmaccbf16_vf_f32m4(acc0, lhs0, rhs, vl);
      acc1 = __riscv_vfwmaccbf16_vf_f32m4(acc1, lhs1, rhs, vl);
      acc2 = __riscv_vfwmaccbf16_vf_f32m4(acc2, lhs2, rhs, vl);
      acc3 = __riscv_vfwmaccbf16_vf_f32m4(acc3, lhs3, rhs, vl);
    }
    __riscv_vse32_v_f32m4(out_ptr, acc0, vl);
    __riscv_vse32_v_f32m4(out_ptr + N0, acc1, vl);
    __riscv_vse32_v_f32m4(out_ptr + N0 * 2, acc2, vl);
    __riscv_vse32_v_f32m4(out_ptr + N0 * 3, acc3, vl);
  } else if (M0 == 7) {
    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
      acc0 = __riscv_vle32_v_f32m4(out_ptr, vl);
      acc1 = __riscv_vle32_v_f32m4(out_ptr + N0, vl);
      acc2 = __riscv_vle32_v_f32m4(out_ptr + N0 * 2, vl);
      acc3 = __riscv_vle32_v_f32m4(out_ptr + N0 * 3, vl);
      acc4 = __riscv_vle32_v_f32m4(out_ptr + N0 * 4, vl);
      acc5 = __riscv_vle32_v_f32m4(out_ptr + N0 * 5, vl);
      acc6 = __riscv_vle32_v_f32m4(out_ptr + N0 * 6, vl);
    } else {
      acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      acc2 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      acc3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      acc4 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      acc5 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      acc6 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    }
    for (int k = 0; k < params->K; ++k) {
      vbfloat16m2_t rhs = __riscv_vle16_v_bf16m2(rhs_ptr, vl);
      rhs_ptr += N0;
      __bf16 lhs0 = *lhs_ptr++;
      __bf16 lhs1 = *lhs_ptr++;
      __bf16 lhs2 = *lhs_ptr++;
      __bf16 lhs3 = *lhs_ptr++;
      __bf16 lhs4 = *lhs_ptr++;
      __bf16 lhs5 = *lhs_ptr++;
      __bf16 lhs6 = *lhs_ptr++;

      acc0 = __riscv_vfwmaccbf16_vf_f32m4(acc0, lhs0, rhs, vl);
      acc1 = __riscv_vfwmaccbf16_vf_f32m4(acc1, lhs1, rhs, vl);
      acc2 = __riscv_vfwmaccbf16_vf_f32m4(acc2, lhs2, rhs, vl);
      acc3 = __riscv_vfwmaccbf16_vf_f32m4(acc3, lhs3, rhs, vl);
      acc4 = __riscv_vfwmaccbf16_vf_f32m4(acc4, lhs4, rhs, vl);
      acc5 = __riscv_vfwmaccbf16_vf_f32m4(acc5, lhs5, rhs, vl);
      acc6 = __riscv_vfwmaccbf16_vf_f32m4(acc6, lhs6, rhs, vl);
    }
    __riscv_vse32_v_f32m4(out_ptr, acc0, vl);
    __riscv_vse32_v_f32m4(out_ptr + N0, acc1, vl);
    __riscv_vse32_v_f32m4(out_ptr + N0 * 2, acc2, vl);
    __riscv_vse32_v_f32m4(out_ptr + N0 * 3, acc3, vl);
    __riscv_vse32_v_f32m4(out_ptr + N0 * 4, acc4, vl);
    __riscv_vse32_v_f32m4(out_ptr + N0 * 5, acc5, vl);
    __riscv_vse32_v_f32m4(out_ptr + N0 * 6, acc6, vl);
  }
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_bf16bf16f32_1xXXx1_to_7xXXx1_riscv_64_zvfbfwma,
    iree_uk_mmt4d_tile_bf16bf16f32_1xXXx1_riscv_64_zvfbfwma, 1)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_bf16bf16f32_1xXXx1_to_7xXXx1_riscv_64_zvfbfwma,
    iree_uk_mmt4d_tile_bf16bf16f32_2xXXx1_riscv_64_zvfbfwma, 2)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_bf16bf16f32_1xXXx1_to_7xXXx1_riscv_64_zvfbfwma,
    iree_uk_mmt4d_tile_bf16bf16f32_4xXXx1_riscv_64_zvfbfwma, 4)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_bf16bf16f32_1xXXx1_to_7xXXx1_riscv_64_zvfbfwma,
    iree_uk_mmt4d_tile_bf16bf16f32_7xXXx1_riscv_64_zvfbfwma, 7)
