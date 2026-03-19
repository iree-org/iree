// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <riscv_vector.h>

#include "iree/builtins/ukernel/arch/riscv_64/common_riscv_64.h"
#include "iree/builtins/ukernel/arch/riscv_64/mmt4d_riscv_64_internal.h"

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_s8s8s32_1xXXx1_to_8xXXx1_riscv_64_v(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 8);
  const iree_uk_int8_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;

  vint32m4_t acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
  int N0 = params->N0;
  size_t vl = N0;

#define IREE_UK_RISCV_64_VWACC_I8(ACC, LHS, RHS, VL)      \
  do {                                                     \
    vint16m2_t prod = __riscv_vwmul_vx_i16m2(RHS, LHS, VL); \
    ACC = __riscv_vwadd_wv_i32m4(ACC, prod, VL);          \
  } while (0)

  if (M0 == 1) {
    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
      acc0 = __riscv_vle32_v_i32m4(out_ptr, vl);
    } else {
      acc0 = __riscv_vmv_v_x_i32m4(0, vl);
    }
    for (int k = 0; k < params->K; ++k) {
      vint8m1_t rhs = __riscv_vle8_v_i8m1(rhs_ptr, vl);
      rhs_ptr += N0;
      iree_uk_int8_t lhs = *lhs_ptr++;
      IREE_UK_RISCV_64_VWACC_I8(acc0, lhs, rhs, vl);
    }
    __riscv_vse32_v_i32m4(out_ptr, acc0, vl);
  } else if (M0 == 2) {
    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
      acc0 = __riscv_vle32_v_i32m4(out_ptr, vl);
      acc1 = __riscv_vle32_v_i32m4(out_ptr + N0, vl);
    } else {
      acc0 = __riscv_vmv_v_x_i32m4(0, vl);
      acc1 = __riscv_vmv_v_x_i32m4(0, vl);
    }
    for (int k = 0; k < params->K; ++k) {
      vint8m1_t rhs = __riscv_vle8_v_i8m1(rhs_ptr, vl);
      rhs_ptr += N0;
      iree_uk_int8_t lhs0 = *lhs_ptr++;
      iree_uk_int8_t lhs1 = *lhs_ptr++;
      IREE_UK_RISCV_64_VWACC_I8(acc0, lhs0, rhs, vl);
      IREE_UK_RISCV_64_VWACC_I8(acc1, lhs1, rhs, vl);
    }
    __riscv_vse32_v_i32m4(out_ptr, acc0, vl);
    __riscv_vse32_v_i32m4(out_ptr + N0, acc1, vl);
  } else if (M0 == 4) {
    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
      acc0 = __riscv_vle32_v_i32m4(out_ptr, vl);
      acc1 = __riscv_vle32_v_i32m4(out_ptr + N0, vl);
      acc2 = __riscv_vle32_v_i32m4(out_ptr + N0 * 2, vl);
      acc3 = __riscv_vle32_v_i32m4(out_ptr + N0 * 3, vl);
    } else {
      acc0 = __riscv_vmv_v_x_i32m4(0, vl);
      acc1 = __riscv_vmv_v_x_i32m4(0, vl);
      acc2 = __riscv_vmv_v_x_i32m4(0, vl);
      acc3 = __riscv_vmv_v_x_i32m4(0, vl);
    }
    for (int k = 0; k < params->K; ++k) {
      vint8m1_t rhs = __riscv_vle8_v_i8m1(rhs_ptr, vl);
      rhs_ptr += N0;
      iree_uk_int8_t lhs0 = *lhs_ptr++;
      iree_uk_int8_t lhs1 = *lhs_ptr++;
      iree_uk_int8_t lhs2 = *lhs_ptr++;
      iree_uk_int8_t lhs3 = *lhs_ptr++;
      IREE_UK_RISCV_64_VWACC_I8(acc0, lhs0, rhs, vl);
      IREE_UK_RISCV_64_VWACC_I8(acc1, lhs1, rhs, vl);
      IREE_UK_RISCV_64_VWACC_I8(acc2, lhs2, rhs, vl);
      IREE_UK_RISCV_64_VWACC_I8(acc3, lhs3, rhs, vl);
    }
    __riscv_vse32_v_i32m4(out_ptr, acc0, vl);
    __riscv_vse32_v_i32m4(out_ptr + N0, acc1, vl);
    __riscv_vse32_v_i32m4(out_ptr + N0 * 2, acc2, vl);
    __riscv_vse32_v_i32m4(out_ptr + N0 * 3, acc3, vl);
  } else if (M0 == 7) {
    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
      acc0 = __riscv_vle32_v_i32m4(out_ptr, vl);
      acc1 = __riscv_vle32_v_i32m4(out_ptr + N0, vl);
      acc2 = __riscv_vle32_v_i32m4(out_ptr + N0 * 2, vl);
      acc3 = __riscv_vle32_v_i32m4(out_ptr + N0 * 3, vl);
      acc4 = __riscv_vle32_v_i32m4(out_ptr + N0 * 4, vl);
      acc5 = __riscv_vle32_v_i32m4(out_ptr + N0 * 5, vl);
      acc6 = __riscv_vle32_v_i32m4(out_ptr + N0 * 6, vl);
    } else {
      acc0 = __riscv_vmv_v_x_i32m4(0, vl);
      acc1 = __riscv_vmv_v_x_i32m4(0, vl);
      acc2 = __riscv_vmv_v_x_i32m4(0, vl);
      acc3 = __riscv_vmv_v_x_i32m4(0, vl);
      acc4 = __riscv_vmv_v_x_i32m4(0, vl);
      acc5 = __riscv_vmv_v_x_i32m4(0, vl);
      acc6 = __riscv_vmv_v_x_i32m4(0, vl);
    }
    for (int k = 0; k < params->K; ++k) {
      vint8m1_t rhs = __riscv_vle8_v_i8m1(rhs_ptr, vl);
      rhs_ptr += N0;
      iree_uk_int8_t lhs0 = *lhs_ptr++;
      iree_uk_int8_t lhs1 = *lhs_ptr++;
      iree_uk_int8_t lhs2 = *lhs_ptr++;
      iree_uk_int8_t lhs3 = *lhs_ptr++;
      iree_uk_int8_t lhs4 = *lhs_ptr++;
      iree_uk_int8_t lhs5 = *lhs_ptr++;
      iree_uk_int8_t lhs6 = *lhs_ptr++;
      IREE_UK_RISCV_64_VWACC_I8(acc0, lhs0, rhs, vl);
      IREE_UK_RISCV_64_VWACC_I8(acc1, lhs1, rhs, vl);
      IREE_UK_RISCV_64_VWACC_I8(acc2, lhs2, rhs, vl);
      IREE_UK_RISCV_64_VWACC_I8(acc3, lhs3, rhs, vl);
      IREE_UK_RISCV_64_VWACC_I8(acc4, lhs4, rhs, vl);
      IREE_UK_RISCV_64_VWACC_I8(acc5, lhs5, rhs, vl);
      IREE_UK_RISCV_64_VWACC_I8(acc6, lhs6, rhs, vl);
    }
    __riscv_vse32_v_i32m4(out_ptr, acc0, vl);
    __riscv_vse32_v_i32m4(out_ptr + N0, acc1, vl);
    __riscv_vse32_v_i32m4(out_ptr + N0 * 2, acc2, vl);
    __riscv_vse32_v_i32m4(out_ptr + N0 * 3, acc3, vl);
    __riscv_vse32_v_i32m4(out_ptr + N0 * 4, acc4, vl);
    __riscv_vse32_v_i32m4(out_ptr + N0 * 5, acc5, vl);
    __riscv_vse32_v_i32m4(out_ptr + N0 * 6, acc6, vl);
  } else if (M0 == 8) {
    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
      acc0 = __riscv_vle32_v_i32m4(out_ptr, vl);
      acc1 = __riscv_vle32_v_i32m4(out_ptr + N0, vl);
      acc2 = __riscv_vle32_v_i32m4(out_ptr + N0 * 2, vl);
      acc3 = __riscv_vle32_v_i32m4(out_ptr + N0 * 3, vl);
      acc4 = __riscv_vle32_v_i32m4(out_ptr + N0 * 4, vl);
      acc5 = __riscv_vle32_v_i32m4(out_ptr + N0 * 5, vl);
      acc6 = __riscv_vle32_v_i32m4(out_ptr + N0 * 6, vl);
      acc7 = __riscv_vle32_v_i32m4(out_ptr + N0 * 7, vl);
    } else {
      acc0 = __riscv_vmv_v_x_i32m4(0, vl);
      acc1 = __riscv_vmv_v_x_i32m4(0, vl);
      acc2 = __riscv_vmv_v_x_i32m4(0, vl);
      acc3 = __riscv_vmv_v_x_i32m4(0, vl);
      acc4 = __riscv_vmv_v_x_i32m4(0, vl);
      acc5 = __riscv_vmv_v_x_i32m4(0, vl);
      acc6 = __riscv_vmv_v_x_i32m4(0, vl);
      acc7 = __riscv_vmv_v_x_i32m4(0, vl);
    }
    for (int k = 0; k < params->K; ++k) {
      vint8m1_t rhs = __riscv_vle8_v_i8m1(rhs_ptr, vl);
      rhs_ptr += N0;
      iree_uk_int8_t lhs0 = *lhs_ptr++;
      iree_uk_int8_t lhs1 = *lhs_ptr++;
      iree_uk_int8_t lhs2 = *lhs_ptr++;
      iree_uk_int8_t lhs3 = *lhs_ptr++;
      iree_uk_int8_t lhs4 = *lhs_ptr++;
      iree_uk_int8_t lhs5 = *lhs_ptr++;
      iree_uk_int8_t lhs6 = *lhs_ptr++;
      iree_uk_int8_t lhs7 = *lhs_ptr++;
      IREE_UK_RISCV_64_VWACC_I8(acc0, lhs0, rhs, vl);
      IREE_UK_RISCV_64_VWACC_I8(acc1, lhs1, rhs, vl);
      IREE_UK_RISCV_64_VWACC_I8(acc2, lhs2, rhs, vl);
      IREE_UK_RISCV_64_VWACC_I8(acc3, lhs3, rhs, vl);
      IREE_UK_RISCV_64_VWACC_I8(acc4, lhs4, rhs, vl);
      IREE_UK_RISCV_64_VWACC_I8(acc5, lhs5, rhs, vl);
      IREE_UK_RISCV_64_VWACC_I8(acc6, lhs6, rhs, vl);
      IREE_UK_RISCV_64_VWACC_I8(acc7, lhs7, rhs, vl);
    }
    __riscv_vse32_v_i32m4(out_ptr, acc0, vl);
    __riscv_vse32_v_i32m4(out_ptr + N0, acc1, vl);
    __riscv_vse32_v_i32m4(out_ptr + N0 * 2, acc2, vl);
    __riscv_vse32_v_i32m4(out_ptr + N0 * 3, acc3, vl);
    __riscv_vse32_v_i32m4(out_ptr + N0 * 4, acc4, vl);
    __riscv_vse32_v_i32m4(out_ptr + N0 * 5, acc5, vl);
    __riscv_vse32_v_i32m4(out_ptr + N0 * 6, acc6, vl);
    __riscv_vse32_v_i32m4(out_ptr + N0 * 7, acc7, vl);
  }

#undef IREE_UK_RISCV_64_VWACC_I8
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1xXXx1_to_8xXXx1_riscv_64_v,
    iree_uk_mmt4d_tile_s8s8s32_1xXXx1_riscv_64_v, 1)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1xXXx1_to_8xXXx1_riscv_64_v,
    iree_uk_mmt4d_tile_s8s8s32_2xXXx1_riscv_64_v, 2)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1xXXx1_to_8xXXx1_riscv_64_v,
    iree_uk_mmt4d_tile_s8s8s32_4xXXx1_riscv_64_v, 4)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1xXXx1_to_8xXXx1_riscv_64_v,
    iree_uk_mmt4d_tile_s8s8s32_7xXXx1_riscv_64_v, 7)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1xXXx1_to_8xXXx1_riscv_64_v,
    iree_uk_mmt4d_tile_s8s8s32_8xXXx1_riscv_64_v, 8)
