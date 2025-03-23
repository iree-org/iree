// Copyright 2025 10xEngineers
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/riscv_64/common_riscv_64.h"
#include "iree/builtins/ukernel/arch/riscv_64/mmt4d_riscv_64_internal.h"

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_f32f32f32_1x32x1_to_7x32x1_riscv_64(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 7);
  const float* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const float* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  float* IREE_UK_RESTRICT out_ptr = out_tile;

  vfloat32m4_t acc0, acc1, acc2, acc3, acc4, acc5, acc6;

  size_t vlmax =  __riscv_vsetvlmax_e32m4();
  if (M0 == 1) {
    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
      acc0 =  __riscv_vle32_v_f32m4(out_ptr, vlmax);
    } else {
      acc0 =  __riscv_vfmv_v_f_f32m4(0.0, vlmax);
    }
    for (int k = 0; k < params->K; ++k) {
      vfloat32m4_t rhs = __riscv_vle32_v_f32m4(rhs_ptr, vlmax);
      rhs_ptr += 32;
      float lhs = *lhs_ptr++;
      acc0 = __riscv_vfmacc_vf_f32m4(acc0, lhs, rhs, vlmax);

    }
    __riscv_vse32_v_f32m4(out_ptr, acc0, vlmax);
  }

  else if (M0 == 2) {
    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
      acc0 =  __riscv_vle32_v_f32m4(out_ptr, vlmax);
      acc1 =  __riscv_vle32_v_f32m4(out_ptr + 32 , vlmax);
    } else {
      acc0 =  __riscv_vfmv_v_f_f32m4(0.0, vlmax);
      acc1 =  __riscv_vfmv_v_f_f32m4(0.0, vlmax);
    }
    for (int k = 0; k < params->K; ++k) {
      vfloat32m4_t rhs = __riscv_vle32_v_f32m4(rhs_ptr, vlmax);
      rhs_ptr += 32;
      float lhs0 = *lhs_ptr++;
      float lhs1 = *lhs_ptr++;
      acc0 = __riscv_vfmacc_vf_f32m4(acc0, lhs0, rhs, vlmax);
      acc1 = __riscv_vfmacc_vf_f32m4(acc1, lhs1, rhs, vlmax);
    }
    __riscv_vse32_v_f32m4(out_ptr, acc0, vlmax);
    __riscv_vse32_v_f32m4(out_ptr + 32, acc1, vlmax);
  }
  else if (M0 == 4) {
    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
      acc0 =  __riscv_vle32_v_f32m4(out_ptr, vlmax);
      acc1 =  __riscv_vle32_v_f32m4(out_ptr + 32 , vlmax);
      acc2 =  __riscv_vle32_v_f32m4(out_ptr + 64, vlmax);
      acc3 =  __riscv_vle32_v_f32m4(out_ptr + 96 , vlmax);
    } else {
      acc0 =  __riscv_vfmv_v_f_f32m4(0.0, vlmax);
      acc1 =  __riscv_vfmv_v_f_f32m4(0.0, vlmax);
      acc2 =  __riscv_vfmv_v_f_f32m4(0.0, vlmax);
      acc3 =  __riscv_vfmv_v_f_f32m4(0.0, vlmax);
    }
    for (int k = 0; k < params->K; ++k) {
      vfloat32m4_t rhs = __riscv_vle32_v_f32m4(rhs_ptr, vlmax);
      rhs_ptr += 32;
      float lhs0 = *lhs_ptr++;
      float lhs1 = *lhs_ptr++;
      float lhs2 = *lhs_ptr++;
      float lhs3 = *lhs_ptr++;
      acc0 = __riscv_vfmacc_vf_f32m4(acc0, lhs0, rhs, vlmax);
      acc1 = __riscv_vfmacc_vf_f32m4(acc1, lhs1, rhs, vlmax);
      acc2 = __riscv_vfmacc_vf_f32m4(acc2, lhs2, rhs, vlmax);
      acc3 = __riscv_vfmacc_vf_f32m4(acc3, lhs3, rhs, vlmax);
    }
    __riscv_vse32_v_f32m4(out_ptr, acc0, vlmax);
    __riscv_vse32_v_f32m4(out_ptr + 32, acc1, vlmax);
    __riscv_vse32_v_f32m4(out_ptr + 64, acc2, vlmax);
    __riscv_vse32_v_f32m4(out_ptr + 96, acc3, vlmax);
  }
  
  else if (M0 == 7) {
    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
      acc0 = __riscv_vle32_v_f32m4(out_ptr, vlmax);
      acc1 = __riscv_vle32_v_f32m4(out_ptr + 32, vlmax);
      acc2 = __riscv_vle32_v_f32m4(out_ptr + 64, vlmax);
      acc3 = __riscv_vle32_v_f32m4(out_ptr + 96, vlmax);
      acc4 = __riscv_vle32_v_f32m4(out_ptr + 128, vlmax);
      acc5 = __riscv_vle32_v_f32m4(out_ptr + 160, vlmax);
      acc6 = __riscv_vle32_v_f32m4(out_ptr + 192, vlmax);
    } else {
      acc0 = __riscv_vfmv_v_f_f32m4(0.0, vlmax);
      acc1 = __riscv_vfmv_v_f_f32m4(0.0, vlmax);
      acc2 = __riscv_vfmv_v_f_f32m4(0.0, vlmax);
      acc3 = __riscv_vfmv_v_f_f32m4(0.0, vlmax);
      acc4 = __riscv_vfmv_v_f_f32m4(0.0, vlmax);
      acc5 = __riscv_vfmv_v_f_f32m4(0.0, vlmax);
      acc6 = __riscv_vfmv_v_f_f32m4(0.0, vlmax);
    }
    for (int k = 0; k < params->K; ++k) {
      vfloat32m4_t rhs = __riscv_vle32_v_f32m4(rhs_ptr, vlmax);
      rhs_ptr += 32;
      float lhs0 = *lhs_ptr++;
      float lhs1 = *lhs_ptr++; 
      float lhs2 = *lhs_ptr++; 
      float lhs3 = *lhs_ptr++; 
      float lhs4 = *lhs_ptr++; 
      float lhs5 = *lhs_ptr++;
      float lhs6 = *lhs_ptr++; 
      acc0 = __riscv_vfmacc_vf_f32m4(acc0, lhs0, rhs, vlmax);
      acc1 = __riscv_vfmacc_vf_f32m4(acc1, lhs1, rhs, vlmax);
      acc2 = __riscv_vfmacc_vf_f32m4(acc2, lhs2, rhs, vlmax);
      acc3 = __riscv_vfmacc_vf_f32m4(acc3, lhs3, rhs, vlmax);
      acc4 = __riscv_vfmacc_vf_f32m4(acc4, lhs4, rhs, vlmax);
      acc5 = __riscv_vfmacc_vf_f32m4(acc5, lhs5, rhs, vlmax);
      acc6 = __riscv_vfmacc_vf_f32m4(acc6, lhs6, rhs, vlmax);
    }
    __riscv_vse32_v_f32m4(out_ptr, acc0, vlmax);
    __riscv_vse32_v_f32m4(out_ptr + 32, acc1, vlmax);
    __riscv_vse32_v_f32m4(out_ptr + 64, acc2, vlmax);
    __riscv_vse32_v_f32m4(out_ptr + 96, acc3, vlmax);
    __riscv_vse32_v_f32m4(out_ptr + 128, acc4, vlmax);
    __riscv_vse32_v_f32m4(out_ptr + 160, acc5, vlmax);
    __riscv_vse32_v_f32m4(out_ptr + 192, acc6, vlmax);
  }
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f32f32f32_1x32x1_to_7x32x1_riscv_64,
    iree_uk_mmt4d_tile_f32f32f32_1x32x1_riscv_64, 1)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f32f32f32_1x32x1_to_7x32x1_riscv_64,
    iree_uk_mmt4d_tile_f32f32f32_2x32x1_riscv_64, 2)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f32f32f32_1x32x1_to_7x32x1_riscv_64,
    iree_uk_mmt4d_tile_f32f32f32_4x32x1_riscv_64, 4)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f32f32f32_1x32x1_to_7x32x1_riscv_64,
    iree_uk_mmt4d_tile_f32f32f32_7x32x1_riscv_64, 7)
