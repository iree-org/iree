// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/pack_tile.h"

#if defined(IREE_UK_ARCH_ARM_64)
#include "iree/builtins/ukernel/arch/arm_64/pack_arm_64.h"
#elif defined(IREE_UK_ARCH_X86_64)
#include "iree/builtins/ukernel/arch/x86_64/pack_x86_64.h"
#endif

static void iree_uk_pack_tile_generic_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_ssize_t outer_size1,
    iree_uk_ssize_t out_stride1, iree_uk_ssize_t in_stride0,
    iree_uk_ssize_t elem_size, iree_uk_ssize_t tile_size0,
    iree_uk_ssize_t tile_size1) {
  const char* IREE_UK_RESTRICT in_ptr_l1 = in_tile_ptr;
  char* IREE_UK_RESTRICT out_ptr_l1 = out_tile_ptr;
  for (iree_uk_ssize_t outer_i1 = 0; outer_i1 < outer_size1; ++outer_i1) {
    const char* IREE_UK_RESTRICT in_ptr = in_ptr_l1;
    char* IREE_UK_RESTRICT out_ptr = out_ptr_l1;
    for (iree_uk_ssize_t tile_i0 = 0; tile_i0 < tile_size0; ++tile_i0) {
      iree_uk_memcpy(out_ptr, in_ptr, tile_size1 * elem_size);
      out_ptr += tile_size1 * elem_size;
      in_ptr += in_stride0 * elem_size;
    }
    out_ptr_l1 += out_stride1 * elem_size;
    in_ptr_l1 += tile_size1 * elem_size;
  }
}

static void iree_uk_pack_tile_generic_transpose(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_ssize_t outer_size1,
    iree_uk_ssize_t out_stride1, iree_uk_ssize_t in_stride0,
    iree_uk_ssize_t elem_size, iree_uk_ssize_t tile_size0,
    iree_uk_ssize_t tile_size1) {
  const char* IREE_UK_RESTRICT in_ptr_l1 = in_tile_ptr;
  char* IREE_UK_RESTRICT out_ptr_l1 = out_tile_ptr;
  for (iree_uk_ssize_t outer_i1 = 0; outer_i1 < outer_size1; ++outer_i1) {
    const char* IREE_UK_RESTRICT in_ptr_l2 = in_ptr_l1;
    char* IREE_UK_RESTRICT out_ptr_l2 = out_ptr_l1;
    for (iree_uk_ssize_t tile_i0 = 0; tile_i0 < tile_size0; ++tile_i0) {
      const char* IREE_UK_RESTRICT in_ptr = in_ptr_l2;
      char* IREE_UK_RESTRICT out_ptr = out_ptr_l2;
      for (iree_uk_ssize_t tile_i1 = 0; tile_i1 < tile_size1; ++tile_i1) {
        iree_uk_memcpy(out_ptr, in_ptr, elem_size);
        out_ptr += tile_size0 * elem_size;
        in_ptr += elem_size;
      }
      out_ptr_l2 += elem_size;
      in_ptr_l2 += in_stride0 * elem_size;
    }
    out_ptr_l1 += out_stride1 * elem_size;
    in_ptr_l1 += tile_size1 * elem_size;
  }
}

static iree_uk_pack_tile_func_t iree_uk_pack_select_tile_func_generic(
    const iree_uk_pack_params_t* params) {
  return (params->flags & IREE_UK_FLAG_PACK_TRANSPOSE_INNER)
             ? iree_uk_pack_tile_generic_transpose
             : iree_uk_pack_tile_generic_direct;
}

static iree_uk_pack_tile_func_t iree_uk_pack_select_tile_func_arch(
    const iree_uk_pack_params_t* params) {
#if defined(IREE_UK_ARCH_ARM_64)
  return iree_uk_pack_select_tile_func_arm_64(params);
#elif defined(IREE_UK_ARCH_X86_64)
  return iree_uk_pack_select_tile_func_x86_64(params);
#endif
  return 0;
}

iree_uk_pack_tile_func_t iree_uk_pack_select_tile_func(
    const iree_uk_pack_params_t* params) {
  iree_uk_pack_tile_func_t arch_tile_func =
      iree_uk_pack_select_tile_func_arch(params);
  if (arch_tile_func) return arch_tile_func;
  return iree_uk_pack_select_tile_func_generic(params);
}
