// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/arm_64/common_arm_64.h"
#include "iree/builtins/ukernel/unpack_internal.h"

static void iree_uk_unpack_tile_8x8_x32_arm_64_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride0, iree_uk_index_t in_stride1,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 4);
  IREE_UK_ASSERT(tile_size0 == 8);
  IREE_UK_ASSERT(tile_size1 == 8);
  iree_uk_int8_t* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  for (; outer_size1 > 0; --outer_size1) {
    iree_uk_neon_copy_8x32xi8_strided_to_strided(out_ptr, in_ptr,
                                                 4 * out_stride0, 32);
    out_ptr += 32;
    in_ptr += 4 * in_stride1;
  }
}

iree_uk_unpack_tile_func_t iree_uk_unpack_select_tile_func_arch(
    const iree_uk_unpack_params_t* params) {
  iree_uk_unpack_type_t unpack_type = iree_uk_unpack_type(params->flags);
  int esize = iree_uk_type_size(iree_uk_unpack_out_type(unpack_type));
  bool transpose = params->flags & IREE_UK_FLAG_UNPACK_TRANSPOSE_INNER;
  // Unpack is currently only used in practice with esize==4 and non-transpose.
  if (esize != 4 || transpose) return 0;
  if (params->in_size2 == 8 && params->in_size3 == 8) {
    return iree_uk_unpack_tile_8x8_x32_arm_64_direct;
  }
  return 0;
}
