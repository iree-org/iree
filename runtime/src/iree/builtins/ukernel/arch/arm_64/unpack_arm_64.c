// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/arm_64/unpack_arm_64.h"

#include <arm_neon.h>

#include "iree/builtins/ukernel/arch/arm_64/common_arm_neon.h"

static void iree_uk_unpack_tile_8x1_x8_arm_64_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_ssize_t outer_size1,
    iree_uk_ssize_t out_stride0, iree_uk_ssize_t in_stride1,
    iree_uk_ssize_t elem_size, iree_uk_ssize_t tile_size0,
    iree_uk_ssize_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 1);
  IREE_UK_ASSERT(tile_size0 == 8);
  IREE_UK_ASSERT(tile_size1 == 1);
  const char* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  char* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  for (; outer_size1 >= 8; outer_size1 -= 8) {
    iree_uk_neon_copy_8x8xi8_transpose_strided_to_strided(
        out_ptr, in_ptr, out_stride0, in_stride1);
    out_ptr += 8;
    in_ptr += 8 * in_stride1;
  }
  for (; outer_size1 > 0; outer_size1--) {
    iree_uk_neon_copy_8x1xi8_unstrided_to_strided(out_ptr, out_stride0, in_ptr);
    in_ptr += in_stride1;
    out_ptr += 1;
  }
}

static void iree_uk_unpack_tile_8x4_x8_arm_64_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_ssize_t outer_size1,
    iree_uk_ssize_t out_stride0, iree_uk_ssize_t in_stride1,
    iree_uk_ssize_t elem_size, iree_uk_ssize_t tile_size0,
    iree_uk_ssize_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 1);
  IREE_UK_ASSERT(tile_size0 == 8);
  IREE_UK_ASSERT(tile_size1 == 4);
  const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  iree_uk_int8_t* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  for (; outer_size1 >= 2; outer_size1 -= 2) {
    iree_uk_neon_copy_8x8xi8_tiled_8x4_transpose_strided_to_strided(
        out_ptr, in_ptr, out_stride0, in_stride1);
    in_ptr += 2 * in_stride1;
    out_ptr += 8;
  }
  for (; outer_size1 > 0; --outer_size1) {
    iree_uk_neon_copy_8x4_unstrided_to_strided(out_ptr, out_stride0, in_ptr);
    in_ptr += in_stride1;
    out_ptr += 4;
  }
}

static void iree_uk_unpack_tile_8x1_x32_arm_64_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_ssize_t outer_size1,
    iree_uk_ssize_t out_stride0, iree_uk_ssize_t in_stride1,
    iree_uk_ssize_t elem_size, iree_uk_ssize_t tile_size0,
    iree_uk_ssize_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 4);
  IREE_UK_ASSERT(tile_size0 == 8);
  IREE_UK_ASSERT(tile_size1 == 1);
  iree_uk_unpack_tile_8x4_x8_arm_64_direct(out_tile_ptr, in_tile_ptr,
                                           outer_size1, 4 * out_stride0,
                                           4 * in_stride1, 1, 8, 4);
}

static void iree_uk_unpack_tile_8x8_x8_arm_64_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_ssize_t outer_size1,
    iree_uk_ssize_t out_stride0, iree_uk_ssize_t in_stride1,
    iree_uk_ssize_t elem_size, iree_uk_ssize_t tile_size0,
    iree_uk_ssize_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 1);
  IREE_UK_ASSERT(tile_size0 == 8);
  IREE_UK_ASSERT(tile_size1 == 8);
  iree_uk_int8_t* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  for (; outer_size1 > 0; --outer_size1) {
    iree_uk_neon_copy_8x8xi8_unstrided_to_strided(out_ptr, in_ptr, out_stride0);
    out_ptr += 8;
    in_ptr += in_stride1;
  }
}

static void iree_uk_unpack_tile_8x1_x32_arm_64_transpose(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_ssize_t outer_size1,
    iree_uk_ssize_t out_stride0, iree_uk_ssize_t in_stride1,
    iree_uk_ssize_t elem_size, iree_uk_ssize_t tile_size0,
    iree_uk_ssize_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 4);
  IREE_UK_ASSERT(tile_size0 == 1);
  IREE_UK_ASSERT(tile_size1 == 8);
  const iree_uk_int32_t* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  for (; outer_size1 > 0; --outer_size1) {
    iree_uk_memcpy(out_ptr, in_ptr, 32);
    in_ptr += in_stride1;
    out_ptr += 8;
  }
}

static void iree_uk_unpack_tile_8x1_x8_arm_64_transpose(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_ssize_t outer_size1,
    iree_uk_ssize_t out_stride0, iree_uk_ssize_t in_stride1,
    iree_uk_ssize_t elem_size, iree_uk_ssize_t tile_size0,
    iree_uk_ssize_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 1);
  IREE_UK_ASSERT(tile_size0 == 1);
  IREE_UK_ASSERT(tile_size1 == 8);
  const char* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  char* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  for (; outer_size1 >= 4; outer_size1 -= 4) {
    iree_uk_memcpy(out_ptr, in_ptr, 8);
    iree_uk_memcpy(out_ptr + 8, in_ptr + in_stride1, 8);
    iree_uk_memcpy(out_ptr + 16, in_ptr + 2 * in_stride1, 8);
    iree_uk_memcpy(out_ptr + 24, in_ptr + 3 * in_stride1, 8);
    in_ptr += 4 * in_stride1;
    out_ptr += 32;
  }
  for (; outer_size1 > 0; --outer_size1) {
    iree_uk_memcpy(out_ptr, in_ptr, 8);
    in_ptr += in_stride1;
    out_ptr += 8;
  }
}

static void iree_uk_unpack_tile_8x4_x8_arm_64_transpose(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_ssize_t outer_size1,
    iree_uk_ssize_t out_stride0, iree_uk_ssize_t in_stride1,
    iree_uk_ssize_t elem_size, iree_uk_ssize_t tile_size0,
    iree_uk_ssize_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 1);
  IREE_UK_ASSERT(tile_size0 == 4);
  IREE_UK_ASSERT(tile_size1 == 8);
  const char* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  char* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  for (; outer_size1 > 0; --outer_size1) {
    int8x8x4_t in = vld4_s8(in_ptr);
    vst1_s8(out_ptr + 0 * out_stride0, in.val[0]);
    vst1_s8(out_ptr + 1 * out_stride0, in.val[1]);
    vst1_s8(out_ptr + 2 * out_stride0, in.val[2]);
    vst1_s8(out_ptr + 3 * out_stride0, in.val[3]);
    in_ptr += in_stride1;
    out_ptr += 8;
  }
}

static void iree_uk_unpack_tile_8x8_x8_arm_64_transpose(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_ssize_t outer_size1,
    iree_uk_ssize_t out_stride0, iree_uk_ssize_t in_stride1,
    iree_uk_ssize_t elem_size, iree_uk_ssize_t tile_size0,
    iree_uk_ssize_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 1);
  IREE_UK_ASSERT(tile_size0 == 8);
  IREE_UK_ASSERT(tile_size1 == 8);
  const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  iree_uk_int8_t* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  for (; outer_size1 > 0; --outer_size1) {
    iree_uk_neon_copy_8x8xi8_transpose_unstrided_to_strided(
        out_ptr, out_stride0, in_ptr);
    out_ptr += 8;
    in_ptr += in_stride1;
  }
}

iree_uk_unpack_tile_func_t iree_uk_unpack_select_tile_func_arm_64(
    const iree_uk_unpack_params_t* params) {
  // At the moment, as sum-reductions are not yet part of pack ops,
  // no arithmetic whatsoever is being done here, so only the element type
  // size matters, not the type itself.
  int esize = iree_uk_type_size(iree_uk_unpack_out_type(params->type));
  bool transpose = params->flags & IREE_UK_FLAG_UNPACK_TRANSPOSE_INNER;
  if (esize == 4 && params->in_size2 == 8 && params->in_size3 == 1) {
    return transpose ? iree_uk_unpack_tile_8x1_x32_arm_64_transpose
                     : iree_uk_unpack_tile_8x1_x32_arm_64_direct;
  } else if (esize == 1 && params->in_size2 == 8 && params->in_size3 == 1) {
    return transpose ? iree_uk_unpack_tile_8x1_x8_arm_64_transpose
                     : iree_uk_unpack_tile_8x1_x8_arm_64_direct;
  } else if (esize == 1 && params->in_size2 == 8 && params->in_size3 == 4) {
    return transpose ? iree_uk_unpack_tile_8x4_x8_arm_64_transpose
                     : iree_uk_unpack_tile_8x4_x8_arm_64_direct;
  } else if (esize == 1 && params->in_size2 == 8 && params->in_size3 == 8) {
    return transpose ? iree_uk_unpack_tile_8x8_x8_arm_64_transpose
                     : iree_uk_unpack_tile_8x8_x8_arm_64_direct;
  }
  return 0;
}
