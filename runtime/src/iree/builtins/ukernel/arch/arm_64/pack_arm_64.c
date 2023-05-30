// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/arm_64/common_arm_64.h"
#include "iree/builtins/ukernel/pack_internal.h"

static void iree_uk_pack_tile_8x1_x8_arm_64_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_ssize_t outer_size1,
    iree_uk_ssize_t out_stride1, iree_uk_ssize_t in_stride0,
    iree_uk_ssize_t elem_size, iree_uk_ssize_t tile_size0,
    iree_uk_ssize_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 1);
  IREE_UK_ASSERT(tile_size0 == 8);
  IREE_UK_ASSERT(tile_size1 == 1);
  iree_uk_int8_t* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  // A further 2x unrolling (outer_size1-=16) yields another 1.2x speedup on
  // A710 thanks to using 16-byte loads. Is it worth the code size? This 8x1
  // tile is used on baseline aarch64 where the matmul kernel is slow anyway.
  for (; outer_size1 >= 8; outer_size1 -= 8) {
    iree_uk_neon_copy_8x8xi8_transpose_strided_to_strided(
        out_ptr, in_ptr, out_stride1, in_stride0);
    out_ptr += 8 * out_stride1;
    in_ptr += 8;
  }
  for (; outer_size1 > 0; --outer_size1) {
    iree_uk_neon_copy_8x1xi8_strided_to_unstrided(out_ptr, in_ptr, in_stride0);
    out_ptr += out_stride1;
    in_ptr += 1;
  }
}

static void iree_uk_pack_tile_8x4_x8_arm_64_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_ssize_t outer_size1,
    iree_uk_ssize_t out_stride1, iree_uk_ssize_t in_stride0,
    iree_uk_ssize_t elem_size, iree_uk_ssize_t tile_size0,
    iree_uk_ssize_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 1);
  IREE_UK_ASSERT(tile_size0 == 8);
  IREE_UK_ASSERT(tile_size1 == 4);
  iree_uk_int8_t* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  for (; outer_size1 >= 2; outer_size1 -= 2) {
    iree_uk_neon_copy_8x8xi8_tiled_1x4_transpose_strided_to_strided(
        out_ptr, in_ptr, out_stride1, in_stride0);
    out_ptr += 2 * out_stride1;
    in_ptr += 8;
  }
  for (; outer_size1 > 0; --outer_size1) {
    iree_uk_neon_copy_8x4xi8_strided_to_unstrided(out_ptr, in_ptr, in_stride0);
    out_ptr += out_stride1;
    in_ptr += 4;
  }
}

static void iree_uk_pack_tile_8x1_x32_arm_64_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_ssize_t outer_size1,
    iree_uk_ssize_t out_stride1, iree_uk_ssize_t in_stride0,
    iree_uk_ssize_t elem_size, iree_uk_ssize_t tile_size0,
    iree_uk_ssize_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 4);
  IREE_UK_ASSERT(tile_size0 == 8);
  IREE_UK_ASSERT(tile_size1 == 1);
  iree_uk_pack_tile_8x4_x8_arm_64_direct(out_tile_ptr, in_tile_ptr, outer_size1,
                                         out_stride1 * 4, in_stride0 * 4, 1, 8,
                                         4);
}

static void iree_uk_pack_tile_8x8_x8_arm_64_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_ssize_t outer_size1,
    iree_uk_ssize_t out_stride1, iree_uk_ssize_t in_stride0,
    iree_uk_ssize_t elem_size, iree_uk_ssize_t tile_size0,
    iree_uk_ssize_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 1);
  IREE_UK_ASSERT(tile_size0 == 8);
  IREE_UK_ASSERT(tile_size1 == 8);
  iree_uk_int8_t* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  for (; outer_size1 > 0; --outer_size1) {
    iree_uk_neon_copy_8x8xi8_strided_to_unstrided(out_ptr, in_ptr, in_stride0);
    out_ptr += out_stride1;
    in_ptr += 8;
  }
}

static void iree_uk_pack_tile_8x1_x32_arm_64_transpose(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_ssize_t outer_size1,
    iree_uk_ssize_t out_stride1, iree_uk_ssize_t in_stride0,
    iree_uk_ssize_t elem_size, iree_uk_ssize_t tile_size0,
    iree_uk_ssize_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 4);
  IREE_UK_ASSERT(tile_size0 == 1);
  IREE_UK_ASSERT(tile_size1 == 8);
  const iree_uk_int32_t* IREE_UK_RESTRICT in_tile_ptr_i32 = in_tile_ptr;
  iree_uk_int32_t* IREE_UK_RESTRICT out_tile_i32_ptr = out_tile_ptr;
  for (; outer_size1 > 0; --outer_size1) {
    iree_uk_memcpy(out_tile_i32_ptr, in_tile_ptr_i32, 32);
    out_tile_i32_ptr += out_stride1;
    in_tile_ptr_i32 += 8;
  }
}

static void iree_uk_pack_tile_8x1_x8_arm_64_transpose(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_ssize_t outer_size1,
    iree_uk_ssize_t out_stride1, iree_uk_ssize_t in_stride0,
    iree_uk_ssize_t elem_size, iree_uk_ssize_t tile_size0,
    iree_uk_ssize_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 1);
  IREE_UK_ASSERT(tile_size0 == 1);
  IREE_UK_ASSERT(tile_size1 == 8);
  const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  iree_uk_int8_t* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  for (; outer_size1 >= 4; outer_size1 -= 4) {
    iree_uk_memcpy(out_ptr + 0 * out_stride1, in_ptr + 0, 8);
    iree_uk_memcpy(out_ptr + 1 * out_stride1, in_ptr + 8, 8);
    iree_uk_memcpy(out_ptr + 2 * out_stride1, in_ptr + 16, 8);
    iree_uk_memcpy(out_ptr + 3 * out_stride1, in_ptr + 24, 8);
    out_ptr += 4 * out_stride1;
    in_ptr += 32;
  }
  for (; outer_size1 > 0; --outer_size1) {
    iree_uk_memcpy(out_ptr, in_ptr, 8);
    out_ptr += out_stride1;
    in_ptr += 8;
  }
}

static void iree_uk_pack_tile_8x4_x8_arm_64_transpose(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_ssize_t outer_size1,
    iree_uk_ssize_t out_stride1, iree_uk_ssize_t in_stride0,
    iree_uk_ssize_t elem_size, iree_uk_ssize_t tile_size0,
    iree_uk_ssize_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 1);
  IREE_UK_ASSERT(tile_size0 == 4);
  IREE_UK_ASSERT(tile_size1 == 8);
  const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  iree_uk_int8_t* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  for (; outer_size1 > 0; --outer_size1) {
    int8x16x2_t in;
    in.val[0] = vcombine_s8(vld1_s8(in_ptr + 0 * in_stride0),
                            vld1_s8(in_ptr + 2 * in_stride0));
    in.val[1] = vcombine_s8(vld1_s8(in_ptr + 1 * in_stride0),
                            vld1_s8(in_ptr + 3 * in_stride0));
    int16x8x2_t zip_i16 = iree_uk_neon_zip_16xi8_as_8xi16(in.val[0], in.val[1]);
    int32x4x2_t zip_i32 =
        iree_uk_neon_zip_8xi16_as_4xi32(zip_i16.val[0], zip_i16.val[1]);
    vst1q_s8(out_ptr, vreinterpretq_s8_s32(zip_i32.val[0]));
    vst1q_s8(out_ptr + 16, vreinterpretq_s8_s32(zip_i32.val[1]));
    out_ptr += out_stride1;
    in_ptr += 8;
  }
}

static void iree_uk_pack_tile_8x8_x8_arm_64_transpose(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_ssize_t outer_size1,
    iree_uk_ssize_t out_stride1, iree_uk_ssize_t in_stride0,
    iree_uk_ssize_t elem_size, iree_uk_ssize_t tile_size0,
    iree_uk_ssize_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 1);
  IREE_UK_ASSERT(tile_size0 == 8);
  IREE_UK_ASSERT(tile_size1 == 8);
  const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  iree_uk_int8_t* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  for (; outer_size1 > 0; --outer_size1) {
    iree_uk_neon_copy_8x8xi8_transpose_strided_to_unstrided(out_ptr, in_ptr,
                                                            in_stride0);

    out_ptr += out_stride1;
    in_ptr += 8;
  }
}

static void iree_uk_pack_tile_8x8_x32_arm_64_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_ssize_t outer_size1,
    iree_uk_ssize_t out_stride1, iree_uk_ssize_t in_stride0,
    iree_uk_ssize_t elem_size, iree_uk_ssize_t tile_size0,
    iree_uk_ssize_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 4);
  IREE_UK_ASSERT(tile_size0 == 8);
  IREE_UK_ASSERT(tile_size1 == 8);
  const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  iree_uk_int8_t* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  for (; outer_size1 > 0; --outer_size1) {
    iree_uk_neon_copy_8x32xi8_strided_to_strided(out_ptr, in_ptr, 32,
                                                 4 * in_stride0);
    out_ptr += 4 * out_stride1;
    in_ptr += 32;
  }
}

iree_uk_pack_tile_func_t iree_uk_pack_select_tile_func_arch(
    const iree_uk_pack_params_t* params) {
  // At the moment, as sum-reductions are not yet part of pack ops,
  // no arithmetic whatsoever is being done here, so only the element type
  // size matters, not the type itself.
  iree_uk_pack_type_t pack_type = iree_uk_pack_type(params->flags);
  int esize = iree_uk_type_size(iree_uk_pack_out_type(pack_type));
  bool transpose = params->flags & IREE_UK_FLAG_PACK_TRANSPOSE_INNER;
  if (esize == 4 && params->out_size2 == 8 && params->out_size3 == 8) {
    // Currently only used for accumulators, which are never transposed.
    return transpose ? 0 : iree_uk_pack_tile_8x8_x32_arm_64_direct;
  } else if (esize == 4 && params->out_size2 == 8 && params->out_size3 == 1) {
    return transpose ? iree_uk_pack_tile_8x1_x32_arm_64_transpose
                     : iree_uk_pack_tile_8x1_x32_arm_64_direct;
  } else if (esize == 1 && params->out_size2 == 8 && params->out_size3 == 1) {
    return transpose ? iree_uk_pack_tile_8x1_x8_arm_64_transpose
                     : iree_uk_pack_tile_8x1_x8_arm_64_direct;
  } else if (esize == 1 && params->out_size2 == 8 && params->out_size3 == 4) {
    return transpose ? iree_uk_pack_tile_8x4_x8_arm_64_transpose
                     : iree_uk_pack_tile_8x4_x8_arm_64_direct;
  } else if (esize == 1 && params->out_size2 == 8 && params->out_size3 == 8) {
    return transpose ? iree_uk_pack_tile_8x8_x8_arm_64_transpose
                     : iree_uk_pack_tile_8x8_x8_arm_64_direct;
  }
  return 0;
}
