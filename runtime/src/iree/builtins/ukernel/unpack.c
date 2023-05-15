// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/unpack_internal.h"

enum { iree_uk_unpack_tmp_buf_size = 4096 };

// Holds some information and a temporary buffer for performing padding.
typedef struct iree_uk_unpack_tmpbuf_helper_t {
  // Temporary buffer to pad the source data into, to pass to the tile_func.
  // Cache line alignment helps in pack_benchmark on ARM Cortex-A510/A710.
  IREE_UK_ATTRIBUTE_ALIGNED(64) char tmp_buf[iree_uk_unpack_tmp_buf_size];
  // How many tiles fit in `tmp_buf`.
  int max_tiles_in_tmp_buf;
} iree_uk_unpack_tmpbuf_helper_t;

// Return x/y for x>=0 and y>0, with a fast path for when y is a power of two.
static iree_uk_ssize_t iree_uk_div_nonneg_by_pos_and_likely_po2_i32(
    iree_uk_ssize_t x, iree_uk_int32_t y) {
  IREE_UK_ASSERT(x >= 0);
  IREE_UK_ASSERT(y > 0);
  return IREE_UK_LIKELY(iree_uk_is_po2_u32(y)) ? (x >> iree_uk_po2_log2_u32(y))
                                               : (x / y);
}

// Initializes a `iree_uk_unpack_padding_helper`. Asserts if the temporary
// buffer is smaller than one tile.
static void iree_uk_unpack_tmpbuf_helper_init(
    iree_uk_ssize_t tile_size0, iree_uk_ssize_t tile_size1,
    iree_uk_ssize_t elem_size, iree_uk_unpack_tmpbuf_helper_t* helper) {
  helper->max_tiles_in_tmp_buf = iree_uk_div_nonneg_by_pos_and_likely_po2_i32(
      iree_uk_unpack_tmp_buf_size, tile_size0 * tile_size1 * elem_size);
  IREE_UK_ASSERT(helper->max_tiles_in_tmp_buf > 0);
}

static void iree_uk_unpack_validate(const iree_uk_unpack_params_t* params) {
#ifdef IREE_UK_ENABLE_ASSERTS
  const iree_uk_uint32_t allflags = IREE_UK_FLAG_UNPACK_TRANSPOSE_INNER |
                                    IREE_UK_FLAG_UNPACK_TRANSPOSE_OUTER |
                                    IREE_UK_FLAG_UNPACK_TYPE_MASK;
  IREE_UK_ASSERT(!(params->flags & ~allflags));
  iree_uk_unpack_type_t unpack_type = iree_uk_unpack_type(params->flags);
  IREE_UK_ASSERT(unpack_type != iree_uk_unpack_type_none);
  IREE_UK_ASSERT(params->in_stride0 >= 0);
  IREE_UK_ASSERT(params->out_stride0 >= 0);
  IREE_UK_ASSERT(params->out_size0 >= 0);
  IREE_UK_ASSERT(params->out_size1 >= 0);
  IREE_UK_ASSERT(params->in_size0 >= 0);
  IREE_UK_ASSERT(params->in_size1 >= 0);
  IREE_UK_ASSERT(params->in_size2 >= 0);
  IREE_UK_ASSERT(params->in_size3 >= 0);
  // Check that the input and output shapes match, give or take padding that
  // must not exceed the inner tile size.s
  iree_uk_ssize_t outer_size0 = params->in_size0;
  iree_uk_ssize_t outer_size1 = params->in_size1;
  iree_uk_ssize_t tile_size0 = params->in_size2;
  iree_uk_ssize_t tile_size1 = params->in_size3;
  if (params->flags & IREE_UK_FLAG_UNPACK_TRANSPOSE_OUTER) {
    iree_uk_ssize_swap(&outer_size0, &outer_size1);
  }
  if (params->flags & IREE_UK_FLAG_UNPACK_TRANSPOSE_INNER) {
    iree_uk_ssize_swap(&tile_size0, &tile_size1);
  }
  IREE_UK_ASSERT(outer_size0 * tile_size0 >= params->out_size0);
  IREE_UK_ASSERT(outer_size1 * tile_size1 >= params->out_size1);
  // TODO(#11632): reenable these conditions.
  // IREE_UK_ASSERT((outer_size0 - 1) * tile_size0 < params->out_size0);
  // IREE_UK_ASSERT((outer_size1 - 1) * tile_size1 < params->out_size1);

  // Initialize a padding helper, just to get the assertion that the tile size
  // does not exceed the internal temporary buffer size, without having to
  // duplicate this arithmetic. Generally, we want to hit all failure modes
  // in the validation function so that the subsequent ukernel code can be
  // treated as infallible.
  iree_uk_unpack_tmpbuf_helper_t helper;
  iree_uk_type_t elem_type = iree_uk_unpack_in_type(unpack_type);
  iree_uk_ssize_t elem_size = iree_uk_type_size(elem_type);
  iree_uk_unpack_tmpbuf_helper_init(tile_size0, tile_size1, elem_size, &helper);
#endif  // IREE_UK_ENABLE_ASSERTS
}

// Early-return implementation for this ukernel. Returns true if already done.
static bool iree_uk_unpack_early(const iree_uk_unpack_params_t* params) {
  return (params->out_size0 == 0 || params->out_size1 == 0);
}

static void iree_uk_copy_slice(iree_uk_ssize_t src_stride0, const char* src_buf,
                               iree_uk_ssize_t dst_size0,
                               iree_uk_ssize_t dst_size1,
                               iree_uk_ssize_t dst_stride0, char* dst_buf,
                               iree_uk_ssize_t elem_size) {
  for (iree_uk_ssize_t in_i0 = 0; in_i0 < dst_size0; in_i0++) {
    iree_uk_memcpy(dst_buf, src_buf, dst_size1 * elem_size);
    dst_buf += dst_stride0 * elem_size;
    src_buf += src_stride0 * elem_size;
  }
}

// Unpacks an entire row, going through the temporary buffer to handle
// incomplete tiles. In cases involving only complete tiles, it is faster to
// call tile_func directly.
static void iree_uk_unpack_row_using_tmpbuf(
    iree_uk_unpack_tile_func_t tile_func, iree_uk_ssize_t dim1_tile_start,
    iree_uk_ssize_t dim1_tile_end, iree_uk_ssize_t dim0_write_size,
    iree_uk_ssize_t tile_size0, iree_uk_ssize_t tile_size1,
    iree_uk_ssize_t elem_size, iree_uk_ssize_t out_size1,
    iree_uk_ssize_t out_stride0, iree_uk_ssize_t in_stride1,
    iree_uk_unpack_tmpbuf_helper_t* helper, const char* in_buf, char* out_buf) {
  iree_uk_ssize_t dim1_tile = dim1_tile_start;
  while (dim1_tile < dim1_tile_end) {
    iree_uk_ssize_t dim1_chunk_tiles = iree_uk_ssize_clamp(
        dim1_tile_end - dim1_tile, 0, helper->max_tiles_in_tmp_buf);
    iree_uk_ssize_t dim1_chunk_src_width = dim1_chunk_tiles * tile_size1;
    iree_uk_ssize_t dim1_chunk_src_pos = dim1_tile * tile_size1;
    iree_uk_ssize_t dim1_write_size = iree_uk_ssize_clamp(
        out_size1 - dim1_chunk_src_pos, 0, dim1_chunk_src_width);
    tile_func(helper->tmp_buf, in_buf + dim1_tile * in_stride1 * elem_size,
              dim1_chunk_tiles, dim1_chunk_src_width, in_stride1, elem_size,
              tile_size0, tile_size1);
    iree_uk_copy_slice(dim1_chunk_src_width, helper->tmp_buf, dim0_write_size,
                       dim1_write_size, out_stride0,
                       out_buf + dim1_tile * tile_size1 * elem_size, elem_size);
    dim1_tile += dim1_chunk_tiles;
  }
}

static void iree_uk_unpack_using_tile_func(
    const iree_uk_unpack_params_t* params,
    iree_uk_unpack_tile_func_t tile_func) {
  // For now, the input and output element types are always the same.
  iree_uk_unpack_type_t unpack_type = iree_uk_unpack_type(params->flags);
  iree_uk_type_t elem_type = iree_uk_unpack_in_type(unpack_type);
  iree_uk_ssize_t elem_size = iree_uk_type_size(elem_type);
  iree_uk_ssize_t outer_size0 = params->in_size0;
  iree_uk_ssize_t outer_size1 = params->in_size1;
  iree_uk_ssize_t tile_size0 = params->in_size2;
  iree_uk_ssize_t tile_size1 = params->in_size3;
  iree_uk_ssize_t in_stride0 = params->in_stride0;
  iree_uk_ssize_t in_stride1 = params->in_size3 * params->in_size2;
  if (params->flags & IREE_UK_FLAG_UNPACK_TRANSPOSE_OUTER) {
    iree_uk_ssize_swap(&outer_size0, &outer_size1);
    iree_uk_ssize_swap(&in_stride0, &in_stride1);
  }
  if (params->flags & IREE_UK_FLAG_UNPACK_TRANSPOSE_INNER) {
    iree_uk_ssize_swap(&tile_size0, &tile_size1);
  }
  const char* in_buf =
      (const char*)params->in_buffer + (params->in_offset * elem_size);
  char* out_buf = (char*)params->out_buffer + (params->out_offset * elem_size);
  // Prepare for handling incomplete tiles with a temporary buffer.
  iree_uk_unpack_tmpbuf_helper_t tmpbuf_helper;
  if (params->out_size0 < outer_size0 * tile_size0 ||
      params->out_size1 < outer_size1 * tile_size1) {
    iree_uk_unpack_tmpbuf_helper_init(tile_size0, tile_size1, elem_size,
                                      &tmpbuf_helper);
  }
  // Compute number of tiles along both dimensions that fit entirely within the
  // source buffer's boundaries.
  int dim1_full_tiles = iree_uk_div_nonneg_by_pos_and_likely_po2_i32(
      params->out_size1, tile_size1);
  iree_uk_ssize_t i0 = 0;
  for (; i0 <= params->out_size0 - tile_size0; i0 += tile_size0) {
    // Pack whole tiles that do not require padding (entirely within the source
    // buffer's boundaries).
    tile_func(out_buf, in_buf, dim1_full_tiles, params->out_stride0, in_stride1,
              elem_size, tile_size0, tile_size1);
    // Right-padding.
    iree_uk_unpack_row_using_tmpbuf(
        tile_func, dim1_full_tiles, outer_size1, tile_size0, tile_size0,
        tile_size1, elem_size, params->out_size1, params->out_stride0,
        in_stride1, &tmpbuf_helper, in_buf, out_buf);
    out_buf += tile_size0 * params->out_stride0 * elem_size;
    in_buf += in_stride0 * elem_size;
  }
  // Bottom-padding.
  for (; i0 < outer_size0 * tile_size0; i0 += tile_size0) {
    iree_uk_ssize_t dim0_write_size =
        iree_uk_ssize_clamp(params->out_size0 - i0, 0, tile_size0);
    iree_uk_unpack_row_using_tmpbuf(
        tile_func, 0, outer_size1, dim0_write_size, tile_size0, tile_size1,
        elem_size, params->out_size1, params->out_stride0, in_stride1,
        &tmpbuf_helper, in_buf, out_buf);
    out_buf += tile_size0 * params->out_stride0 * elem_size;
    in_buf += in_stride0 * elem_size;
  }
}

IREE_UK_EXPORT int iree_uk_unpack(const iree_uk_unpack_params_t* params) {
  iree_uk_unpack_validate(params);

  if (iree_uk_unpack_early(params)) return 0;

  // Select a target-specific tile_func and use that with generic outer loops.
  iree_uk_unpack_tile_func_t func = iree_uk_unpack_select_tile_func(params);
  iree_uk_unpack_using_tile_func(params, func);
  return 0;
}
