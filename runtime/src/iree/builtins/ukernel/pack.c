// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/pack.h"

#include "iree/builtins/ukernel/arch/pack_arch.h"
#include "iree/builtins/ukernel/pack_generic.h"

static void iree_uk_pack_validate(const iree_uk_pack_params_t* params) {
#ifdef IREE_UK_ENABLE_ASSERTS
  const iree_uk_uint32_t allflags =
      IREE_UK_FLAG_PACK_TRANSPOSE_INNER | IREE_UK_FLAG_PACK_TRANSPOSE_OUTER;
  IREE_UK_ASSERT(!(params->flags & ~allflags));
  IREE_UK_ASSERT(params->type == iree_uk_pack_type_f32f32 ||
                 params->type == iree_uk_pack_type_i8i8 ||
                 params->type == iree_uk_pack_type_i32i32);
  IREE_UK_ASSERT(params->in_stride0 >= 0);
  IREE_UK_ASSERT(params->out_stride0 >= 0);
  IREE_UK_ASSERT(params->in_size0 >= 0);
  IREE_UK_ASSERT(params->in_size1 >= 0);
  IREE_UK_ASSERT(params->out_size0 >= 0);
  IREE_UK_ASSERT(params->out_size1 >= 0);
  IREE_UK_ASSERT(params->out_size2 >= 0);
  IREE_UK_ASSERT(params->out_size3 >= 0);
  // Check that the input and output shapes match, give or take padding that
  // must not exceed the inner tile size.s
  iree_uk_ssize_t outer_size0 = params->out_size0;
  iree_uk_ssize_t outer_size1 = params->out_size1;
  iree_uk_ssize_t tile_size0 = params->out_size2;
  iree_uk_ssize_t tile_size1 = params->out_size3;
  if (params->flags & IREE_UK_FLAG_PACK_TRANSPOSE_OUTER) {
    iree_uk_ssize_swap(&outer_size0, &outer_size1);
  }
  if (params->flags & IREE_UK_FLAG_PACK_TRANSPOSE_INNER) {
    iree_uk_ssize_swap(&tile_size0, &tile_size1);
  }
  IREE_UK_ASSERT(outer_size0 * tile_size0 >= params->in_size0);
  IREE_UK_ASSERT(outer_size1 * tile_size1 >= params->in_size1);
  // TODO(#11632): reenable these conditions.
  // IREE_UK_ASSERT((outer_size0 - 1) * tile_size0 < params->in_size0);
  // IREE_UK_ASSERT((outer_size1 - 1) * tile_size1 < params->in_size1);
#endif  // IREE_UK_ENABLE_ASSERTS
}

static bool iree_uk_pack_early(const iree_uk_pack_params_t* params) {
  return (params->out_size0 == 0 || params->out_size1 == 0 ||
          params->out_size2 == 0 || params->out_size3 == 0);
}

static iree_uk_pack_tile_func_t iree_uk_pack_select_tile_func(
    const iree_uk_pack_params_t* params) {
  iree_uk_pack_tile_func_t arch_tile_func =
      iree_uk_pack_select_tile_func_arch(params);
  if (arch_tile_func) {
    return arch_tile_func;
  }
  return iree_uk_pack_select_tile_func_generic(params);
}

static void iree_uk_pack_using_tile_func(const iree_uk_pack_params_t* params,
                                         iree_uk_pack_tile_func_t tile_func) {
  // For now, the input and output element types are always the same.
  iree_uk_type_t elem_type = iree_uk_pack_in_type(params->type);
  iree_uk_ssize_t elem_size = iree_uk_type_size(elem_type);
  iree_uk_ssize_t outer_size0 = params->out_size0;
  iree_uk_ssize_t outer_size1 = params->out_size1;
  iree_uk_ssize_t tile_size0 = params->out_size2;
  iree_uk_ssize_t tile_size1 = params->out_size3;
  iree_uk_ssize_t out_stride_l0 = params->out_stride0;
  iree_uk_ssize_t out_stride_l1 = params->out_size3 * params->out_size2;
  iree_uk_ssize_t out_stride_l2 = params->out_size3;
  iree_uk_ssize_t out_stride_l3 = 1;
  if (params->flags & IREE_UK_FLAG_PACK_TRANSPOSE_OUTER) {
    iree_uk_ssize_swap(&outer_size0, &outer_size1);
    iree_uk_ssize_swap(&out_stride_l0, &out_stride_l1);
  }
  if (params->flags & IREE_UK_FLAG_PACK_TRANSPOSE_INNER) {
    iree_uk_ssize_swap(&tile_size0, &tile_size1);
    iree_uk_ssize_swap(&out_stride_l2, &out_stride_l3);
  }
  const char* in_row_ptr = params->in_buffer;
  char* out_row_ptr = params->out_buffer;
  bool l0_has_padding = outer_size0 * tile_size0 != params->in_size0;
  bool l1_has_padding = outer_size1 * tile_size1 != params->in_size1;
  iree_uk_ssize_t l0_full_tile_end = outer_size0 - (l0_has_padding ? 1 : 0);
  iree_uk_ssize_t l1_full_tile_end = outer_size1 - (l1_has_padding ? 1 : 0);

  // TODO(#11632): this fix-up is needed only because at the moment,
  // there may actually be padding in more than just the last tile along each
  // dimension, because dispatch region formation rounds small sizes up.
  while (l0_full_tile_end * tile_size0 > params->in_size0) --l0_full_tile_end;
  while (l1_full_tile_end * tile_size1 > params->in_size1) --l1_full_tile_end;

  for (iree_uk_ssize_t outer_i0 = 0; outer_i0 < outer_size0; ++outer_i0) {
    // If we're on the final iteration of outer loop 0 and there is padding,
    // set l1_full_tile_end to 0, so henceforth it is sufficient to check
    // against l1_full_tile_end to tell if we are padding.
    if (outer_i0 >= l0_full_tile_end) {
      l1_full_tile_end = 0;
    }
    // Handle full tiles, using the (fast) tile_func.
    char* out_tile_ptr =
        tile_func(out_row_ptr, in_row_ptr, l1_full_tile_end, out_stride_l1,
                  params->in_stride0, elem_size, tile_size0, tile_size1);
    // Handle incomplete tiles, with padding, using slow code here.
    for (iree_uk_ssize_t outer_i1 = l1_full_tile_end; outer_i1 < outer_size1;
         ++outer_i1) {
      for (iree_uk_ssize_t tile_i0 = 0; tile_i0 < tile_size0; ++tile_i0) {
        for (iree_uk_ssize_t tile_i1 = 0; tile_i1 < tile_size1; ++tile_i1) {
          iree_uk_ssize_t i0 = outer_i0 * tile_size0 + tile_i0;
          iree_uk_ssize_t i1 = outer_i1 * tile_size1 + tile_i1;
          char* out_ptr =
              out_tile_ptr +
              (tile_i0 * out_stride_l2 + tile_i1 * out_stride_l3) * elem_size;
          if (i0 >= params->in_size0 || i1 >= params->in_size1) {
            iree_uk_memcpy(out_ptr, params->padding_value, elem_size);
          } else {
            iree_uk_ssize_t in_offset = i1 + i0 * params->in_stride0;
            const char* in_ptr =
                ((char*)params->in_buffer) + in_offset * elem_size;
            iree_uk_memcpy(out_ptr, in_ptr, elem_size);
          }
        }
      }
      out_tile_ptr += out_stride_l1 * elem_size;
    }
    out_row_ptr += out_stride_l0 * elem_size;
    in_row_ptr += tile_size0 * params->in_stride0 * elem_size;
  }
}

IREE_UK_EXPORT void iree_uk_pack(const iree_uk_pack_params_t* params) {
  iree_uk_pack_validate(params);

  if (iree_uk_pack_early(params)) return;

  // Select a target-specific tile_func (inner loop on K, computing one M0xN0
  // tile) and use that with generic outer loops.
  iree_uk_pack_tile_func_t row_func = iree_uk_pack_select_tile_func(params);
  iree_uk_pack_using_tile_func(params, row_func);
}
