// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_UNPACK_INTERNAL_H_
#define IREE_BUILTINS_UKERNEL_UNPACK_INTERNAL_H_

#include "iree/builtins/ukernel/unpack.h"

typedef enum iree_uk_unpack_type_t {
  iree_uk_unpack_type_f32f32 = IREE_UK_TIE_2_TYPES_LITERAL(FLOAT_32, FLOAT_32),
  iree_uk_unpack_type_i32i32 = IREE_UK_TIE_2_TYPES_LITERAL(INT_32, INT_32),
} iree_uk_unpack_type_t;

static inline iree_uk_unpack_type_t iree_uk_unpack_type(
    iree_uk_uint32_t flags) {
  switch (flags & IREE_UK_FLAG_UNPACK_TYPE_MASK) {
    case IREE_UK_FLAG_UNPACK_TYPE_F32F32:
      return iree_uk_unpack_type_f32f32;
    case IREE_UK_FLAG_UNPACK_TYPE_I32I32:
      return iree_uk_unpack_type_i32i32;
    default:
      IREE_UK_ASSUME_UNREACHABLE;
  }
}

static inline iree_uk_type_t iree_uk_unpack_in_type(
    iree_uk_unpack_type_t type) {
  return iree_uk_untie_type(0, type);
}

static inline iree_uk_type_t iree_uk_unpack_out_type(
    iree_uk_unpack_type_t type) {
  return iree_uk_untie_type(1, type);
}

typedef void (*iree_uk_unpack_tile_func_t)(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride0, iree_uk_index_t in_stride1,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1);

// Tile kernel declarations. Prototype matches iree_uk_unpack_tile_func_t.
#define IREE_UK_UNPACK_TILE_FUNC_DECL(NAME)                           \
  void NAME(void* IREE_UK_RESTRICT out_tile_ptr,                      \
            const void* IREE_UK_RESTRICT in_tile_ptr,                 \
            iree_uk_index_t outer_size1, iree_uk_index_t out_stride0, \
            iree_uk_index_t in_stride1, iree_uk_index_t elem_size,    \
            iree_uk_index_t tile_size0, iree_uk_index_t tile_size1);

// Returns the tile function to use for the unpack op with the given params.
iree_uk_unpack_tile_func_t iree_uk_unpack_select_tile_func(
    const iree_uk_unpack_params_t* params);

// Architecture-specific implementation.
iree_uk_unpack_tile_func_t iree_uk_unpack_select_tile_func_arch(
    const iree_uk_unpack_params_t* params);

#endif  // IREE_BUILTINS_UKERNEL_UNPACK_INTERNAL_H_
