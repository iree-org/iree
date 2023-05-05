// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_PACK_H_
#define IREE_BUILTINS_UKERNEL_PACK_H_

#include "iree/builtins/ukernel/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum iree_uk_pack_type_t {
  iree_uk_pack_type_f32f32 = IREE_UK_TIE_2_TYPES_LITERAL(FLOAT_32, FLOAT_32),
  iree_uk_pack_type_i8i8 = IREE_UK_TIE_2_TYPES_LITERAL(INT_8, INT_8),
  iree_uk_pack_type_i32i32 = IREE_UK_TIE_2_TYPES_LITERAL(INT_32, INT_32),
} iree_uk_pack_type_t;

static inline iree_uk_type_t iree_uk_pack_in_type(iree_uk_pack_type_t type) {
  return iree_uk_untie_type(0, type);
}

static inline iree_uk_type_t iree_uk_pack_out_type(iree_uk_pack_type_t type) {
  return iree_uk_untie_type(1, type);
}

// Parameters for a pack operation.
typedef struct iree_uk_pack_params_t {
  iree_uk_pack_type_t type;
  iree_uk_uint32_t flags;
  iree_uk_ssize_t in_stride0;
  iree_uk_ssize_t out_stride0;
  iree_uk_ssize_t in_size0;
  iree_uk_ssize_t in_size1;
  iree_uk_ssize_t out_size0;
  iree_uk_ssize_t out_size1;
  iree_uk_ssize_t out_size2;
  iree_uk_ssize_t out_size3;
  const void* in_buffer;
  void* out_buffer;
  const void* padding_value;
  const iree_uk_uint64_t* cpu_data;
} iree_uk_pack_params_t;

typedef void (*iree_uk_pack_tile_func_t)(
    void* IREE_UK_RESTRICT /*out_tile_ptr*/,
    const void* IREE_UK_RESTRICT /*in_tile_ptr*/,
    iree_uk_ssize_t /*outer_size1*/, iree_uk_ssize_t /*out_stride1*/,
    iree_uk_ssize_t /*in_stride0*/, iree_uk_ssize_t /*elem_size*/,
    iree_uk_ssize_t /*tile_size0*/, iree_uk_ssize_t /*tile_size1*/);

// Tile kernel declarations. Prototype matches iree_uk_unpack_tile_func_t.
#define IREE_UK_PACK_TILE_FUNC_DECL(NAME)                             \
  void NAME(void* IREE_UK_RESTRICT out_tile_ptr,                      \
            const void* IREE_UK_RESTRICT in_tile_ptr,                 \
            iree_uk_ssize_t outer_size1, iree_uk_ssize_t out_stride1, \
            iree_uk_ssize_t in_stride0, iree_uk_ssize_t elem_size,    \
            iree_uk_ssize_t tile_size0, iree_uk_ssize_t tile_size1);

// Main entry point.
IREE_UK_EXPORT void iree_uk_pack(const iree_uk_pack_params_t* params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BUILTINS_UKERNEL_PACK_H_
