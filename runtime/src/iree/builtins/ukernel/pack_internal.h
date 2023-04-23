// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_PACK_INTERNAL_H_
#define IREE_BUILTINS_UKERNEL_PACK_INTERNAL_H_

#include "iree/builtins/ukernel/pack.h"

static inline iree_uk_type_t iree_uk_pack_in_type(iree_uk_pack_type_t type) {
  return iree_uk_untie_type(0, type);
}

static inline iree_uk_type_t iree_uk_pack_out_type(iree_uk_pack_type_t type) {
  return iree_uk_untie_type(1, type);
}

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

// Returns the tile function to use for the pack op with the given params.
iree_uk_pack_tile_func_t iree_uk_pack_select_tile_func(
    const iree_uk_pack_params_t* params);

#endif  // IREE_BUILTINS_UKERNEL_PACK_INTERNAL_H_
