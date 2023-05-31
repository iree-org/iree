// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/mmt4d_internal.h"
#include "iree/builtins/ukernel/pack_internal.h"
#include "iree/builtins/ukernel/query_tile_sizes_internal.h"
#include "iree/builtins/ukernel/unpack_internal.h"

#if defined(IREE_UK_HAVE_WEAK)

IREE_UK_WEAK iree_uk_mmt4d_tile_func_t
iree_uk_mmt4d_select_tile_func_arch(const iree_uk_mmt4d_params_t* params) {
  return 0;
}

IREE_UK_WEAK iree_uk_pack_tile_func_t
iree_uk_pack_select_tile_func_arch(const iree_uk_pack_params_t* params) {
  return 0;
}

IREE_UK_WEAK iree_uk_unpack_tile_func_t
iree_uk_unpack_select_tile_func_arch(const iree_uk_unpack_params_t* params) {
  return 0;
}

IREE_UK_WEAK bool iree_uk_query_matmul_tile_sizes_arch(
    const iree_uk_query_tile_sizes_2d_params_t* params,
    iree_uk_matmul_tile_sizes_t* out_matmul_tile_sizes) {
  return false;
}

#endif  // defined(IREE_UK_HAVE_WEAK)
