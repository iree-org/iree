// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_QUERY_TILE_SIZES_H_
#define IREE_BUILTINS_UKERNEL_QUERY_TILE_SIZES_H_

#include "iree/builtins/ukernel/common.h"

// `query_tile_sizes` microkernel. Only used in the VMVX backend, because that
// is the only place where target information is not known at compile time,
// forcing deferral of tile-size selection to runtime.

// Parameters for a query_tile_sizes operation.
typedef struct iree_uk_query_tile_sizes_2d_params_t {
  iree_uk_uint32_t flags;
  iree_uk_index_t size0;
  iree_uk_index_t size1;
  const iree_uk_uint64_t* cpu_data;
} iree_uk_query_tile_sizes_2d_params_t;

typedef struct iree_uk_query_tile_sizes_2d_out_params_t {
  iree_uk_index_t tile_size0;
  iree_uk_index_t tile_size1;
} iree_uk_query_tile_sizes_2d_out_params_t;

IREE_UK_EXPORT int iree_uk_query_tile_sizes_2d(
    const iree_uk_query_tile_sizes_2d_params_t* params,
    iree_uk_query_tile_sizes_2d_out_params_t* out_params);

#endif  // IREE_BUILTINS_UKERNEL_QUERY_TILE_SIZES_H_
