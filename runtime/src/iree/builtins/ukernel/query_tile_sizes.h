// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_QUERY_TILE_SIZES_H_
#define IREE_BUILTINS_UKERNEL_QUERY_TILE_SIZES_H_

#include "iree/builtins/ukernel/query_tile_sizes_types.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Main entry point.
IREE_UK_EXPORT void iree_uk_query_tile_sizes_2d(
    const iree_uk_query_tile_sizes_2d_params_t* params,
    iree_uk_query_tile_sizes_2d_out_params_t* out_params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BUILTINS_UKERNEL_QUERY_TILE_SIZES_H_
