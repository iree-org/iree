// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_PACK_TILE_H_
#define IREE_BUILTINS_UKERNEL_PACK_TILE_H_

#include "iree/builtins/ukernel/pack.h"

// Returns the tile function to use for the pack op with the given params.
iree_uk_pack_tile_func_t iree_uk_pack_select_tile_func(
    const iree_uk_pack_params_t* params);

#endif  // IREE_BUILTINS_UKERNEL_PACK_TILE_H_
