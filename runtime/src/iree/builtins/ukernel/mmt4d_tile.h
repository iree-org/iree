// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_MMT4D_TILE_H_
#define IREE_BUILTINS_UKERNEL_MMT4D_TILE_H_

#include "iree/builtins/ukernel/mmt4d.h"

// Returns the tile function to use for the mmt4d op with the given params.
iree_uk_mmt4d_tile_func_t iree_uk_mmt4d_select_tile_func(
    const iree_uk_mmt4d_params_t* params);

#endif  // IREE_BUILTINS_UKERNEL_MMT4D_TILE_H_
