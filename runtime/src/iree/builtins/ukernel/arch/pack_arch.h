// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_ARCH_PACK_ARCH_H_
#define IREE_BUILTINS_UKERNEL_ARCH_PACK_ARCH_H_

#include "iree/builtins/ukernel/pack.h"

// Returns the architecture-specific tile function to use for the pack op with
// given params, or NULL if no suitable architecture-specific tile function
// exists for these params, in which case the caller may fall back to a generic
// tile function.
iree_uk_pack_tile_func_t iree_uk_pack_select_tile_func_arch(
    const iree_uk_pack_params_t* params);

#endif  // IREE_BUILTINS_UKERNEL_ARCH_PACK_ARCH_H_
