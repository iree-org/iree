// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_ARCH_X86_64_MMT4D_X86_64_H_
#define IREE_BUILTINS_UKERNEL_ARCH_X86_64_MMT4D_X86_64_H_

#include "iree/builtins/ukernel/mmt4d.h"

// Returns the x86_64 tile function to use for the mmt4d with given params, or
// NULL if no suitable x86_64 tile function exists for these params, in which
// case the caller may fall back to a generic tile function.
iree_uk_mmt4d_tile_func_t iree_uk_mmt4d_select_tile_func_x86_64(
    const iree_uk_mmt4d_params_t* params);

#endif  // IREE_BUILTINS_UKERNEL_ARCH_X86_64_MMT4D_X86_64_H_
