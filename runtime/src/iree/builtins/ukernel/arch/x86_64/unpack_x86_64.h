// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_ARCH_X86_64_UNPACK_X86_64_H_
#define IREE_BUILTINS_UKERNEL_ARCH_X86_64_UNPACK_X86_64_H_

#include "iree/builtins/ukernel/unpack_internal.h"

// Returns the x86_64 tile function to use for the unpack op with given params,
// or NULL if none is available, so the caller may fall back to generic code.
iree_uk_unpack_tile_func_t iree_uk_unpack_select_tile_func_x86_64(
    const iree_uk_unpack_params_t* params);

#endif  // IREE_BUILTINS_UKERNEL_ARCH_X86_64_UNPACK_X86_64_H_
