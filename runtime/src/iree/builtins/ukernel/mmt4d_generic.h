// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_MMT4D_GENERIC_H_
#define IREE_BUILTINS_UKERNEL_MMT4D_GENERIC_H_

#include "iree/builtins/ukernel/mmt4d.h"

int iree_ukernel_mmt4d_f32f32f32_generic(
    const iree_ukernel_mmt4d_f32f32f32_params_t* params);
int iree_ukernel_mmt4d_i8i8i32_generic(
    const iree_ukernel_mmt4d_i8i8i32_params_t* params);

#endif  // IREE_BUILTINS_UKERNEL_MMT4D_GENERIC_H_
