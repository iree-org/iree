// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_MMT4D_H_
#define IREE_BUILTINS_UKERNEL_MMT4D_H_

#include "iree/builtins/ukernel/mmt4d_types.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Main entry point.
IREE_UKERNEL_EXPORT iree_ukernel_mmt4d_status_t
iree_ukernel_mmt4d(const iree_ukernel_mmt4d_params_t* params);

// Convert a status code to a human-readable string.
IREE_UKERNEL_EXPORT const char* iree_ukernel_mmt4d_status_message(
    iree_ukernel_mmt4d_status_t status);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BUILTINS_UKERNEL_MMT4D_H_
