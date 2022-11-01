// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_PACK_H_
#define IREE_BUILTINS_UKERNEL_PACK_H_

#include "iree/builtins/ukernel/pack_types.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Main entry point.
IREE_UKERNEL_EXPORT iree_ukernel_status_t
iree_ukernel_pack(const iree_ukernel_pack_params_t* params);

// Convert a status code to a human-readable string.
IREE_UKERNEL_EXPORT const char* iree_ukernel_status_message(
    iree_ukernel_status_t status);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BUILTINS_UKERNEL_PACK_H_
