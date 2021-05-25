// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BUILTIN_TYPES_H_
#define IREE_VM_BUILTIN_TYPES_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Registers the builtin VM types. This must be called on startup. Safe to call
// multiple times.
IREE_API_EXPORT iree_status_t iree_vm_register_builtin_types(void);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BUILTIN_TYPES_H_
