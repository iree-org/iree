// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MODULES_STRINGS_STRINGS_MODULE_H_
#define IREE_MODULES_STRINGS_STRINGS_MODULE_H_

#include "iree/base/api.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Registers the custom types used by the strings module.
// WARNING: Not threadsafe; call at startup before using..
iree_status_t iree_strings_module_register_types(void);

// Creates a strings module.
// Modules may exist in multiple contexts should be thread-safe and immutable.
// Use the per-context allocated state for retaining data.
iree_status_t iree_strings_module_create(iree_allocator_t allocator,
                                         iree_vm_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MODULES_STRINGS_STRINGS_MODULE_H_
