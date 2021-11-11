// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_SAMPLES_CUSTOM_MODULES_MODULE_H_
#define IREE_SAMPLES_CUSTOM_MODULES_MODULE_H_

#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_custom_message_t iree_custom_message_t;

// Creates a new !custom.message object with a copy of the given |value|.
iree_status_t iree_custom_message_create(iree_string_view_t value,
                                         iree_allocator_t allocator,
                                         iree_custom_message_t** out_message);

// Wraps an externally-owned |value| in a !custom.message object.
iree_status_t iree_custom_message_wrap(iree_string_view_t value,
                                       iree_allocator_t allocator,
                                       iree_custom_message_t** out_message);

// Copies the value of the !custom.message to the given output buffer and adds
// a \0 terminator.
iree_status_t iree_custom_message_read_value(iree_custom_message_t* message,
                                             char* buffer,
                                             size_t buffer_capacity);

// Registers the custom types used by the module.
// WARNING: not thread-safe; call at startup before using.
iree_status_t iree_custom_native_module_register_types(void);

// Creates a native custom module.
// Modules may exist in multiple contexts and should be thread-safe and (mostly)
// immutable. Use the per-context allocated state for retaining data.
iree_status_t iree_custom_native_module_create(
    iree_allocator_t host_allocator, iree_hal_allocator_t* device_allocator,
    iree_vm_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

IREE_VM_DECLARE_TYPE_ADAPTERS(iree_custom_message, iree_custom_message_t);

#endif  // IREE_SAMPLES_CUSTOM_MODULES_MODULE_H_
