// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_SAMPLES_CUSTOM_MODULES_NATIVE_MODULE_H_
#define IREE_SAMPLES_CUSTOM_MODULES_NATIVE_MODULE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/vm2/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_custom_message iree_custom_message_t;

// Creates a new !custom.message object with a copy of the given |value|.
iree_status_t iree_custom_message_create(iree_string_view_t value,
                                         iree_allocator_t allocator,
                                         iree_vm_ref_t* out_message_ref);

// Wraps an externally-owned |value| in a !custom.message object.
iree_status_t iree_custom_message_wrap(iree_string_view_t value,
                                       iree_allocator_t allocator,
                                       iree_vm_ref_t* out_message_ref);

// Copies the value of the !custom.message to the given output buffer and adds
// a \0 terminator.
iree_status_t iree_custom_message_read_value(iree_vm_ref_t* message_ref,
                                             char* buffer,
                                             size_t buffer_capacity);

// Registers the custom types used by the module.
// WARNING: not thread-safe; call at startup before using.
iree_status_t iree_custom_native_module_register_types();

// Creates a native custom module.
// Modules may exist in multiple contexts and should be thread-safe and (mostly)
// immutable. Use the per-context allocated state for retaining data.
iree_status_t iree_custom_native_module_create(iree_allocator_t allocator,
                                               iree_vm_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_SAMPLES_CUSTOM_MODULES_NATIVE_MODULE_H_
