// Copyright 2020 Google LLC
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

#ifndef IREE_MODULES_TENSORLIST_NATIVE_MODULE_H_
#define IREE_MODULES_TENSORLIST_NATIVE_MODULE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Registers the custom types used by the module.
// WARNING: not thread-safe; call at startup before using.
iree_status_t iree_tensorlist_module_register_types();

// Creates a native custom module.
// Modules may exist in multiple contexts and should be thread-safe and (mostly)
// immutable. Use the per-context allocated state for retaining data.
iree_status_t iree_tensorlist_module_create(iree_allocator_t allocator,
                                            iree_vm_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MODULES_TENSORLIST_NATIVE_MODULE_H_
