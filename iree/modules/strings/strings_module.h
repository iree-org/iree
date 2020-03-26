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

#ifndef IREE_MODULES_STRINGS_STRINGS_MODULE_H_
#define IREE_MODULES_STRINGS_STRINGS_MODULE_H_

#include "iree/base/api.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct string string_t;
typedef struct string_tensor string_tensor_t;

// Registers the custom types used by the strings module.
// WARNING: Not threadsafe; call at startup before using..
iree_status_t strings_module_register_types();

// Creates a strings module.
// Modules may exist in multiple contexts should be thread-safe and immutable.
// Use the per-context allocated state for retaining data.
iree_status_t strings_module_create(iree_allocator_t allocator,
                                    iree_vm_module_t** out_module);

// Creates a string type.
iree_status_t string_create(iree_string_view_t value,
                            iree_allocator_t allocator, string_t** out_message);

// Creates a string tensor type.
iree_status_t string_tensor_create(iree_allocator_t allocator,
                                   iree_string_view_t* value,
                                   int64_t value_count, const int32_t* shape,
                                   size_t rank, string_tensor_t** out_message);

// Returns the count of elements in the tensor.
iree_status_t string_tensor_get_count(const string_tensor_t* tensor,
                                      size_t* count);

// returns the list of stored string views.
iree_status_t string_tensor_get_elements(const string_tensor_t* tensor,
                                         iree_string_view_t* strs, size_t count,
                                         size_t offset);

// Returns the rank of the tensor.
iree_status_t string_tensor_get_rank(const string_tensor_t* tensor,
                                     int32_t* rank);

// Returns the shape of the tensor.
iree_status_t string_tensor_get_shape(const string_tensor_t* tensor,
                                      int32_t* shape, size_t rank);

// Returns the store string view using the provided indices.
iree_status_t string_tensor_get_element(const string_tensor_t* tensor,
                                        int32_t* indices, size_t rank,
                                        iree_string_view_t* str);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

IREE_VM_DECLARE_TYPE_ADAPTERS(string, string_t);
IREE_VM_DECLARE_TYPE_ADAPTERS(string_tensor, string_tensor_t);

#endif  // IREE_MODULES_STRINGS_STRINGS_MODULE_H_
