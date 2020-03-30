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

#ifndef IREE_MODULES_STRINGS_STRINGS_API_H_
#define IREE_MODULES_STRINGS_STRINGS_API_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct strings_string strings_string_t;
typedef struct strings_string_tensor strings_string_tensor_t;

// Creates a string type.
iree_status_t strings_string_create(iree_string_view_t value,
                                    iree_allocator_t allocator,
                                    strings_string_t** out_message);

// Creates a string tensor type.
iree_status_t strings_string_tensor_create(
    iree_allocator_t allocator, const iree_string_view_t* value,
    int64_t value_count, const int32_t* shape, size_t rank,
    strings_string_tensor_t** out_message);

// Destroys a string type.
void strings_string_destroy(void* ptr);

// Destroys a string tensor
void strings_string_tensor_destroy(void* ptr);

// Returns the count of elements in the tensor.
iree_status_t strings_string_tensor_get_count(
    const strings_string_tensor_t* tensor, size_t* count);

// returns the list of stored string views.
iree_status_t strings_string_tensor_get_elements(
    const strings_string_tensor_t* tensor, iree_string_view_t* strs,
    size_t count, size_t offset);

// Returns the rank of the tensor.
iree_status_t strings_string_tensor_get_rank(
    const strings_string_tensor_t* tensor, int32_t* rank);

// Returns the shape of the tensor.
iree_status_t strings_string_tensor_get_shape(
    const strings_string_tensor_t* tensor, int32_t* shape, size_t rank);

// Returns the store string view using the provided indices.
iree_status_t strings_string_tensor_get_element(
    const strings_string_tensor_t* tensor, int32_t* indices, size_t rank,
    iree_string_view_t* str);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MODULES_STRINGS_STRINGS_API_H_
