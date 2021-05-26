// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MODULES_STRINGS_STRINGS_API_H_
#define IREE_MODULES_STRINGS_STRINGS_API_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct strings_string_t strings_string_t;
typedef struct strings_string_tensor_t strings_string_tensor_t;

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
