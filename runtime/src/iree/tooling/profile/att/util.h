// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_PROFILE_ATT_UTIL_H_
#define IREE_TOOLING_PROFILE_ATT_UTIL_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Grows an array to hold at least |element_count| entries.
iree_status_t iree_profile_att_grow_array(iree_allocator_t host_allocator,
                                          iree_host_size_t element_count,
                                          iree_host_size_t element_size,
                                          iree_host_size_t* inout_capacity,
                                          void** inout_ptr);

// Copies |value| into a NUL-terminated string allocated from |host_allocator|.
iree_status_t iree_profile_att_copy_cstring(iree_string_view_t value,
                                            iree_allocator_t host_allocator,
                                            char** out_string);

// Returns a string view for |string|, or an empty view when |string| is NULL.
iree_string_view_t iree_profile_att_cstring_view_or_empty(const char* string);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_PROFILE_ATT_UTIL_H_
