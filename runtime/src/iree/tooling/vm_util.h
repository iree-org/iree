// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_VM_UTIL_H_
#define IREE_TOOLING_VM_UTIL_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// NOTE: this file is not best-practice and needs to be rewritten; consider this
// appropriate only for test code.

// Parses |input_strings| into a variant list of VM scalars and buffers.
// Scalars should be in the format:
//   type=value
// Buffers should be in the IREE standard shaped buffer format:
//   [shape]xtype=[value]
// described in iree/hal/api.h
// Uses |device_allocator| to allocate the buffers.
// The returned variant list must be freed by the caller.
iree_status_t iree_create_and_parse_to_variant_list(
    iree_hal_allocator_t* device_allocator, iree_string_view_t* input_strings,
    iree_host_size_t input_strings_count, iree_allocator_t host_allocator,
    iree_vm_list_t** out_list);

// Appends a variant list of VM scalars and buffers to |builder|.
// Prints scalars in the format:
//   value
// Prints buffers in the IREE standard shaped buffer format:
//   [shape]xtype=[value]
// described in
// https://github.com/iree-org/iree/tree/main/iree/hal/api.h
iree_status_t iree_append_variant_list(iree_vm_list_t* variant_list,
                                       size_t max_element_count,
                                       iree_string_builder_t* builder);

// Prints a variant list to a file.
iree_status_t iree_print_variant_list(iree_vm_list_t* variant_list,
                                      size_t max_element_count, FILE* file);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOOLING_VM_UTIL_H_
