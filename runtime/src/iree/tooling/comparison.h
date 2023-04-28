// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_COMPARISON_H_
#define IREE_TOOLING_COMPARISON_H_

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Compares expected vs actual results and appends to |builder|.
// Returns true if all values match and false otherwise.
// Errors when performing comparison will abort the process.
// When all list elements match no output is written and otherwise
// newline-separated strings detailing the differing elements is appended.
bool iree_tooling_compare_variant_lists_and_append(
    iree_vm_list_t* expected_list, iree_vm_list_t* actual_list,
    iree_allocator_t host_allocator, iree_string_builder_t* builder);

// Compares expected vs actual results and appends to |file|.
// Refer to iree_tooling_compare_variant_lists_and_append for details.
bool iree_tooling_compare_variant_lists(iree_vm_list_t* expected_list,
                                        iree_vm_list_t* actual_list,
                                        iree_allocator_t host_allocator,
                                        FILE* file);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_COMPARISON_H_
