// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_VM_UTIL_CC_H_
#define IREE_TOOLING_VM_UTIL_CC_H_

#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/internal/span.h"
#include "iree/hal/api.h"
#include "iree/tooling/vm_util.h"
#include "iree/vm/api.h"

namespace iree {

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
Status ParseToVariantList(iree_hal_allocator_t* device_allocator,
                          iree::span<const std::string> input_strings,
                          iree_allocator_t host_allocator,
                          iree_vm_list_t** out_list);

// Prints a variant list to |out_string|.
Status PrintVariantList(iree_vm_list_t* variant_list, size_t max_element_count,
                        std::string* out_string);

inline Status PrintVariantList(iree_vm_list_t* variant_list,
                               std::string* out_string) {
  return PrintVariantList(variant_list, 1024, out_string);
}

// Prints a variant list to stdout.
inline Status PrintVariantList(iree_vm_list_t* variant_list,
                               size_t max_element_count = 1024) {
  return iree_tooling_variant_list_fprint(variant_list, max_element_count,
                                          stdout);
}

}  // namespace iree

#endif  // IREE_TOOLING_VM_UTIL_CC_H_
