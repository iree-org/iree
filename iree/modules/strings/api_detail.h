// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MODULES_STRINGS_STRINGS_API_DETAIL_H_
#define IREE_MODULES_STRINGS_STRINGS_API_DETAIL_H_

#include "iree/base/api.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct strings_string {
  iree_vm_ref_object_t ref_object;
  iree_allocator_t allocator;
  iree_string_view_t value;
} strings_string_t;

typedef struct strings_string_tensor {
  iree_vm_ref_object_t ref_object;
  iree_allocator_t allocator;
  iree_string_view_t* values;
  size_t count;
  const int32_t* shape;
  size_t rank;
} strings_string_tensor_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

IREE_VM_DECLARE_TYPE_ADAPTERS(strings_string, strings_string_t);
IREE_VM_DECLARE_TYPE_ADAPTERS(strings_string_tensor, strings_string_tensor_t);

#endif  // IREE_MODULES_STRINGS_STRINGS_API_DETAIL_H_
