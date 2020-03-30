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

IREE_VM_DECLARE_TYPE_ADAPTERS(strings_string, strings_string_t);
IREE_VM_DECLARE_TYPE_ADAPTERS(strings_string_tensor, strings_string_tensor_t);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MODULES_STRINGS_STRINGS_API_DETAIL_H_
