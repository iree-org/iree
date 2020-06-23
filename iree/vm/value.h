// Copyright 2019 Google LLC
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

#ifndef IREE_VM_VALUE_H_
#define IREE_VM_VALUE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Defines the type of a primitive value.
typedef enum {
  // Not a value type.
  IREE_VM_VALUE_TYPE_NONE = 0,
  // int32_t.
  IREE_VM_VALUE_TYPE_I32 = 1,
} iree_vm_value_type_t;

// A variant value type.
typedef struct iree_vm_value {
  iree_vm_value_type_t type;
  union {
    int32_t i32;
  };
} iree_vm_value_t;

#define IREE_VM_VALUE_MAKE_I32(value)            \
  {                                              \
    IREE_VM_VALUE_TYPE_I32, { (int32_t)(value) } \
  }

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_VALUE_H_
