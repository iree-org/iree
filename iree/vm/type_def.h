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

#ifndef IREE_VM_TYPE_DEF_H_
#define IREE_VM_TYPE_DEF_H_

#include <stdint.h>

#include "iree/vm/ref.h"
#include "iree/vm/value.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Describes a type in the type table, mapping from a local module type ID to
// either a primitive value type or registered ref type.
//
// * ?: variant (value_type/ref_type == 0)
// * i8: primitive value (value_type != 0)
// * !vm.ref<?>: any ref value (ref_type == IREE_VM_REF_TYPE_ANY)
// * !vm.ref<!foo>: ref value of type !foo (ref_type > 0)
typedef struct {
  iree_vm_value_type_t value_type : 8;
  iree_vm_ref_type_t ref_type : 24;
} iree_vm_type_def_t;

static inline iree_vm_type_def_t iree_vm_type_def_make_variant_type() {
  iree_vm_type_def_t result;
  result.value_type = IREE_VM_VALUE_TYPE_NONE;
  result.ref_type = IREE_VM_REF_TYPE_NULL;
  return result;
}

static inline iree_vm_type_def_t iree_vm_type_def_make_value_type(
    iree_vm_value_type_t value_type) {
  iree_vm_type_def_t result;
  result.value_type = value_type;
  result.ref_type = IREE_VM_REF_TYPE_NULL;
  return result;
}

static inline iree_vm_type_def_t iree_vm_type_def_make_ref_type(
    iree_vm_ref_type_t ref_type) {
  iree_vm_type_def_t result;
  result.value_type = IREE_VM_VALUE_TYPE_NONE;
  result.ref_type = ref_type;
  return result;
}

#define iree_vm_type_def_is_value(v) \
  ((v)->value_type != IREE_VM_VALUE_TYPE_NONE)
#define iree_vm_type_def_is_ref(v) ((v)->ref_type != IREE_VM_REF_TYPE_NULL)
#define iree_vm_type_def_is_variant(v)           \
  ((v)->value_type == IREE_VM_VALUE_TYPE_NONE && \
   (v)->ref_type == IREE_VM_REF_TYPE_NULL)

// An variant value that can be either a primitive value type or a ref type.
// Each variant value stores its type but users are required to check the type
// prior to accessing any of the data.
typedef struct {
  iree_vm_type_def_t type;
  union {
    int8_t i8;
    int16_t i16;
    int32_t i32;
    int64_t i64;
    iree_vm_ref_t ref;

    uint8_t value_storage[IREE_VM_VALUE_STORAGE_SIZE];  // max size of all value
                                                        // types
  };
} iree_vm_variant_t;

#define iree_vm_variant_empty() \
  { {IREE_VM_VALUE_TYPE_NONE, IREE_VM_REF_TYPE_NULL}, {0}, }
#define iree_vm_variant_is_value(v) iree_vm_type_def_is_value(&(v).type)
#define iree_vm_variant_is_ref(v) iree_vm_type_def_is_ref(&(v).type)

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_TYPE_DEF_H_
