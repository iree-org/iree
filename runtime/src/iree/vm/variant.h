// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_VARIANT_H_
#define IREE_VM_VARIANT_H_

#include "iree/vm/ref.h"
#include "iree/vm/value.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// An variant value that can be either a primitive value type or a ref type.
// Each variant value stores its type but users are required to check the type
// prior to accessing any of the data.
typedef struct iree_vm_variant_t {
  iree_vm_type_def_t type;
  union {
    // TODO(benvanik): replace with iree_vm_value_t. Don't want to pay for 2x
    // the type storage, though.
    int8_t i8;
    int16_t i16;
    int32_t i32;
    int64_t i64;
    float f32;
    double f64;
    iree_vm_ref_t ref;
    uint8_t value_storage[IREE_VM_VALUE_STORAGE_SIZE];  // max size of all value
                                                        // types
  };
} iree_vm_variant_t;

// Returns an empty variant.
static inline iree_vm_variant_t iree_vm_variant_empty(void) {
  iree_vm_variant_t result;
  result.type = iree_vm_type_def_make_variant_type();
  result.ref = iree_vm_ref_null();
  return result;
}

// Returns true if |variant| is empty (no value/NULL ref).
static inline bool iree_vm_variant_is_empty(iree_vm_variant_t variant) {
  return iree_vm_type_def_is_variant(&variant.type);
}

// Returns true if |variant| represents a primitive value.
static inline bool iree_vm_variant_is_value(iree_vm_variant_t variant) {
  return iree_vm_type_def_is_value(&variant.type);
}

// Returns true if |variant| represents a non-NULL ref type.
static inline bool iree_vm_variant_is_ref(iree_vm_variant_t variant) {
  return iree_vm_type_def_is_ref(&variant.type);
}

// Makes a variant containing the given primitive |value|.
static inline iree_vm_variant_t iree_vm_make_variant_value(
    iree_vm_value_t value) {
  iree_vm_variant_t result = iree_vm_variant_empty();
  result.type.value_type = value.type;
  memcpy(result.value_storage, value.value_storage,
         sizeof(result.value_storage));
  return result;
}

// Makes a variant containing the given |ref| type with assignment semantics.
static inline iree_vm_variant_t iree_vm_make_variant_ref_assign(
    iree_vm_ref_t ref) {
  iree_vm_variant_t result = iree_vm_variant_empty();
  result.type.ref_type = ref.type;
  result.ref = ref;
  return result;
}

// Returns the primitive value contained within |variant|, if any.
// If the variant is not a value type the return will be the same as
// iree_vm_value_make_none.
static inline iree_vm_value_t iree_vm_variant_value(iree_vm_variant_t variant) {
  iree_vm_value_t value;
  value.type = variant.type.value_type;
  memcpy(value.value_storage, variant.value_storage,
         sizeof(value.value_storage));
  return value;
}

// Resets |variant| to empty in-place and releases the contained ref, if set.
static inline void iree_vm_variant_reset(iree_vm_variant_t* variant) {
  if (!variant) return;
  if (iree_vm_variant_is_ref(*variant)) {
    iree_vm_ref_release(&variant->ref);
  }
  *variant = iree_vm_variant_empty();
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_VARIANT_H_
