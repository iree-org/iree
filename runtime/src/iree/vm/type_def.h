// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
//
// Implementation note: since type defs are used frequently and live in tables
// and on the stack we pack the value and ref types together into a single
// native machine word. This exploits the fact that iree_vm_ref_type_t is a
// pointer to a struct that should always be aligned to some multiple of the
// native machine word and we'll have low bits to spare.
typedef struct iree_vm_type_def_t {
  uintptr_t value_type_bits : IREE_VM_REF_TYPE_TAG_BITS;
  uintptr_t ref_type_bits : IREE_VM_REF_TYPE_PTR_BITS;
} iree_vm_type_def_t;
static_assert(sizeof(iree_vm_type_def_t) == sizeof(uintptr_t),
              "iree_vm_type_def_t should be a single native machine word");
static_assert(
    IREE_VM_VALUE_TYPE_MAX <= IREE_VM_REF_TYPE_TAG_BIT_MASK,
    "iree_vm_value_type_t must fit within the iree_vm_ref_type_t tag bits");

#define iree_vm_type_def_as_value(v) (iree_vm_value_type_t)((v).value_type_bits)
#define iree_vm_type_def_as_ref(v) \
  (((iree_vm_ref_type_t)(v).ref_type_bits) << IREE_VM_REF_TYPE_TAG_BITS)

#define iree_vm_type_def_is_value(v) \
  (iree_vm_type_def_as_value(v) != IREE_VM_VALUE_TYPE_NONE)
#define iree_vm_type_def_is_ref(v) \
  (iree_vm_type_def_as_ref(v) != IREE_VM_REF_TYPE_NULL)
#define iree_vm_type_def_is_variant(v)                        \
  (iree_vm_type_def_as_value(v) == IREE_VM_VALUE_TYPE_NONE && \
   iree_vm_type_def_as_ref(v) == IREE_VM_REF_TYPE_NULL)
#define iree_vm_type_def_is_undefined(v) iree_vm_type_def_is_variant(v)

static bool iree_vm_type_def_equal(iree_vm_type_def_t a, iree_vm_type_def_t b) {
  return a.value_type_bits == b.value_type_bits &&
         a.ref_type_bits == b.ref_type_bits;
}

static inline iree_vm_type_def_t iree_vm_make_undefined_type_def(void) {
  iree_vm_type_def_t result;
  result.value_type_bits = IREE_VM_VALUE_TYPE_NONE;
  result.ref_type_bits = IREE_VM_REF_TYPE_NULL;
  return result;
}

static inline iree_vm_type_def_t iree_vm_make_value_type_def(
    iree_vm_value_type_t value_type) {
  iree_vm_type_def_t result;
  result.value_type_bits = value_type;
  result.ref_type_bits = IREE_VM_REF_TYPE_NULL;
  return result;
}

static inline iree_vm_type_def_t iree_vm_make_ref_type_def(
    iree_vm_ref_type_t ref_type) {
  iree_vm_type_def_t result;
  result.value_type_bits = IREE_VM_VALUE_TYPE_NONE;
  result.ref_type_bits = ref_type >> IREE_VM_REF_TYPE_TAG_BITS;
  return result;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_TYPE_DEF_H_
