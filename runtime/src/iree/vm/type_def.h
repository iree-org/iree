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
typedef struct iree_vm_type_def_t {
  iree_vm_value_type_t value_type : 8;
  iree_vm_ref_type_t ref_type : 24;
} iree_vm_type_def_t;

static inline iree_vm_type_def_t iree_vm_type_def_make_variant_type(void) {
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

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_TYPE_DEF_H_
