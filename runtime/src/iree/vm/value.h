// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_VALUE_H_
#define IREE_VM_VALUE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// TODO(benvanik): support variable size in modules. vm.imports would need index
// type and we'd have to make sure all native modules used this size type. It
// would be a compiler runtime flag and runtime compile flag.
typedef int32_t iree_vm_size_t;

// Defines the type of a primitive value.
typedef enum iree_vm_value_type_e {
  // Not a value type.
  IREE_VM_VALUE_TYPE_NONE = 0,
  // int8_t.
  IREE_VM_VALUE_TYPE_I8 = 1,
  // int16_t.
  IREE_VM_VALUE_TYPE_I16 = 2,
  // int32_t.
  IREE_VM_VALUE_TYPE_I32 = 3,
  // int64_t.
  IREE_VM_VALUE_TYPE_I64 = 4,
  // float.
  IREE_VM_VALUE_TYPE_F32 = 5,
  // double.
  IREE_VM_VALUE_TYPE_F64 = 6,

  IREE_VM_VALUE_TYPE_MAX = IREE_VM_VALUE_TYPE_F64,
  IREE_VM_VALUE_TYPE_COUNT = IREE_VM_VALUE_TYPE_MAX + 1,  // used for lookup
} iree_vm_value_type_t;
static_assert(
    IREE_VM_VALUE_TYPE_MAX < (1 << IREE_VM_REF_TYPE_TAG_BITS),
    "bitpacking in iree_vm_type_def_t requires this enum to be small");

// Maximum size, in bytes, of any value type we can represent.
#define IREE_VM_VALUE_STORAGE_SIZE 8

// A variant value type.
typedef struct iree_vm_value_t {
  iree_vm_value_type_t type;
  union {
    int8_t i8;
    int16_t i16;
    int32_t i32;
    int64_t i64;
    float f32;
    double f64;

    uint8_t value_storage[IREE_VM_VALUE_STORAGE_SIZE];  // max size of all value
                                                        // types
  };
} iree_vm_value_t;

static inline iree_vm_value_t iree_vm_value_make_none() {
  iree_vm_value_t result;
  result.type = IREE_VM_VALUE_TYPE_NONE;
  return result;
}

static inline iree_vm_value_t iree_vm_value_make_i8(int8_t value) {
  iree_vm_value_t result;
  result.type = IREE_VM_VALUE_TYPE_I8;
  result.i8 = value;
  return result;
}

static inline iree_vm_value_t iree_vm_value_make_i16(int16_t value) {
  iree_vm_value_t result;
  result.type = IREE_VM_VALUE_TYPE_I16;
  result.i16 = value;
  return result;
}

static inline iree_vm_value_t iree_vm_value_make_i32(int32_t value) {
  iree_vm_value_t result;
  result.type = IREE_VM_VALUE_TYPE_I32;
  result.i32 = value;
  return result;
}

// TODO(#5542): check the value type before accessing the union.
static inline int32_t iree_vm_value_get_i32(iree_vm_value_t *value) {
  return value->i32;
}

static inline iree_vm_value_t iree_vm_value_make_i64(int64_t value) {
  iree_vm_value_t result;
  result.type = IREE_VM_VALUE_TYPE_I64;
  result.i64 = value;
  return result;
}

// TODO(#5542): check the value type before accessing the union.
static inline int64_t iree_vm_value_get_i64(iree_vm_value_t *value) {
  return value->i64;
}

static inline iree_vm_value_t iree_vm_value_make_f32(float value) {
  iree_vm_value_t result;
  result.type = IREE_VM_VALUE_TYPE_F32;
  result.f32 = value;
  return result;
}

// TODO(#5542): check the value type before accessing the union.
static inline float iree_vm_value_get_f32(iree_vm_value_t *value) {
  return value->f32;
}

static inline iree_vm_value_t iree_vm_value_make_f64(double value) {
  iree_vm_value_t result;
  result.type = IREE_VM_VALUE_TYPE_F64;
  result.f64 = value;
  return result;
}

// TODO(#5542): check the value type before accessing the union.
static inline double iree_vm_value_get_f64(iree_vm_value_t *value) {
  return value->f64;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_VALUE_H_
