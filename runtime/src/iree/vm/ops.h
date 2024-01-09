// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_OPS_H_
#define IREE_VM_OPS_H_

#include <math.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/math.h"
#include "iree/vm/value.h"

// The kernels below have undefined behavior in cases where the corresponding
// higher-level ops that map to them have undefined/implementation defined
// behavior and no additional checking was inserted as part of lowering.
// Avoiding UB is expected to happen above this level.
// Note: Setting this variable merely doesn't disable UBSAN.
#if defined(IREE_COMPILER_CLANG) && !IREE_VM_UBSAN_CHECKABLE_ENABLE
#pragma clang attribute push(__attribute__((no_sanitize("undefined"))), \
                             apply_to = function)
#endif  // IREE_COMPILER_CLANG && !IREE_VM_UBSAN_CHECKABLE_ENABLE

static inline int32_t vm_ext_i8i32u(int32_t);
static inline int32_t vm_ext_i8i32s(int32_t);
static inline int32_t vm_ext_i16i32u(int32_t);
static inline int32_t vm_ext_i16i32s(int32_t);

//===------------------------------------------------------------------===//
// Globals
//===------------------------------------------------------------------===//

static inline int32_t vm_global_load_i32(uint8_t* base, uint32_t byte_offset) {
  const int32_t* global_ptr = (const int32_t*)(base + byte_offset);
  return *global_ptr;
}

static inline void vm_global_store_i32(uint8_t* base, uint32_t byte_offset,
                                       int32_t value) {
  int32_t* global_ptr = (int32_t*)(base + byte_offset);
  *global_ptr = value;
}

//===------------------------------------------------------------------===//
// Buffers
//===------------------------------------------------------------------===//

static inline iree_status_t vm_buffer_compare(
    const iree_vm_buffer_t* lhs_buffer, iree_host_size_t lhs_offset,
    const iree_vm_buffer_t* rhs_buffer, iree_host_size_t rhs_offset,
    iree_host_size_t length, int32_t* result) {
  bool bool_result = false;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_compare_bytes(
      lhs_buffer, lhs_offset, rhs_buffer, rhs_offset, length, &bool_result));
  *result = bool_result ? 1 : 0;
  return iree_ok_status();
}

#define vm_buffer_fill_inline(buffer, element_offset, element_length, \
                              element_type, value)                    \
  element_type* IREE_RESTRICT buffer_ptr = NULL;                      \
  iree_vm_buffer_check_rw(buffer, element_offset, element_length,     \
                          element_type, buffer_ptr);                  \
  for (iree_host_size_t i = 0; i < element_length; ++i) {             \
    buffer_ptr[i] = value;                                            \
  }

#define vm_buffer_fill_i8_inline(buffer, element_offset, element_length, \
                                 value)                                  \
  uint8_t* IREE_RESTRICT buffer_ptr = NULL;                              \
  iree_vm_buffer_check_rw(buffer, offset, length, uint8_t, buffer_ptr);  \
  memset(buffer_ptr, value, length);
static inline iree_status_t vm_buffer_fill_i8(iree_vm_buffer_t* buffer,
                                              iree_host_size_t offset,
                                              iree_host_size_t length,
                                              uint8_t value) {
  vm_buffer_fill_i8_inline(buffer, offset, length, value);
  return iree_ok_status();
}

#define vm_buffer_fill_i16_inline(buffer, element_offset, element_length, \
                                  value)                                  \
  vm_buffer_fill_inline(buffer, element_offset, element_length, uint16_t, value)
static inline iree_status_t vm_buffer_fill_i16(iree_vm_buffer_t* buffer,
                                               iree_host_size_t offset,
                                               iree_host_size_t length,
                                               uint16_t value) {
  vm_buffer_fill_i16_inline(buffer, offset, length, value);
  return iree_ok_status();
}

#define vm_buffer_fill_i32_inline(buffer, element_offset, element_length, \
                                  value)                                  \
  vm_buffer_fill_inline(buffer, element_offset, element_length, uint32_t, value)
static inline iree_status_t vm_buffer_fill_i32(iree_vm_buffer_t* buffer,
                                               iree_host_size_t offset,
                                               iree_host_size_t length,
                                               uint32_t value) {
  vm_buffer_fill_i32_inline(buffer, offset, length, value);
  return iree_ok_status();
}

#define vm_buffer_fill_i64_inline(buffer, element_offset, element_length, \
                                  value)                                  \
  vm_buffer_fill_inline(buffer, element_offset, element_length, uint64_t, value)
static inline iree_status_t vm_buffer_fill_i64(iree_vm_buffer_t* buffer,
                                               iree_host_size_t offset,
                                               iree_host_size_t length,
                                               uint64_t value) {
  vm_buffer_fill_i64_inline(buffer, offset, length, value);
  return iree_ok_status();
}

#define vm_buffer_load_i8u_inline(buffer, element_offset, result)          \
  const uint8_t* IREE_RESTRICT buffer_ptr = NULL;                          \
  iree_vm_buffer_check_ro(buffer, element_offset, 1, uint8_t, buffer_ptr); \
  *result = vm_ext_i8i32u(*buffer_ptr);
static inline iree_status_t vm_buffer_load_i8u(iree_vm_buffer_t* buffer,
                                               iree_host_size_t offset,
                                               int32_t* result) {
  vm_buffer_load_i8u_inline(buffer, offset, result);
  return iree_ok_status();
}

#define vm_buffer_load_i8s_inline(buffer, element_offset, result)         \
  const int8_t* IREE_RESTRICT buffer_ptr = NULL;                          \
  iree_vm_buffer_check_ro(buffer, element_offset, 1, int8_t, buffer_ptr); \
  *result = vm_ext_i8i32s(*buffer_ptr);
static inline iree_status_t vm_buffer_load_i8s(iree_vm_buffer_t* buffer,
                                               iree_host_size_t offset,
                                               int32_t* result) {
  vm_buffer_load_i8s_inline(buffer, offset, result);
  return iree_ok_status();
}

#define vm_buffer_load_i16u_inline(buffer, element_offset, result)         \
  const int16_t* IREE_RESTRICT buffer_ptr = NULL;                          \
  iree_vm_buffer_check_ro(buffer, element_offset, 1, int16_t, buffer_ptr); \
  *result = vm_ext_i16i32u(*buffer_ptr);
static inline iree_status_t vm_buffer_load_i16u(iree_vm_buffer_t* buffer,
                                                iree_host_size_t offset,
                                                int32_t* result) {
  vm_buffer_load_i16u_inline(buffer, offset, result);
  return iree_ok_status();
}

#define vm_buffer_load_i16s_inline(buffer, element_offset, result)         \
  const int16_t* IREE_RESTRICT buffer_ptr = NULL;                          \
  iree_vm_buffer_check_ro(buffer, element_offset, 1, int16_t, buffer_ptr); \
  *result = vm_ext_i16i32s(*buffer_ptr);
static inline iree_status_t vm_buffer_load_i16s(iree_vm_buffer_t* buffer,
                                                iree_host_size_t offset,
                                                int32_t* result) {
  vm_buffer_load_i16s_inline(buffer, offset, result);
  return iree_ok_status();
}

#define vm_buffer_load_i32_inline(buffer, element_offset, result)          \
  const int32_t* IREE_RESTRICT buffer_ptr = NULL;                          \
  iree_vm_buffer_check_ro(buffer, element_offset, 1, int32_t, buffer_ptr); \
  *result = *buffer_ptr;
static inline iree_status_t vm_buffer_load_i32(iree_vm_buffer_t* buffer,
                                               iree_host_size_t offset,
                                               int32_t* result) {
  vm_buffer_load_i32_inline(buffer, offset, result);
  return iree_ok_status();
}

#define vm_buffer_load_i64_inline(buffer, element_offset, result)          \
  const int64_t* IREE_RESTRICT buffer_ptr = NULL;                          \
  iree_vm_buffer_check_ro(buffer, element_offset, 1, int64_t, buffer_ptr); \
  *result = *buffer_ptr;
static inline iree_status_t vm_buffer_load_i64(iree_vm_buffer_t* buffer,
                                               iree_host_size_t offset,
                                               int64_t* result) {
  vm_buffer_load_i64_inline(buffer, offset, result);
  return iree_ok_status();
}

#define vm_buffer_store_i8_inline(buffer, element_offset, value)           \
  uint8_t* IREE_RESTRICT buffer_ptr = NULL;                                \
  iree_vm_buffer_check_rw(buffer, element_offset, 1, uint8_t, buffer_ptr); \
  *buffer_ptr = value;
static inline iree_status_t vm_buffer_store_i8(iree_vm_buffer_t* buffer,
                                               iree_host_size_t offset,
                                               uint8_t value) {
  vm_buffer_store_i8_inline(buffer, offset, value);
  return iree_ok_status();
}

#define vm_buffer_store_i16_inline(buffer, element_offset, value)           \
  uint16_t* IREE_RESTRICT buffer_ptr = NULL;                                \
  iree_vm_buffer_check_rw(buffer, element_offset, 1, uint16_t, buffer_ptr); \
  *buffer_ptr = value;
static inline iree_status_t vm_buffer_store_i16(iree_vm_buffer_t* buffer,
                                                iree_host_size_t offset,
                                                uint16_t value) {
  vm_buffer_store_i16_inline(buffer, offset, value);
  return iree_ok_status();
}

#define vm_buffer_store_i32_inline(buffer, element_offset, value)           \
  uint32_t* IREE_RESTRICT buffer_ptr = NULL;                                \
  iree_vm_buffer_check_rw(buffer, element_offset, 1, uint32_t, buffer_ptr); \
  *buffer_ptr = value;
static inline iree_status_t vm_buffer_store_i32(iree_vm_buffer_t* buffer,
                                                iree_host_size_t offset,
                                                uint32_t value) {
  vm_buffer_store_i32_inline(buffer, offset, value);
  return iree_ok_status();
}

#define vm_buffer_store_i64_inline(buffer, element_offset, value)           \
  uint64_t* IREE_RESTRICT buffer_ptr = NULL;                                \
  iree_vm_buffer_check_rw(buffer, element_offset, 1, uint64_t, buffer_ptr); \
  *buffer_ptr = value;
static inline iree_status_t vm_buffer_store_i64(iree_vm_buffer_t* buffer,
                                                iree_host_size_t offset,
                                                uint64_t value) {
  vm_buffer_store_i64_inline(buffer, offset, value);
  return iree_ok_status();
}

//===------------------------------------------------------------------===//
// Conditional assignment
//===------------------------------------------------------------------===//

static inline int32_t vm_select_i32(int32_t condition, int32_t true_value,
                                    int32_t false_value) {
  return condition ? true_value : false_value;
}

//===------------------------------------------------------------------===//
// Native integer arithmetic
//===------------------------------------------------------------------===//

static inline int32_t vm_add_i32(int32_t lhs, int32_t rhs) { return lhs + rhs; }
static inline int32_t vm_sub_i32(int32_t lhs, int32_t rhs) { return lhs - rhs; }
static inline int32_t vm_mul_i32(int32_t lhs, int32_t rhs) { return lhs * rhs; }
static inline int32_t vm_div_i32s(int32_t lhs, int32_t rhs) {
  return lhs / rhs;
}
static inline int32_t vm_div_i32u(int32_t lhs, int32_t rhs) {
  return (int32_t)(((uint32_t)lhs) / ((uint32_t)rhs));
}
static inline int32_t vm_rem_i32s(int32_t lhs, int32_t rhs) {
  return lhs % rhs;
}
static inline int32_t vm_rem_i32u(int32_t lhs, int32_t rhs) {
  return (int32_t)(((uint32_t)lhs) % ((uint32_t)rhs));
}
static inline int32_t vm_fma_i32(int32_t a, int32_t b, int32_t c) {
  return a * b + c;
}
static inline int32_t vm_abs_i32(int32_t operand) { return abs(operand); }
static inline int32_t vm_min_i32s(int32_t lhs, int32_t rhs) {
  return rhs < lhs ? rhs : lhs;
}
static inline int32_t vm_min_i32u(int32_t lhs, int32_t rhs) {
  return (uint32_t)rhs < (uint32_t)lhs ? rhs : lhs;
}
static inline int32_t vm_max_i32s(int32_t lhs, int32_t rhs) {
  return lhs < rhs ? rhs : lhs;
}
static inline int32_t vm_max_i32u(int32_t lhs, int32_t rhs) {
  return (uint32_t)lhs < (uint32_t)rhs ? rhs : lhs;
}
static inline int32_t vm_not_i32(int32_t operand) {
  return (int32_t)(~((uint32_t)operand));
}
static inline int32_t vm_and_i32(int32_t lhs, int32_t rhs) { return lhs & rhs; }
static inline int32_t vm_or_i32(int32_t lhs, int32_t rhs) { return lhs | rhs; }
static inline int32_t vm_xor_i32(int32_t lhs, int32_t rhs) { return lhs ^ rhs; }
static inline int32_t vm_ctlz_i32(int32_t operand) {
  return (int32_t)iree_math_count_leading_zeros_u32((uint32_t)operand);
}

//===------------------------------------------------------------------===//
// Casting and type conversion/emulation
//===------------------------------------------------------------------===//

static inline int32_t vm_trunc_i32i8(int32_t operand) {
  return (uint8_t)((uint32_t)operand);
}
static inline int32_t vm_trunc_i32i16(int32_t operand) {
  return (uint16_t)((uint32_t)operand);
}
static inline int32_t vm_ext_i8i32s(int32_t operand) {
  return (int32_t)((int8_t)operand);
}
static inline int32_t vm_ext_i8i32u(int32_t operand) {
  return (uint32_t)((uint8_t)operand);
}
static inline int32_t vm_ext_i16i32s(int32_t operand) {
  return (int32_t)((int16_t)operand);
}
static inline int32_t vm_ext_i16i32u(int32_t operand) {
  return (uint32_t)((uint16_t)operand);
}

//===------------------------------------------------------------------===//
// Native bitwise shifts and rotates
//===------------------------------------------------------------------===//

static inline int32_t vm_shl_i32(int32_t operand, int32_t amount) {
  amount &= 0x1F;
  return (int32_t)(operand << amount);
}
static inline int32_t vm_shr_i32s(int32_t operand, int32_t amount) {
  amount &= 0x1F;
  return (int32_t)(operand >> amount);
}
static inline int32_t vm_shr_i32u(int32_t operand, int32_t amount) {
  amount &= 0x1F;
  return (int32_t)(((uint32_t)operand) >> amount);
}

//===------------------------------------------------------------------===//
// Comparison ops
//===------------------------------------------------------------------===//

static inline int32_t vm_cmp_eq_i32(int32_t lhs, int32_t rhs) {
  return (lhs == rhs) ? 1 : 0;
}
static inline int32_t vm_cmp_ne_i32(int32_t lhs, int32_t rhs) {
  return (lhs != rhs) ? 1 : 0;
}
static inline int32_t vm_cmp_lt_i32s(int32_t lhs, int32_t rhs) {
  return (lhs < rhs) ? 1 : 0;
}
static inline int32_t vm_cmp_lt_i32u(int32_t lhs, int32_t rhs) {
  return (((uint32_t)lhs) < ((uint32_t)rhs)) ? 1 : 0;
}
static inline int32_t vm_cmp_nz_i32(int32_t operand) {
  return (operand != 0) ? 1 : 0;
}
static inline int32_t vm_cmp_eq_ref(iree_vm_ref_t* lhs, iree_vm_ref_t* rhs) {
  return iree_vm_ref_equal(lhs, rhs) ? 1 : 0;
}
static inline int32_t vm_cmp_ne_ref(iree_vm_ref_t* lhs, iree_vm_ref_t* rhs) {
  return (!iree_vm_ref_equal(lhs, rhs)) ? 1 : 0;
}
static inline int32_t vm_cmp_nz_ref(iree_vm_ref_t* operand) {
  return (operand->ptr != NULL) ? 1 : 0;
}

//===------------------------------------------------------------------===//
// ExtI64: Globals
//===------------------------------------------------------------------===//

static inline int64_t vm_global_load_i64(uint8_t* base, uint32_t byte_offset) {
  const int64_t* global_ptr = (const int64_t*)(base + byte_offset);
  return *global_ptr;
}

static inline void vm_global_store_i64(uint8_t* base, uint32_t byte_offset,
                                       int64_t value) {
  int64_t* global_ptr = (int64_t*)(base + byte_offset);
  *global_ptr = value;
}

//===------------------------------------------------------------------===//
// ExtI64: Conditional assignment
//===------------------------------------------------------------------===//

static inline int64_t vm_select_i64(int32_t condition, int64_t true_value,
                                    int64_t false_value) {
  return condition ? true_value : false_value;
}

//===------------------------------------------------------------------===//
// ExtI64: Native integer arithmetic ops
//===------------------------------------------------------------------===//

static inline int64_t vm_add_i64(int64_t lhs, int64_t rhs) { return lhs + rhs; }
static inline int64_t vm_sub_i64(int64_t lhs, int64_t rhs) { return lhs - rhs; }
static inline int64_t vm_mul_i64(int64_t lhs, int64_t rhs) { return lhs * rhs; }
static inline int64_t vm_div_i64s(int64_t lhs, int64_t rhs) {
  return lhs / rhs;
}
static inline int64_t vm_div_i64u(int64_t lhs, int64_t rhs) {
  return (int64_t)(((uint64_t)lhs) / ((uint64_t)rhs));
}
static inline int64_t vm_rem_i64s(int64_t lhs, int64_t rhs) {
  return lhs % rhs;
}
static inline int64_t vm_rem_i64u(int64_t lhs, int64_t rhs) {
  return (int64_t)(((uint64_t)lhs) % ((uint64_t)rhs));
}
static inline int64_t vm_fma_i64(int64_t a, int64_t b, int64_t c) {
  return a * b + c;
}
static inline int64_t vm_abs_i64(int64_t operand) { return llabs(operand); }
static inline int64_t vm_min_i64s(int64_t lhs, int64_t rhs) {
  return rhs < lhs ? rhs : lhs;
}
static inline int64_t vm_min_i64u(int64_t lhs, int64_t rhs) {
  return (uint64_t)rhs < (uint64_t)lhs ? rhs : lhs;
}
static inline int64_t vm_max_i64s(int64_t lhs, int64_t rhs) {
  return lhs < rhs ? rhs : lhs;
}
static inline int64_t vm_max_i64u(int64_t lhs, int64_t rhs) {
  return (uint64_t)lhs < (uint64_t)rhs ? rhs : lhs;
}
static inline int64_t vm_not_i64(int64_t operand) {
  return (int64_t)(~((uint64_t)operand));
}
static inline int64_t vm_and_i64(int64_t lhs, int64_t rhs) { return lhs & rhs; }
static inline int64_t vm_or_i64(int64_t lhs, int64_t rhs) { return lhs | rhs; }
static inline int64_t vm_xor_i64(int64_t lhs, int64_t rhs) { return lhs ^ rhs; }
static inline int64_t vm_ctlz_i64(int64_t operand) {
  return (int64_t)iree_math_count_leading_zeros_u64((uint64_t)operand);
}

//===------------------------------------------------------------------===//
// ExtI64: Casting and type conversion/emulation
//===------------------------------------------------------------------===//

static inline int32_t vm_trunc_i64i32(int64_t operand) {
  return (uint32_t)((uint64_t)operand);
}
static inline int64_t vm_ext_i32i64s(int32_t operand) {
  return (int64_t)((int32_t)operand);
}
static inline int64_t vm_ext_i32i64u(int32_t operand) {
  return (uint64_t)((uint32_t)operand);
}

//===------------------------------------------------------------------===//
// ExtI64: Native bitwise shifts and rotates
//===------------------------------------------------------------------===//

static inline int64_t vm_shl_i64(int64_t operand, int32_t amount) {
  amount &= 0x3F;
  return (int64_t)(operand << amount);
}
static inline int64_t vm_shr_i64s(int64_t operand, int32_t amount) {
  amount &= 0x3F;
  return (int64_t)(operand >> amount);
}
static inline int64_t vm_shr_i64u(int64_t operand, int32_t amount) {
  amount &= 0x3F;
  return (int64_t)(((uint64_t)operand) >> amount);
}

//===------------------------------------------------------------------===//
// ExtI64: Comparison ops
//===------------------------------------------------------------------===//

static inline int32_t vm_cmp_eq_i64(int64_t lhs, int64_t rhs) {
  return (lhs == rhs) ? 1 : 0;
}
static inline int32_t vm_cmp_ne_i64(int64_t lhs, int64_t rhs) {
  return (lhs != rhs) ? 1 : 0;
}
static inline int32_t vm_cmp_lt_i64s(int64_t lhs, int64_t rhs) {
  return (lhs < rhs) ? 1 : 0;
}
static inline int32_t vm_cmp_lt_i64u(int64_t lhs, int64_t rhs) {
  return (((uint64_t)lhs) < ((uint64_t)rhs)) ? 1 : 0;
}
static inline int32_t vm_cmp_nz_i64(int64_t operand) {
  return (operand != 0) ? 1 : 0;
}

//===------------------------------------------------------------------===//
// ExtF32: Globals
//===------------------------------------------------------------------===//

static inline float vm_global_load_f32(uint8_t* base, uint32_t byte_offset) {
  const float* global_ptr = (const float*)(base + byte_offset);
  return *global_ptr;
}

static inline void vm_global_store_f32(uint8_t* base, uint32_t byte_offset,
                                       float value) {
  float* global_ptr = (float*)(base + byte_offset);
  *global_ptr = value;
}

//===------------------------------------------------------------------===//
// ExtF32: Buffers
//===------------------------------------------------------------------===//

#define vm_buffer_fill_f32_inline(buffer, element_offset, element_length, \
                                  value)                                  \
  vm_buffer_fill_inline(buffer, element_offset, element_length, float, value)
static inline iree_status_t vm_buffer_fill_f32(iree_vm_buffer_t* buffer,
                                               iree_host_size_t offset,
                                               iree_host_size_t length,
                                               float value) {
  vm_buffer_fill_f32_inline(buffer, offset, length, value);
  return iree_ok_status();
}

#define vm_buffer_load_f32_inline(buffer, element_offset, result)        \
  const float* IREE_RESTRICT buffer_ptr = NULL;                          \
  iree_vm_buffer_check_ro(buffer, element_offset, 1, float, buffer_ptr); \
  *result = *buffer_ptr;
static inline iree_status_t vm_buffer_load_f32(iree_vm_buffer_t* buffer,
                                               iree_host_size_t offset,
                                               float* result) {
  vm_buffer_load_f32_inline(buffer, offset, result);
  return iree_ok_status();
}

#define vm_buffer_store_f32_inline(buffer, element_offset, value)        \
  float* IREE_RESTRICT buffer_ptr = NULL;                                \
  iree_vm_buffer_check_rw(buffer, element_offset, 1, float, buffer_ptr); \
  *buffer_ptr = value;
static inline iree_status_t vm_buffer_store_f32(iree_vm_buffer_t* buffer,
                                                iree_host_size_t offset,
                                                float value) {
  vm_buffer_store_f32_inline(buffer, offset, value);
  return iree_ok_status();
}

//===------------------------------------------------------------------===//
// ExtF32: Conditional assignment
//===------------------------------------------------------------------===//

static inline float vm_select_f32(int32_t condition, float true_value,
                                  float false_value) {
  return condition ? true_value : false_value;
}

//===------------------------------------------------------------------===//
// ExtF32: Native floating-point arithmetic
//===------------------------------------------------------------------===//

static inline float vm_add_f32(float lhs, float rhs) { return lhs + rhs; }
static inline float vm_sub_f32(float lhs, float rhs) { return lhs - rhs; }
static inline float vm_mul_f32(float lhs, float rhs) { return lhs * rhs; }
static inline float vm_div_f32(float lhs, float rhs) { return lhs / rhs; }
static inline float vm_rem_f32(float lhs, float rhs) {
  return remainderf(lhs, rhs);
}
static inline float vm_fma_f32(float a, float b, float c) {
#ifdef FP_FAST_FMAF
  return fmaf(a, b, c);
#else
  return a * b + c;
#endif  // FP_FAST_FMAF
}
static inline float vm_abs_f32(float operand) { return fabsf(operand); }
static inline float vm_neg_f32(float operand) { return -operand; }
static inline float vm_ceil_f32(float operand) { return ceilf(operand); }
static inline float vm_floor_f32(float operand) { return floorf(operand); }
static inline float vm_round_f32(float operand) { return roundf(operand); }
static inline float vm_round_f32_even(float operand) {
#if __STC_VERSION__ >= 202300L  // C23
  return roundevenf(operand);
#else
  float rounded = roundf(operand);
  if (fabsf(operand - rounded) == 0.5f) {
    if (fmodf(rounded, 2.0f) != 0) {
      if (rounded > 0.0f) {
        rounded -= 1.0f;
      } else {
        rounded += 1.0f;
      }
    }
  }
  return rounded;
#endif  // C23
}
static inline float vm_min_f32(float lhs, float rhs) {
  return rhs < lhs ? rhs : lhs;
}
static inline float vm_max_f32(float lhs, float rhs) {
  return lhs < rhs ? rhs : lhs;
}

static inline float vm_atan_f32(float operand) { return atanf(operand); }
static inline float vm_atan2_f32(float y, float x) { return atan2f(y, x); }
static inline float vm_cos_f32(float operand) { return cosf(operand); }
static inline float vm_sin_f32(float operand) { return sinf(operand); }
static inline float vm_exp_f32(float operand) { return expf(operand); }
static inline float vm_exp2_f32(float operand) { return exp2f(operand); }
static inline float vm_expm1_f32(float operand) { return expm1f(operand); }
static inline float vm_log_f32(float operand) { return logf(operand); }
static inline float vm_log10_f32(float operand) { return log10f(operand); }
static inline float vm_log1p_f32(float operand) { return log1pf(operand); }
static inline float vm_log2_f32(float operand) { return log2f(operand); }
static inline float vm_pow_f32(float b, float e) { return powf(b, e); }
static inline float vm_rsqrt_f32(float operand) {
  return 1.0f / sqrtf(operand);
}
static inline float vm_sqrt_f32(float operand) { return sqrtf(operand); }
static inline float vm_tanh_f32(float operand) { return tanhf(operand); }
static inline float vm_erf_f32(float operand) { return erff(operand); }

//===------------------------------------------------------------------===//
// ExtF32: Casting and type conversion/emulation
//===------------------------------------------------------------------===//

static inline float vm_cast_si32f32(int32_t operand) { return (float)operand; }
static inline float vm_cast_ui32f32(int32_t operand) {
  return (float)(uint32_t)operand;
}
static inline int32_t vm_cast_f32si32(float operand) {
  return (int32_t)lroundf(operand);
}
static inline int32_t vm_cast_f32ui32(float operand) {
  return (uint32_t)llroundf(operand);
}
static inline float vm_bitcast_i32f32(int32_t operand) {
  float result;
  memcpy(&result, &operand, sizeof(result));
  return result;
}
static inline int32_t vm_bitcast_f32i32(float operand) {
  int32_t result;
  memcpy(&result, &operand, sizeof(result));
  return result;
}

//===------------------------------------------------------------------===//
// ExtF32: Comparison ops
//===------------------------------------------------------------------===//

static inline int32_t vm_cmp_eq_f32o(float lhs, float rhs) {
  return (lhs == rhs) ? 1 : 0;
}
static inline int32_t vm_cmp_eq_f32u(float lhs, float rhs) {
  return (isunordered(lhs, rhs) || (lhs == rhs)) ? 1 : 0;
}
static inline int32_t vm_cmp_ne_f32o(float lhs, float rhs) {
  return (lhs != rhs) ? 1 : 0;
}
static inline int32_t vm_cmp_ne_f32u(float lhs, float rhs) {
  return (isunordered(lhs, rhs) || (lhs != rhs)) ? 1 : 0;
}
static inline int32_t vm_cmp_lt_f32o(float lhs, float rhs) {
  return isless(lhs, rhs) ? 1 : 0;
}
static inline int32_t vm_cmp_lt_f32u(float lhs, float rhs) {
  return (isunordered(lhs, rhs) || isless(lhs, rhs)) ? 1 : 0;
}
static inline int32_t vm_cmp_lte_f32o(float lhs, float rhs) {
  return islessequal(lhs, rhs) ? 1 : 0;
}
static inline int32_t vm_cmp_lte_f32u(float lhs, float rhs) {
  return (isunordered(lhs, rhs) || islessequal(lhs, rhs)) ? 1 : 0;
}
static inline int32_t vm_cmp_nan_f32(float operand) {
  return isnan(operand) ? 1 : 0;
}

#if defined(IREE_COMPILER_CLANG) && !IREE_VM_UBSAN_CHECKABLE_ENABLE
#pragma clang attribute pop
#endif  // IREE_COMPILER_CLANG && !IREE_VM_UBSAN_CHECKABLE_ENABLE

#endif  // IREE_VM_OPS_H_
