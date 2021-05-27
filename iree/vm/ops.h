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

#ifndef IREE_VM_OPS_H_
#define IREE_VM_OPS_H_

#include <math.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/vm/value.h"

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
// Conditional assignment
//===------------------------------------------------------------------===//

static inline int32_t vm_select_i32(int32_t condition, int32_t true_value,
                                    int32_t false_value) {
  return condition ? true_value : false_value;
}

static inline float vm_select_f32(int32_t condition, float true_value,
                                  float false_value) {
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
static inline int32_t vm_not_i32(int32_t operand) {
  return (int32_t)(~((uint32_t)operand));
}
static inline int32_t vm_and_i32(int32_t lhs, int32_t rhs) { return lhs & rhs; }
static inline int32_t vm_or_i32(int32_t lhs, int32_t rhs) { return lhs | rhs; }
static inline int32_t vm_xor_i32(int32_t lhs, int32_t rhs) { return lhs ^ rhs; }

//===------------------------------------------------------------------===//
// Native floating-point arithmetic
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

static inline float vm_cast_si32f32(int32_t operand) { return (float)operand; }
static inline float vm_cast_ui32f32(int32_t operand) {
  return (float)(uint32_t)operand;
}
static inline int32_t vm_cast_f32si32(float operand) {
  return (int32_t)roundf(operand);
}
static inline int32_t vm_cast_f32ui32(float operand) {
  return (uint32_t)roundf(operand);
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
static inline int32_t vm_cmp_nan_f32(float operand) { return isnan(operand); }

//===------------------------------------------------------------------===//
// Control flow ops
//===------------------------------------------------------------------===//

static inline iree_status_t vm_fail_or_ok(int32_t status_code,
                                          iree_string_view_t message) {
  if (status_code != 0) {
    return iree_status_allocate(IREE_STATUS_FAILED_PRECONDITION, "<vm>", 0,
                                message);
  }
  return iree_ok_status();
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
static inline int64_t vm_not_i64(int64_t operand) {
  return (int64_t)(~((uint64_t)operand));
}
static inline int64_t vm_and_i64(int64_t lhs, int64_t rhs) { return lhs & rhs; }
static inline int64_t vm_or_i64(int64_t lhs, int64_t rhs) { return lhs | rhs; }
static inline int64_t vm_xor_i64(int64_t lhs, int64_t rhs) { return lhs ^ rhs; }

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
// Utility macros (Used for things that EmitC can't hadnle)
//===------------------------------------------------------------------===//

// Get the address of an array element
#define VM_ARRAY_ELEMENT_ADDRESS(array, index) &array[index]

// Release all refs from the given array
#define VM_REF_ARRAY_RELEASE(array)                          \
  for (int i = 0; i < IREE_ARRAYSIZE(array); i++) {          \
    iree_vm_ref_release(VM_ARRAY_ELEMENT_ADDRESS(array, i)); \
  }

// TODO(simon-camp): This macro should resemble the error handling part of the
// IREE_RETURN_IF_ERROR macro. There are two different definitions in
// iree/base/api.h depending on a feature flag.
#define VM_RETURN_IF_ERROR(status, array) \
  if (status) {                           \
    VM_REF_ARRAY_RELEASE(array);          \
    return status;                        \
  }

#define VM_RETURN_IF_LIST_NULL(list, array)                \
  if (!list) {                                             \
    VM_REF_ARRAY_RELEASE(array);                           \
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT); \
  }

#endif  // IREE_VM_OPS_H_
