// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/cpu.h"
#include "iree/base/internal/flags.h"
#include "iree/base/internal/math.h"
#include "iree/base/internal/path.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/tooling/context_util.h"
#include "iree/tooling/device_util.h"
#include "iree/tooling/vm_util.h"
#include "iree/vm/api.h"
#include "iree/vm/native_module_cc.h"

IREE_FLAG(bool, require_exact_results, true,
          "Requires floating point result elements to match exactly.");
IREE_FLAG(
    float, acceptable_fp_delta, 1e-5f,
    "Maximum absolute difference allowed with inexact floating point results.");
IREE_FLAG(
    int32_t, max_elements_to_check, 10000,
    "Maximum number of matrix elements to check for each matmul. For larger "
    "matrices, only every n-th element will be checked for some n chosed to "
    "stay just under that threshold and to avoid being a divisor of the inner "
    "dimension size to avoid special patterns. As the check uses a slow "
    "reference implementation, this is a trade-off between test latency and "
    "coverage. The value 0 means check all elements.");

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static const char* emoji(bool good) { return good ? "🦄" : "🐞"; }

static int calculate_check_every(iree_hal_dim_t m_size, iree_hal_dim_t n_size) {
  int check_every = 1;
  if (FLAG_max_elements_to_check) {
    check_every = ((m_size * n_size) + FLAG_max_elements_to_check - 1) /
                  FLAG_max_elements_to_check;
    if (check_every < 1) check_every = 1;
    if (check_every > 1)
      while ((n_size % check_every) == 0) ++check_every;
  }
  return check_every;
}

// Defines the type of a primitive value.
typedef enum iree_e2e_test_value_type_e {
  // Not a value type.
  IREE_E2E_TEST_VALUE_TYPE_NONE = 0,
  // int8_t.
  IREE_E2E_TEST_VALUE_TYPE_I8 = 1,
  // int16_t.
  IREE_E2E_TEST_VALUE_TYPE_I16 = 2,
  // int32_t.
  IREE_E2E_TEST_VALUE_TYPE_I32 = 3,
  // int64_t.
  IREE_E2E_TEST_VALUE_TYPE_I64 = 4,
  // halft_t.
  IREE_E2E_TEST_VALUE_TYPE_F16 = 5,
  // float.
  IREE_E2E_TEST_VALUE_TYPE_F32 = 6,
  // double.
  IREE_E2E_TEST_VALUE_TYPE_F64 = 7,
  // bfloat16
  IREE_E2E_TEST_VALUE_TYPE_BF16 = 8,
} iree_e2e_test_value_type_t;

// Maximum size, in bytes, of any value type we can represent.
#define IREE_E2E_TEST_VALUE_STORAGE_SIZE 8

// A variant value type.
typedef struct iree_e2e_test_value_t {
  iree_e2e_test_value_type_t type;
  union {
    int8_t i8;
    int16_t i16;
    int32_t i32;
    int64_t i64;
    float f32;
    uint16_t f16_u16;
    uint16_t bf16_u16;
    double f64;
    uint8_t value_storage[IREE_E2E_TEST_VALUE_STORAGE_SIZE];  // max size of all
                                                              // value types
  };
} iree_e2e_test_value_t;

static inline iree_e2e_test_value_t iree_e2e_test_value_make_none() {
  iree_e2e_test_value_t result;
  result.type = IREE_E2E_TEST_VALUE_TYPE_NONE;
  return result;
}

static inline iree_e2e_test_value_t iree_e2e_test_value_make_i8(int8_t value) {
  iree_e2e_test_value_t result;
  result.type = IREE_E2E_TEST_VALUE_TYPE_I8;
  result.i8 = value;
  return result;
}

static inline iree_e2e_test_value_t iree_e2e_test_value_make_i16(
    int16_t value) {
  iree_e2e_test_value_t result;
  result.type = IREE_E2E_TEST_VALUE_TYPE_I16;
  result.i16 = value;
  return result;
}

static inline iree_e2e_test_value_t iree_e2e_test_value_make_i32(
    int32_t value) {
  iree_e2e_test_value_t result;
  result.type = IREE_E2E_TEST_VALUE_TYPE_I32;
  result.i32 = value;
  return result;
}

static inline iree_e2e_test_value_t iree_e2e_test_value_make_f16(
    uint16_t value) {
  iree_e2e_test_value_t result;
  result.type = IREE_E2E_TEST_VALUE_TYPE_F16;
  result.f16_u16 = value;
  return result;
}

static inline iree_e2e_test_value_t iree_e2e_test_value_make_bf16(
    uint16_t value) {
  iree_e2e_test_value_t result;
  result.type = IREE_E2E_TEST_VALUE_TYPE_BF16;
  result.bf16_u16 = value;
  return result;
}

static inline iree_e2e_test_value_t iree_e2e_test_value_make_f32(float value) {
  iree_e2e_test_value_t result;
  result.type = IREE_E2E_TEST_VALUE_TYPE_F32;
  result.f32 = value;
  return result;
}

//===----------------------------------------------------------------------===//
// Reference matmul
//===----------------------------------------------------------------------===//

// Reads an element from a mapped row-major matrix buffer.
static iree_e2e_test_value_t read_matrix_element(
    iree_hal_dim_t m_size, iree_hal_dim_t n_size,
    iree_hal_element_type_t result_type, const void* data, iree_hal_dim_t m,
    iree_hal_dim_t n) {
  iree_host_size_t index = n + m * n_size;
  (void)m_size;
  if (iree_hal_element_type_is_integer(result_type, 8)) {
    return iree_e2e_test_value_make_i8(((int8_t*)data)[index]);
  } else if (iree_hal_element_type_is_integer(result_type, 16)) {
    return iree_e2e_test_value_make_i16(((int16_t*)data)[index]);
  } else if (iree_hal_element_type_is_integer(result_type, 32)) {
    return iree_e2e_test_value_make_i32(((int32_t*)data)[index]);
  } else if (result_type == IREE_HAL_ELEMENT_TYPE_FLOAT_16) {
    return iree_e2e_test_value_make_f16(((uint16_t*)data)[index]);
  } else if (result_type == IREE_HAL_ELEMENT_TYPE_BFLOAT_16) {
    return iree_e2e_test_value_make_bf16(((uint16_t*)data)[index]);
  } else if (result_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32) {
    return iree_e2e_test_value_make_f32(((float*)data)[index]);
  }
  iree_status_abort(iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                     "unhandled matmul result type"));
  return iree_e2e_test_value_make_none();
}

// Get the shape of a buffer_view that is a matrix, i.e. 2D shape.
static iree_status_t get_matrix_shape(iree_hal_buffer_view_t* buffer_view,
                                      iree_hal_dim_t* dims) {
  iree_host_size_t shape_rank = iree_hal_buffer_view_shape_rank(buffer_view);
  if (shape_rank != 2) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "expected a matrix (2D tensor) shape, got a %" PRIhsz
        "-dimensional shape",
        shape_rank);
  }
  dims[0] = iree_hal_buffer_view_shape_dim(buffer_view, 0);
  dims[1] = iree_hal_buffer_view_shape_dim(buffer_view, 1);
  if (!(dims[0] > 0 && dims[1] > 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected matrix dims to be positive, got %" PRIdim
                            "x%" PRIdim,
                            dims[0], dims[1]);
  }
  return iree_ok_status();
}

#define REFERENCE_MATMUL(LHSTYPE, RHSTYPE, RESTYPE, ACCTYPE)                  \
  static void reference_matmul_##LHSTYPE##_##RHSTYPE##_##RESTYPE##_##ACCTYPE( \
      iree_hal_dim_t m_size, iree_hal_dim_t k_size, iree_hal_dim_t n_size,    \
      iree_hal_element_type_t lhs_type, iree_hal_element_type_t rhs_type,     \
      iree_hal_element_type_t acc_type, const LHSTYPE* lhs_data,              \
      const RHSTYPE* rhs_data, const ACCTYPE* acc_data, RESTYPE* result_data, \
      iree_hal_dim_t m, iree_hal_dim_t n) {                                   \
    ACCTYPE acc = acc_data ? acc_data[n + m * n_size] : 0;                    \
    for (iree_hal_dim_t k = 0; k < k_size; ++k) {                             \
      LHSTYPE lhs_value = lhs_data[k + m * k_size];                           \
      RHSTYPE rhs_value = rhs_data[n + k * n_size];                           \
      acc += (ACCTYPE)lhs_value * (ACCTYPE)rhs_value;                         \
    }                                                                         \
    result_data[n + m * n_size] = acc;                                        \
  }

// Reference mamtul instantiations from macro REFERENCE_MATMUL
// for the f32 input, f32 accumlation, and f32 result.
// [float <= float * float + float]
REFERENCE_MATMUL(float, float, float, float)

// Reference mamtul instantiations from macro REFERENCE_MATMUL
// for the int8_t input, int32_t accumlation, and int32_t result.
// [i32 <= i8 * i8 + i32]
REFERENCE_MATMUL(int8_t, int8_t, int32_t, int32_t)

// Reference mamtul instantiations from macro REFERENCE_MATMUL
// for the int8_t input, int32_t accumlation, and int32_t result.
// [i32 <= i32 * i32 + i32]
REFERENCE_MATMUL(int32_t, int32_t, int32_t, int32_t)

// Reference mamtul for the f16 input, f16 accumlation, and f16 result.
// [f16 <= f16 * f16 + f16]
static void reference_matmul_f16_f16_f16_f16(
    iree_hal_dim_t m_size, iree_hal_dim_t k_size, iree_hal_dim_t n_size,
    iree_hal_element_type_t lhs_type, iree_hal_element_type_t rhs_type,
    iree_hal_element_type_t acc_type, const uint16_t* lhs_data,
    const uint16_t* rhs_data, const uint16_t* acc_data, uint16_t* result_data,
    iree_hal_dim_t m, iree_hal_dim_t n) {
  float acc = acc_data ? iree_math_f16_to_f32(acc_data[n + m * n_size]) : 0.f;
  for (iree_hal_dim_t k = 0; k < k_size; ++k) {
    acc += iree_math_f16_to_f32(lhs_data[k + m * k_size]) *
           iree_math_f16_to_f32(rhs_data[n + k * n_size]);
  }
  result_data[n + m * n_size] = iree_math_f32_to_f16(acc);
}

// Reference mamtul for the f16 input, f32 accumlation, and f32 result.
// [f32 <= f16 * f16 + f32]
static void reference_matmul_f16_f16_f32_f32(
    iree_hal_dim_t m_size, iree_hal_dim_t k_size, iree_hal_dim_t n_size,
    iree_hal_element_type_t lhs_type, iree_hal_element_type_t rhs_type,
    iree_hal_element_type_t acc_type, const uint16_t* lhs_data,
    const uint16_t* rhs_data, const float* acc_data, float* result_data,
    iree_hal_dim_t m, iree_hal_dim_t n) {
  float acc = acc_data ? acc_data[n + m * n_size] : 0.f;
  for (iree_hal_dim_t k = 0; k < k_size; ++k) {
    acc += iree_math_f16_to_f32(lhs_data[k + m * k_size]) *
           iree_math_f16_to_f32(rhs_data[n + k * n_size]);
  }
  result_data[n + m * n_size] = acc;
}

// Reference mamtul for the bf16 input, bf16 accumlation, and bf16 result.
// [bf16 <= bf16 * bf16 + bf16]
static void reference_matmul_bf16_bf16_bf16_bf16(
    iree_hal_dim_t m_size, iree_hal_dim_t k_size, iree_hal_dim_t n_size,
    iree_hal_element_type_t lhs_type, iree_hal_element_type_t rhs_type,
    iree_hal_element_type_t acc_type, const uint16_t* lhs_data,
    const uint16_t* rhs_data, const uint16_t* acc_data, uint16_t* result_data,
    iree_hal_dim_t m, iree_hal_dim_t n) {
  float acc = acc_data ? iree_math_bf16_to_f32(acc_data[n + m * n_size]) : 0.f;
  for (iree_hal_dim_t k = 0; k < k_size; ++k) {
    acc += iree_math_bf16_to_f32(lhs_data[k + m * k_size]) *
           iree_math_bf16_to_f32(rhs_data[n + k * n_size]);
  }
  result_data[n + m * n_size] = iree_math_f32_to_bf16(acc);
}

// Reference mamtul for the bf16 input, f32 accumlation, and f32 result.
// [f32 <= bf16 * bf16 + f32]
static void reference_matmul_bf16_bf16_f32_f32(
    iree_hal_dim_t m_size, iree_hal_dim_t k_size, iree_hal_dim_t n_size,
    iree_hal_element_type_t lhs_type, iree_hal_element_type_t rhs_type,
    iree_hal_element_type_t acc_type, const uint16_t* lhs_data,
    const uint16_t* rhs_data, const float* acc_data, float* result_data,
    iree_hal_dim_t m, iree_hal_dim_t n) {
  float acc = acc_data ? acc_data[n + m * n_size] : 0.f;
  for (iree_hal_dim_t k = 0; k < k_size; ++k) {
    acc += iree_math_bf16_to_f32(lhs_data[k + m * k_size]) *
           iree_math_bf16_to_f32(rhs_data[n + k * n_size]);
  }
  result_data[n + m * n_size] = acc;
}

// Helper for reference_matmul.
// Computes one element in the result matrix.
static iree_status_t reference_matmul_element(
    iree_hal_dim_t m_size, iree_hal_dim_t k_size, iree_hal_dim_t n_size,
    iree_hal_element_type_t lhs_type, iree_hal_element_type_t rhs_type,
    iree_hal_element_type_t acc_type, void* lhs_data, void* rhs_data,
    void* acc_data, void* result_data, iree_hal_dim_t m, iree_hal_dim_t n) {
  if (lhs_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32 &&
      rhs_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32 &&
      acc_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32) {
    reference_matmul_float_float_float_float(
        m_size, k_size, n_size, lhs_type, rhs_type, acc_type,
        (const float*)lhs_data, (const float*)rhs_data, (const float*)acc_data,
        (float*)result_data, m, n);
  } else if (iree_hal_element_type_is_integer(lhs_type, 8) &&
             iree_hal_element_type_is_integer(rhs_type, 8) &&
             iree_hal_element_type_is_integer(acc_type, 32)) {
    reference_matmul_int8_t_int8_t_int32_t_int32_t(
        m_size, k_size, n_size, lhs_type, rhs_type, acc_type,
        (const int8_t*)lhs_data, (const int8_t*)rhs_data,
        (const int32_t*)acc_data, (int32_t*)result_data, m, n);
  } else if (iree_hal_element_type_is_integer(lhs_type, 32) &&
             iree_hal_element_type_is_integer(rhs_type, 32) &&
             iree_hal_element_type_is_integer(acc_type, 32)) {
    reference_matmul_int32_t_int32_t_int32_t_int32_t(
        m_size, k_size, n_size, lhs_type, rhs_type, acc_type,
        (const int32_t*)lhs_data, (const int32_t*)rhs_data,
        (const int32_t*)acc_data, (int32_t*)result_data, m, n);
  } else if (lhs_type == IREE_HAL_ELEMENT_TYPE_FLOAT_16 &&
             rhs_type == IREE_HAL_ELEMENT_TYPE_FLOAT_16 &&
             acc_type == IREE_HAL_ELEMENT_TYPE_FLOAT_16) {
    reference_matmul_f16_f16_f16_f16(
        m_size, k_size, n_size, lhs_type, rhs_type, acc_type,
        (const uint16_t*)lhs_data, (const uint16_t*)rhs_data,
        (const uint16_t*)acc_data, (uint16_t*)result_data, m, n);
  } else if (lhs_type == IREE_HAL_ELEMENT_TYPE_FLOAT_16 &&
             rhs_type == IREE_HAL_ELEMENT_TYPE_FLOAT_16 &&
             acc_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32) {
    reference_matmul_f16_f16_f32_f32(
        m_size, k_size, n_size, lhs_type, rhs_type, acc_type,
        (const uint16_t*)lhs_data, (const uint16_t*)rhs_data,
        (const float*)acc_data, (float*)result_data, m, n);
  } else if (lhs_type == IREE_HAL_ELEMENT_TYPE_BFLOAT_16 &&
             rhs_type == IREE_HAL_ELEMENT_TYPE_BFLOAT_16 &&
             acc_type == IREE_HAL_ELEMENT_TYPE_BFLOAT_16) {
    reference_matmul_bf16_bf16_bf16_bf16(
        m_size, k_size, n_size, lhs_type, rhs_type, acc_type,
        (const uint16_t*)lhs_data, (const uint16_t*)rhs_data,
        (const uint16_t*)acc_data, (uint16_t*)result_data, m, n);
  } else if (lhs_type == IREE_HAL_ELEMENT_TYPE_BFLOAT_16 &&
             rhs_type == IREE_HAL_ELEMENT_TYPE_BFLOAT_16 &&
             acc_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32) {
    reference_matmul_bf16_bf16_f32_f32(
        m_size, k_size, n_size, lhs_type, rhs_type, acc_type,
        (const uint16_t*)lhs_data, (const uint16_t*)rhs_data,
        (const float*)acc_data, (float*)result_data, m, n);
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unhandled combination of element types in matmul");
  }
  return iree_ok_status();
}

// Reference matmul implementation, used to compare matmul results against.
static iree_status_t reference_matmul(
    iree_hal_dim_t m_size, iree_hal_dim_t k_size, iree_hal_dim_t n_size,
    iree_hal_element_type_t lhs_type, iree_hal_element_type_t rhs_type,
    iree_hal_element_type_t acc_type, iree_byte_span_t lhs_contents,
    iree_byte_span_t rhs_contents, iree_byte_span_t acc_contents,
    iree_byte_span_t result_contents, int compute_every) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, m_size);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, k_size);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, n_size);

  iree_host_size_t count = 0;
  for (iree_hal_dim_t m = 0; m < m_size; ++m) {
    for (iree_hal_dim_t n = 0; n < n_size; ++n) {
      if (++count < compute_every) continue;
      count = 0;
      IREE_RETURN_IF_ERROR(reference_matmul_element(
          m_size, k_size, n_size, lhs_type, rhs_type, acc_type,
          lhs_contents.data, rhs_contents.data, acc_contents.data,
          result_contents.data, m, n));
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Matmul comparison/logging
//===----------------------------------------------------------------------===//

typedef struct {
  iree_allocator_t host_allocator;
  iree_hal_dim_t m;
  iree_hal_dim_t k;
  iree_hal_dim_t n;
  iree_hal_element_type_t lhs_type;
  iree_hal_element_type_t rhs_type;
  iree_hal_element_type_t acc_type;
  iree_hal_element_type_t result_type;
  iree_byte_span_t lhs_contents;
  iree_byte_span_t rhs_contents;
  iree_byte_span_t acc_contents;
  iree_byte_span_t actual_contents;
  iree_byte_span_t expected_contents;
} matmul_results_t;

static void matmul_results_deinitialize(matmul_results_t* results);

static iree_status_t matmul_results_initialize(
    iree_hal_device_t* device, iree_hal_dim_t m_size, iree_hal_dim_t k_size,
    iree_hal_dim_t n_size, iree_hal_buffer_view_t* lhs,
    iree_hal_buffer_view_t* rhs, iree_hal_buffer_view_t* acc,
    iree_hal_buffer_view_t* result, iree_allocator_t host_allocator,
    matmul_results_t* out_results) {
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_results, 0, sizeof(*out_results));
  out_results->host_allocator = host_allocator;

  out_results->m = m_size;
  out_results->k = k_size;
  out_results->n = n_size;

  out_results->lhs_type = iree_hal_buffer_view_element_type(lhs);
  out_results->rhs_type = iree_hal_buffer_view_element_type(rhs);
  out_results->acc_type = iree_hal_buffer_view_element_type(result);
  out_results->result_type = iree_hal_buffer_view_element_type(result);

  iree_hal_buffer_t* lhs_buffer = iree_hal_buffer_view_buffer(lhs);
  iree_hal_buffer_t* rhs_buffer = iree_hal_buffer_view_buffer(rhs);
  iree_hal_buffer_t* acc_buffer = acc ? iree_hal_buffer_view_buffer(acc) : NULL;
  iree_hal_buffer_t* result_buffer = iree_hal_buffer_view_buffer(result);

  iree_status_t status = iree_ok_status();

  if (iree_status_is_ok(status)) {
    out_results->lhs_contents.data_length =
        iree_hal_buffer_byte_length(lhs_buffer);
    status = iree_allocator_malloc(host_allocator,
                                   out_results->lhs_contents.data_length,
                                   (void**)&out_results->lhs_contents.data);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_transfer_d2h(
        device, lhs_buffer, 0, out_results->lhs_contents.data,
        out_results->lhs_contents.data_length,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  }

  if (iree_status_is_ok(status)) {
    out_results->rhs_contents.data_length =
        iree_hal_buffer_byte_length(rhs_buffer);
    status = iree_allocator_malloc(host_allocator,
                                   out_results->rhs_contents.data_length,
                                   (void**)&out_results->rhs_contents.data);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_transfer_d2h(
        device, rhs_buffer, 0, out_results->rhs_contents.data,
        out_results->rhs_contents.data_length,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  }

  if (acc_buffer) {
    if (iree_status_is_ok(status)) {
      out_results->acc_contents.data_length =
          iree_hal_buffer_byte_length(acc_buffer);
      status = iree_allocator_malloc(host_allocator,
                                     out_results->acc_contents.data_length,
                                     (void**)&out_results->acc_contents.data);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_transfer_d2h(
          device, acc_buffer, 0, out_results->acc_contents.data,
          out_results->acc_contents.data_length,
          IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
    }
  }

  if (iree_status_is_ok(status)) {
    out_results->actual_contents.data_length =
        iree_hal_buffer_byte_length(result_buffer);
    status = iree_allocator_malloc(host_allocator,
                                   out_results->actual_contents.data_length,
                                   (void**)&out_results->actual_contents.data);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_transfer_d2h(
        device, result_buffer, 0, out_results->actual_contents.data,
        out_results->actual_contents.data_length,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  }

  if (iree_status_is_ok(status)) {
    out_results->expected_contents.data_length =
        iree_hal_buffer_byte_length(result_buffer);
    status = iree_allocator_malloc(
        host_allocator, out_results->expected_contents.data_length,
        (void**)&out_results->expected_contents.data);
  }

  if (!iree_status_is_ok(status)) {
    matmul_results_deinitialize(out_results);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void matmul_results_deinitialize(matmul_results_t* results) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(results->host_allocator, results->lhs_contents.data);
  iree_allocator_free(results->host_allocator, results->rhs_contents.data);
  if (!iree_byte_span_is_empty(results->acc_contents)) {
    iree_allocator_free(results->host_allocator, results->acc_contents.data);
  }
  iree_allocator_free(results->host_allocator, results->actual_contents.data);
  iree_allocator_free(results->host_allocator, results->expected_contents.data);

  IREE_TRACE_ZONE_END(z0);
}

// Enum controlling how many decimals to print floats with.
typedef enum precision_e {
  PRECISION_LOW,
  PRECISION_HIGH,
} precision_t;

// Prints a iree_e2e_test_value_t to a string buffer. Returns the number of
// characters written. Like snprintf.
static int snprintf_value(char* buf, size_t bufsize,
                          iree_e2e_test_value_t value, precision_t precision) {
  switch (value.type) {
    case IREE_E2E_TEST_VALUE_TYPE_I8:
      return snprintf(buf, bufsize, "%" PRIi8, value.i8);
    case IREE_E2E_TEST_VALUE_TYPE_I16:
      return snprintf(buf, bufsize, "%" PRIi16, value.i16);
    case IREE_E2E_TEST_VALUE_TYPE_I32:
      return snprintf(buf, bufsize, "%" PRIi32, value.i32);
    case IREE_E2E_TEST_VALUE_TYPE_I64:
      return snprintf(buf, bufsize, "%" PRIi64, value.i64);
    case IREE_E2E_TEST_VALUE_TYPE_F16:
      return snprintf(buf, bufsize,
                      precision == PRECISION_HIGH ? "%.5g" : "%.4g",
                      iree_math_f16_to_f32(value.f16_u16));
    case IREE_E2E_TEST_VALUE_TYPE_BF16:
      return snprintf(buf, bufsize,
                      precision == PRECISION_HIGH ? "%.5g" : "%.4g",
                      iree_math_bf16_to_f32(value.bf16_u16));
    case IREE_E2E_TEST_VALUE_TYPE_F32:
      return snprintf(buf, bufsize,
                      precision == PRECISION_HIGH ? "%.8g" : "%.4g", value.f32);
    case IREE_E2E_TEST_VALUE_TYPE_F64:
      return snprintf(buf, bufsize,
                      precision == PRECISION_HIGH ? "%.16g" : "%.4g",
                      value.f64);
    default:
      iree_status_abort(iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                         "unhandled value type"));
      return 0;
  }
}

// Returns true if |expected| and |actual| agree to tolerable accuracy.
static bool matmul_result_elements_agree(iree_e2e_test_value_t expected,
                                         iree_e2e_test_value_t actual) {
  if (expected.type != actual.type) {
    iree_status_abort(
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "mismatched types"));
    return false;
  }
  switch (expected.type) {
    case IREE_E2E_TEST_VALUE_TYPE_I32:
      return actual.i32 == expected.i32;
    // Since we fill buffers with small integers for floating point GEMMs
    // functional testing, we can test for bit-exactness on the actual and
    // expected values. Inexact results are only permitted when the
    // `require_exact_results` flag is set to `false`.
    case IREE_E2E_TEST_VALUE_TYPE_F16:
      if (actual.f16_u16 == expected.f16_u16) return true;
      if (FLAG_require_exact_results) return false;
      return fabsf(iree_math_f16_to_f32(actual.f16_u16) -
                   iree_math_f16_to_f32(expected.f16_u16)) <
             FLAG_acceptable_fp_delta;
    case IREE_E2E_TEST_VALUE_TYPE_BF16:
      if (actual.bf16_u16 == expected.bf16_u16) return true;
      if (FLAG_require_exact_results) return false;
      return fabsf(iree_math_bf16_to_f32(actual.bf16_u16) -
                   iree_math_bf16_to_f32(expected.bf16_u16)) <
             FLAG_acceptable_fp_delta;
    case IREE_E2E_TEST_VALUE_TYPE_F32:
      if (actual.f32 == expected.f32) return true;
      if (FLAG_require_exact_results) return false;
      return fabsf(actual.f32 - expected.f32) < FLAG_acceptable_fp_delta;
    default:
      iree_status_abort(iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                         "unhandled value type"));
      return false;
  }
}

// Returns the largest number of characters to print any matrix element.
static int get_max_elem_width(precision_t precision, iree_hal_dim_t rows,
                              iree_hal_dim_t row_start, iree_hal_dim_t row_end,
                              iree_hal_dim_t cols, iree_hal_dim_t col_start,
                              iree_hal_dim_t col_end,
                              iree_hal_element_type_t element_type,
                              const uint8_t* matrix) {
  int max_elem_width = 0;
  for (int row = row_start; row < row_end; row++) {
    for (int col = col_start; col < col_end; col++) {
      iree_e2e_test_value_t elem =
          read_matrix_element(rows, cols, element_type, matrix, row, col);
      // NOTE: iree_max is a macro and may evaluate its args twice.
      char buf[64];
      int this_elem_width = snprintf_value(buf, sizeof(buf), elem, precision);
      max_elem_width = iree_max(max_elem_width, this_elem_width);
    }
  }
  return max_elem_width;
}

// Prints |matrix| to |file|, with |label| as caption.
// |precision| controls how many decimals are printed for float values.
//
// If |other_matrix| is not NULL, then any matrix entries that disagree
// between |matrix| and |other_matrix| (according to
// matmul_result_elements_agree) are highlighted.
//
// |highlight| is either NULL or is a UTF-8 string that will be printed next to
// any entry of |matrix| that disagrees with the corresponding entry of
// |other_matrix|.
//
// |highlight| should be NULL if and only if |other_matrix| is NULL.
//
// In order for matrix columns to be properly laid out, the rendering of
// |highlight| in a fixed-width font should have the width of two regular Latin
// characters. According to
// https://www.unicode.org/reports/tr11/#Recommendations, a single emoji
// character should meet that requirement.
static void print_matrix(FILE* file, const char* label, precision_t precision,
                         iree_hal_dim_t rows, iree_hal_dim_t row_start,
                         iree_hal_dim_t row_end, iree_hal_dim_t cols,
                         iree_hal_dim_t col_start, iree_hal_dim_t col_end,
                         iree_hal_element_type_t element_type,
                         const uint8_t* matrix, const uint8_t* other_matrix,
                         const char* highlight) {
  IREE_ASSERT((other_matrix == NULL) == (highlight == NULL));
  int max_elem_width =
      get_max_elem_width(precision, rows, row_start, row_end, cols, col_start,
                         col_end, element_type, matrix);
  if (other_matrix) {
    // NOTE: iree_max is a macro and may evaluate its args twice.
    int other_matrix_max_elem_width =
        get_max_elem_width(precision, rows, row_start, row_end, cols, col_start,
                           col_end, element_type, other_matrix);
    max_elem_width = iree_max(max_elem_width, other_matrix_max_elem_width);
  }

  fprintf(file,
          "%s (rows %" PRIdsz "..%" PRIdsz " out of 0..%" PRIdsz
          ", columns %" PRIdsz "..%" PRIdsz " out of 0..%" PRIdsz ")\n",
          label, row_start, row_end - 1, rows - 1, col_start, col_end - 1,
          cols - 1);
  for (int row = row_start; row < row_end; row++) {
    for (int col = col_start; col < col_end; col++) {
      iree_e2e_test_value_t element =
          read_matrix_element(rows, cols, element_type, matrix, row, col);
      bool disagree = false;
      if (other_matrix) {
        iree_e2e_test_value_t other_element = read_matrix_element(
            rows, cols, element_type, other_matrix, row, col);
        disagree = !matmul_result_elements_agree(element, other_element);
      }
      char buf[64];
      snprintf_value(buf, sizeof(buf), element, precision);
      fprintf(file, "%*s", max_elem_width, buf);
      // See comment on |highlight| function parameter for why 2 spaces.
      // A 3rd space is added unconditionally to make it clear that a highlight
      // concerns the matrix entry to its left.
      fprintf(file, "%s ", disagree ? highlight : "  ");
    }
    fprintf(file, "\n");
  }
}

// Helper for check_matmul_results: handler for the failure case.
// If |file| is not NULL, detailed logging is written to it.
static iree_status_t check_matmul_failure(FILE* file,
                                          const matmul_results_t* results,
                                          iree_e2e_test_value_t actual_value,
                                          iree_e2e_test_value_t expected_value,
                                          iree_hal_dim_t row,
                                          iree_hal_dim_t col, int check_every) {
  if (!file || check_every > 1) {
    // No logging of errors with check_every>1 as most of the reference matrix
    // elements have not been computed. The caller is expected to retry with
    // check_every=1.
    return iree_make_status(IREE_STATUS_ABORTED);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  fprintf(file,
          "\n\nerror: the actual and expected result matrices disagree "
          "at row %" PRIdim ", column %" PRIdim ".\n\n",
          row, col);
  char actual_value_buf[32];
  char expected_value_buf[32];
  snprintf_value(actual_value_buf, sizeof(actual_value_buf), actual_value,
                 PRECISION_HIGH);
  snprintf_value(expected_value_buf, sizeof(expected_value_buf), expected_value,
                 PRECISION_HIGH);
  fprintf(file, "actual value: %s\n", actual_value_buf);
  fprintf(file, "expected value: %s\n", expected_value_buf);

  iree_hal_dim_t context = 8;
  const char* context_env = getenv("IREE_MATMUL_TEST_SHOW_CONTEXT");
  if (context_env) {
    if (1 != sscanf(context_env, "%" PRIdim, &context)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "failed to parse IREE_MATMUL_TEST_SHOW_CONTEXT "
                              "as \"%%" PRIdim "\"; got \"%s\"",
                              context_env);
    }
  }
  iree_hal_dim_t m_start =
      (iree_hal_dim_t)iree_max(0, (int64_t)row - (int64_t)context);
  iree_hal_dim_t m_end = iree_min(results->m, row + context);
  iree_hal_dim_t n_start =
      (iree_hal_dim_t)iree_max(0, (int64_t)col - (int64_t)context);
  iree_hal_dim_t n_end = iree_min(results->n, col + context);
  iree_hal_dim_t k_start = 0;
  iree_hal_dim_t k_end = iree_min(results->k, 2 * context);
  // [k_start, k_end) could be arbitrarily long at this point. Constrain it a
  // bit to avoid huge output.
  k_end = iree_min(k_end, k_start + 4 * context);

  fprintf(file, "\n");
  print_matrix(file, "left-hand side", PRECISION_LOW, results->m, m_start,
               m_end, results->k, k_start, k_end, results->lhs_type,
               results->lhs_contents.data, NULL, NULL);
  fprintf(file, "\n");
  print_matrix(file, "right-hand side", PRECISION_LOW, results->k, k_start,
               k_end, results->n, n_start, n_end, results->rhs_type,
               results->rhs_contents.data, NULL, NULL);
  fprintf(file, "\n");
  if (results->acc_contents.data) {
    print_matrix(file, "input accumulator", PRECISION_LOW, results->m, m_start,
                 m_end, results->n, n_start, n_end, results->acc_type,
                 results->acc_contents.data, NULL, NULL);
    fprintf(file, "\n");
  }
  print_matrix(file, "expected result", PRECISION_LOW, results->m, m_start,
               m_end, results->n, n_start, n_end, results->result_type,
               results->expected_contents.data, results->actual_contents.data,
               emoji(true));
  fprintf(file, "\n");
  print_matrix(file, "actual result", PRECISION_LOW, results->m, m_start, m_end,
               results->n, n_start, n_end, results->result_type,
               results->actual_contents.data, results->expected_contents.data,
               emoji(false));
  fprintf(file, "\n");

  IREE_TRACE_ZONE_END(z0);
  return iree_make_status(IREE_STATUS_ABORTED);
}

// Helper for check_matmul_results: the actual interesting part once we've
// obtained and validated the {m,k,n}_size values. On error, detailed logging is
// written to |file| if it is not NULL.
static iree_status_t check_matmul_results_impl(FILE* file,
                                               const matmul_results_t* results,
                                               int check_every) {
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, reference_matmul(results->m, results->k, results->n,
                           results->lhs_type, results->rhs_type,
                           results->acc_type, results->lhs_contents,
                           results->rhs_contents, results->acc_contents,
                           results->expected_contents, check_every));

  int count = 0;
  for (iree_hal_dim_t m = 0; m < results->m; ++m) {
    for (iree_hal_dim_t n = 0; n < results->n; ++n) {
      if (++count < check_every) continue;
      count = 0;
      iree_e2e_test_value_t actual_value =
          read_matrix_element(results->m, results->n, results->result_type,
                              results->actual_contents.data, m, n);
      iree_e2e_test_value_t expected_value =
          read_matrix_element(results->m, results->n, results->result_type,
                              results->expected_contents.data, m, n);
      if (!matmul_result_elements_agree(actual_value, expected_value)) {
        iree_status_t status = check_matmul_failure(
            file, results, actual_value, expected_value, m, n, check_every);
        IREE_TRACE_ZONE_END(z0);
        return status;
      }
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Given an actual matmul's inputs and output (all host-local), uses a reference
// matmul implementation on the same inputs to check if the output is correct.
// On error, detailed logging is written to |file| if it is not NULL.
static iree_status_t check_matmul_results(FILE* file,
                                          const matmul_results_t* results) {
  IREE_TRACE_ZONE_BEGIN(z0);
  int check_every = calculate_check_every(results->m, results->n);
  iree_status_t status = check_matmul_results_impl(file, results, check_every);
  if (!iree_status_is_ok(status) && check_every > 1) {
    // If we got a failure with check_every>1, that didn't log a useful
    // numerical summary, as most of the reference matrix entries hadn't been
    // computed. Rerun now with check_every=1 to get that numerical logging.
    iree_status_ignore(status);
    status = check_matmul_results_impl(file, results, 1);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// RNG utilities
//===----------------------------------------------------------------------===//

// Parameter for locally defined lcg similar to std::minstd_rand.
#define IREE_PRNG_MULTIPLIER 48271
#define IREE_PRNG_MODULUS 2147483647

// Writes an element of the given |element_type| with the given integral |value|
// to |dst|.
static void write_element(iree_hal_element_type_t element_type, int32_t value,
                          void* dst) {
#define WRITE_ELEMENT_CASE(ETYPE, CTYPE) \
  case IREE_HAL_ELEMENT_TYPE_##ETYPE:    \
    *(CTYPE*)dst = (CTYPE)value;         \
    break;

  switch (element_type) {
    WRITE_ELEMENT_CASE(INT_8, int8_t)
    WRITE_ELEMENT_CASE(INT_16, int16_t)
    WRITE_ELEMENT_CASE(INT_32, int32_t)
    WRITE_ELEMENT_CASE(INT_64, int64_t)
    WRITE_ELEMENT_CASE(SINT_8, int8_t)
    WRITE_ELEMENT_CASE(SINT_16, int16_t)
    WRITE_ELEMENT_CASE(SINT_32, int32_t)
    WRITE_ELEMENT_CASE(SINT_64, int64_t)
    WRITE_ELEMENT_CASE(UINT_8, uint8_t)
    WRITE_ELEMENT_CASE(UINT_16, uint16_t)
    WRITE_ELEMENT_CASE(UINT_32, uint32_t)
    WRITE_ELEMENT_CASE(UINT_64, uint64_t)
      // clang-format off
    case IREE_HAL_ELEMENT_TYPE_FLOAT_16:
      *(uint16_t*)dst = iree_math_f32_to_f16((float)value);
      break;
    case IREE_HAL_ELEMENT_TYPE_BFLOAT_16:
      *(uint16_t*)dst = iree_math_f32_to_bf16((float)value);
      break;
    WRITE_ELEMENT_CASE(FLOAT_32, float)
    WRITE_ELEMENT_CASE(FLOAT_64, double)
    // clang-format on
    default:
      IREE_ASSERT(false, "unhandled element type");
      break;
  }

#undef WRITE_ELEMENT_CASE
}

// Simple deterministic pseudorandom generator.
// This function is same as C++'s std::minstd_rand.
static uint32_t pseudorandom_uint32(uint32_t* state) {
  *state = (*state * IREE_PRNG_MULTIPLIER) % IREE_PRNG_MODULUS;
  return *state;
}

// Returns a random uint32_t in the range [0, range).
static inline uint32_t pseudorandom_range(uint32_t* state, uint32_t range) {
  return pseudorandom_uint32(state) % range;
}

// Get minimum and maximum for integer-valued uniform distribution.
static void get_min_max_for_element_type(iree_hal_element_type_t element_type,
                                         int32_t* min, int32_t* max) {
  switch (element_type) {
    case IREE_HAL_ELEMENT_TYPE_INT_8:
    case IREE_HAL_ELEMENT_TYPE_SINT_8:
      *min = -2;
      *max = +2;
      break;
    case IREE_HAL_ELEMENT_TYPE_UINT_8:
      *min = 0;
      *max = +2;
      break;
    case IREE_HAL_ELEMENT_TYPE_INT_16:
    case IREE_HAL_ELEMENT_TYPE_SINT_16:
    case IREE_HAL_ELEMENT_TYPE_FLOAT_16:
      *min = -4;
      *max = +4;
      break;
    case IREE_HAL_ELEMENT_TYPE_BFLOAT_16:
      *min = -2;
      *max = +2;
      break;
    case IREE_HAL_ELEMENT_TYPE_UINT_16:
      *min = 0;
      *max = +4;
      break;
    case IREE_HAL_ELEMENT_TYPE_INT_32:
    case IREE_HAL_ELEMENT_TYPE_SINT_32:
    case IREE_HAL_ELEMENT_TYPE_FLOAT_32:
      *min = -8;
      *max = +8;
      break;
    case IREE_HAL_ELEMENT_TYPE_UINT_32:
      *min = 0;
      *max = +8;
      break;
    case IREE_HAL_ELEMENT_TYPE_INT_64:
    case IREE_HAL_ELEMENT_TYPE_SINT_64:
    case IREE_HAL_ELEMENT_TYPE_FLOAT_64:
      *min = -16;
      *min = +16;
      break;
    case IREE_HAL_ELEMENT_TYPE_UINT_64:
      *min = 0;
      *max = +16;
      break;
    default:
      IREE_ASSERT(false, "unhandled element type");
      break;
  }
}

//===----------------------------------------------------------------------===//
// `matmul_test` custom module
//===----------------------------------------------------------------------===//
// This uses the C++ wrapper to keep things simple. Though easier to use it's
// got additional overhead/code-size bloat that doesn't matter in a test like
// this. Making a C module builder API that removes the boilerplate there is TBD
// so this file is written in C besides this module so that we can swap it back
// to being pure C in the future.

namespace {

using namespace iree;

class MatmulTestModuleState final {
 public:
  explicit MatmulTestModuleState(iree_allocator_t host_allocator)
      : host_allocator_(host_allocator) {}
  ~MatmulTestModuleState() = default;

  // Fills the destination span with pseudorandom values of the given
  // |element_type|. The given |seed| is passed to the pseudorandom generator.
  // The pseudorandom values are reproducible both across runs and across
  // machines.
  StatusOr<vm::ref<iree_hal_buffer_view_t>> GenerateRandomMatrix(
      const vm::ref<iree_hal_device_t> device, int64_t dim0, int64_t dim1,
      iree_hal_element_type_t element_type, int32_t seed) {
    iree_hal_dim_t dims[2] = {
        (iree_hal_dim_t)dim0,
        (iree_hal_dim_t)dim1,
    };
    iree_hal_buffer_params_t buffer_params = {0};
    buffer_params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
    buffer_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    buffer_params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
    vm::ref<iree_hal_buffer_view_t> result_view;
    struct callback_state_t {
      iree_hal_element_type_t element_type;
      int32_t seed;
    } callback_state = {
        element_type,
        seed,
    };
    IREE_RETURN_IF_ERROR(iree_hal_buffer_view_generate_buffer(
        device.get(), iree_hal_device_allocator(device.get()),
        IREE_ARRAYSIZE(dims), dims, element_type,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, buffer_params,
        +[](iree_hal_buffer_mapping_t* mapping, void* user_data) {
          callback_state_t callback_state = *(callback_state_t*)user_data;
          iree_byte_span_t span = mapping->contents;
          // Generate "uniform" integer-valued numbers in the range [min, max].
          int32_t min = 0;
          int32_t max = 0;
          get_min_max_for_element_type(callback_state.element_type, &min, &max);
          uint32_t range = (max - min + 1);
          iree_host_size_t element_byte_count =
              iree_hal_element_dense_byte_count(callback_state.element_type);
          uint8_t* data_end = span.data + span.data_length;
          uint32_t state = callback_state.seed;
          for (uint8_t* data = span.data; data < data_end;
               data += element_byte_count) {
            int32_t value = (int32_t)pseudorandom_range(&state, range) + min;
            write_element(callback_state.element_type, value, data);
          }
          return iree_ok_status();
        },
        &callback_state, &result_view));
    return std::move(result_view);
  }

  Status CheckMatmulResults(
      const vm::ref<iree_hal_device_t> device, int64_t m, int64_t k, int64_t n,
      const vm::ref<iree_hal_buffer_view_t> lhs,
      const vm::ref<iree_hal_buffer_view_t> rhs,
      const vm::ref<iree_hal_buffer_view_t> acc,
      const vm::ref<iree_hal_buffer_view_t> actual_result) {
    matmul_results_t results = {};
    IREE_RETURN_IF_ERROR(matmul_results_initialize(
        device.get(), (iree_hal_dim_t)m, (iree_hal_dim_t)k, (iree_hal_dim_t)n,
        lhs.get(), rhs.get(), acc.get(), actual_result.get(), host_allocator_,
        &results));
    iree_status_t status = check_matmul_results(stderr, &results);
    matmul_results_deinitialize(&results);
    return status;
  }

 private:
  iree_allocator_t host_allocator_;
};

static const vm::NativeFunction<MatmulTestModuleState>
    kMatmulTestModuleFunctions[] = {
        vm::MakeNativeFunction("generate_random_matrix",
                               &MatmulTestModuleState::GenerateRandomMatrix),
        vm::MakeNativeFunction("check_matmul_results",
                               &MatmulTestModuleState::CheckMatmulResults),
};

struct MatmulTestModule final : public vm::NativeModule<MatmulTestModuleState> {
  using vm::NativeModule<MatmulTestModuleState>::NativeModule;
  StatusOr<std::unique_ptr<MatmulTestModuleState>> CreateState(
      iree_allocator_t host_allocator) override {
    return std::make_unique<MatmulTestModuleState>(host_allocator);
  }
};

}  // namespace

static iree_status_t matmul_test_module_create(iree_vm_instance_t* instance,
                                               iree_allocator_t host_allocator,
                                               iree_vm_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;
  auto module = std::make_unique<MatmulTestModule>(
      "matmul_test", /*version=*/0, instance, host_allocator,
      iree::span<const vm::NativeFunction<MatmulTestModuleState>>(
          kMatmulTestModuleFunctions));
  *out_module = module.release()->interface();
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Test runner
//===----------------------------------------------------------------------===//

// Returns true if the |function| is a supported callable test function.
// We only support functions that are publicly exported, not an internal
// compiler/runtime function (__ prefixed), and take/return no args/results.
static iree_status_t check_test_function(iree_vm_function_t function,
                                         bool* out_is_valid) {
  *out_is_valid = true;

  iree_string_view_t function_name = iree_vm_function_name(&function);
  if (iree_string_view_starts_with(function_name,
                                   iree_make_cstring_view("__"))) {
    // Internal compiler/runtime support function.
    *out_is_valid = false;
  }

  iree_vm_function_signature_t function_signature =
      iree_vm_function_signature(&function);
  iree_host_size_t argument_count = 0;
  iree_host_size_t result_count = 0;
  IREE_RETURN_IF_ERROR(iree_vm_function_call_count_arguments_and_results(
      &function_signature, &argument_count, &result_count));
  if (argument_count || result_count) {
    // Takes args or has results we don't expect.
    *out_is_valid = false;
  }

  return iree_ok_status();
}

// Synchronous runs a test |function|.
// If the test fails then the failure status is returned to the caller.
static iree_status_t run_test_function(iree_vm_context_t* context,
                                       iree_vm_function_t function,
                                       iree_allocator_t host_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_string_view_t function_name = iree_vm_function_name(&function);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, function_name.data, function_name.size);
  fprintf(stderr, "--- TEST[%.*s] ---\n", (int)function_name.size,
          function_name.data);
  iree_string_view_t function_desc =
      iree_vm_function_lookup_attr_by_name(&function, IREE_SV("description"));
  if (!iree_string_view_is_empty(function_desc)) {
    fprintf(stderr, "%.*s\n", (int)function_desc.size, function_desc.data);
  }
  iree_status_t status = iree_vm_invoke(
      context, function, IREE_VM_INVOCATION_FLAG_NONE, /*policy=*/NULL,
      /*inputs=*/NULL, /*outputs=*/NULL, host_allocator);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Runs all test functions in |test_module|.
static iree_status_t run_all_test_functions(iree_vm_context_t* context,
                                            iree_vm_module_t* test_module,
                                            iree_allocator_t host_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Walk all functions and find the ones we can run (no args, non-internal).
  const iree_vm_module_signature_t module_signature =
      iree_vm_module_signature(test_module);
  for (iree_host_size_t i = 0; i < module_signature.export_function_count;
       ++i) {
    // Get the function and filter to just the public user exports.
    iree_vm_function_t function;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_vm_module_lookup_function_by_ordinal(
                test_module, IREE_VM_FUNCTION_LINKAGE_EXPORT, i, &function));
    bool is_valid = false;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                      check_test_function(function, &is_valid));
    if (is_valid) {
      // Try to run the function and fail on mismatch.
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, run_test_function(context, function, host_allocator));
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Returns OK if there are declared requirements on |module| and they are all
// met and otherwise UNAVAILABLE indicating that the module should not be run.
static iree_status_t check_module_requirements(iree_vm_module_t* module) {
  iree_string_view_t target_features =
      iree_vm_module_lookup_attr_by_name(module, IREE_SV("target_features"));
  while (!iree_string_view_is_empty(target_features)) {
    iree_string_view_t required_feature;
    iree_string_view_split(target_features, ',', &required_feature,
                           &target_features);
    if (iree_string_view_is_empty(required_feature)) continue;
    int64_t feature_is_supported = 0;
    IREE_RETURN_IF_ERROR(
        iree_cpu_lookup_data_by_key(required_feature, &feature_is_supported));
    if (!feature_is_supported) {
      return iree_make_status(
          // The error status matters. We distinguish "feature not supported"
          // which is a normal thing to happen from actual errors.
          IREE_STATUS_UNAVAILABLE,
          "target device does not have the required feature '%.*s'",
          (int)required_feature.size, required_feature.data);
    }
  }
  return iree_ok_status();
}

static iree_status_t load_and_run_e2e_tests(iree_allocator_t host_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_cpu_initialize(host_allocator);

  iree_vm_instance_t* instance = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_tooling_create_instance(host_allocator, &instance));

  iree_tooling_module_list_t module_list;
  iree_tooling_module_list_initialize(&module_list);

  // Create the test module providing helper functions used by test programs.
  iree_vm_module_t* matmul_test_module = NULL;
  iree_status_t status =
      matmul_test_module_create(instance, host_allocator, &matmul_test_module);
  if (iree_status_is_ok(status)) {
    status =
        iree_tooling_module_list_push_back(&module_list, matmul_test_module);
  }
  iree_vm_module_release(matmul_test_module);

  // Load all modules specified by --module= flags.
  if (iree_status_is_ok(status)) {
    status = iree_tooling_load_modules_from_flags(instance, host_allocator,
                                                  &module_list);
  }
  iree_vm_module_t* test_module = iree_tooling_module_list_back(&module_list);

  // Create the context with our support module and all --module= flags.
  iree_vm_context_t* context = NULL;
  iree_hal_device_t* device = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_tooling_create_context_from_flags(
        instance, module_list.count, module_list.values,
        /*default_device_uri=*/iree_string_view_empty(), host_allocator,
        &context, &device, /*out_device_allocator=*/NULL);
  }

  // Ensure the test module is possible to run.
  if (iree_status_is_ok(status)) {
    status = check_module_requirements(test_module);
  }
  iree_tooling_module_list_reset(&module_list);

  // Begin profiling (if enabled).
  if (iree_status_is_ok(status)) {
    status = iree_hal_begin_profiling_from_flags(device);
  }

  // Run all of the tests in the test module.
  if (iree_status_is_ok(status)) {
    status = run_all_test_functions(context, test_module, host_allocator);
  }

  // End profiling (if enabled).
  if (iree_status_is_ok(status)) {
    status = iree_hal_end_profiling_from_flags(device);
  }

  iree_hal_device_release(device);
  iree_vm_context_release(context);
  iree_vm_instance_release(instance);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();

  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (argc != 1) {
    fprintf(stderr, "use --module= flags to specify the modules to run\n");
    IREE_TRACE_APP_EXIT(EXIT_FAILURE);
    return EXIT_FAILURE;
  }

  iree_status_t status = load_and_run_e2e_tests(iree_allocator_system());
  int exit_code = EXIT_SUCCESS;
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    bool is_unavailable = iree_status_is_unavailable(status);
    iree_status_free(status);
    exit_code = is_unavailable ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
