// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
#include "iree/vm/api.h"
#include "iree/vm/native_module_cc.h"
#include "tools/testing/e2e/test_utils.h"

//===----------------------------------------------------------------------===//
// Reference matmul
//===----------------------------------------------------------------------===//

#define REFERENCE_MATMUL(LHSTYPE, RHSTYPE, RESTYPE, ACCTYPE)                   \
  static void reference_matmul_##LHSTYPE##_##RHSTYPE##_##RESTYPE##_##ACCTYPE(  \
      iree_hal_dim_t m_size, iree_hal_dim_t k_size, iree_hal_dim_t n_size,     \
      iree_hal_element_type_t lhs_type, iree_hal_element_type_t rhs_type,      \
      iree_hal_element_type_t acc_type, bool transpose_rhs,                    \
      const LHSTYPE* lhs_data, const RHSTYPE* rhs_data,                        \
      const ACCTYPE* acc_data, RESTYPE* result_data, iree_hal_dim_t m,         \
      iree_hal_dim_t n) {                                                      \
    ACCTYPE acc = acc_data ? acc_data[n + m * n_size] : 0;                     \
    for (iree_hal_dim_t k = 0; k < k_size; ++k) {                              \
      LHSTYPE lhs_value = lhs_data[k + m * k_size];                            \
      RHSTYPE rhs_value =                                                      \
          transpose_rhs ? rhs_data[k + n * k_size] : rhs_data[n + k * n_size]; \
      acc += (ACCTYPE)lhs_value * (ACCTYPE)rhs_value;                          \
    }                                                                          \
    result_data[n + m * n_size] = acc;                                         \
  }

// Reference matmul instantiations from macro REFERENCE_MATMUL
// for the f32 input, f32 accumlation, and f32 result.
// [float <= float * float + float]
REFERENCE_MATMUL(float, float, float, float)

// Reference matmul instantiations from macro REFERENCE_MATMUL
// for the int8_t input, int32_t accumlation, and int32_t result.
// [i32 <= i8 * i8 + i32]
REFERENCE_MATMUL(int8_t, int8_t, int32_t, int32_t)

// Reference matmul instantiations from macro REFERENCE_MATMUL
// for the int32_t input, int32_t accumlation, and int32_t result.
// [i32 <= i32 * i32 + i32]
REFERENCE_MATMUL(int32_t, int32_t, int32_t, int32_t)

// Reference matmul for the f16 input, f16 accumlation, and f16 result.
// [f16 <= f16 * f16 + f16]
static void reference_matmul_f16_f16_f16_f16(
    iree_hal_dim_t m_size, iree_hal_dim_t k_size, iree_hal_dim_t n_size,
    iree_hal_element_type_t lhs_type, iree_hal_element_type_t rhs_type,
    iree_hal_element_type_t acc_type, bool transpose_rhs,
    const uint16_t* lhs_data, const uint16_t* rhs_data,
    const uint16_t* acc_data, uint16_t* result_data, iree_hal_dim_t m,
    iree_hal_dim_t n) {
  float acc = acc_data ? iree_math_f16_to_f32(acc_data[n + m * n_size]) : 0.f;
  for (iree_hal_dim_t k = 0; k < k_size; ++k) {
    int64_t rhs_index = transpose_rhs ? k + n * k_size : n + k * n_size;
    acc += iree_math_f16_to_f32(lhs_data[k + m * k_size]) *
           iree_math_f16_to_f32(rhs_data[rhs_index]);
  }
  result_data[n + m * n_size] = iree_math_f32_to_f16(acc);
}

// Reference matmul for the f16 input, f32 accumlation, and f32 result.
// [f32 <= f16 * f16 + f32]
static void reference_matmul_f16_f16_f32_f32(
    iree_hal_dim_t m_size, iree_hal_dim_t k_size, iree_hal_dim_t n_size,
    iree_hal_element_type_t lhs_type, iree_hal_element_type_t rhs_type,
    iree_hal_element_type_t acc_type, bool transpose_rhs,
    const uint16_t* lhs_data, const uint16_t* rhs_data, const float* acc_data,
    float* result_data, iree_hal_dim_t m, iree_hal_dim_t n) {
  float acc = acc_data ? acc_data[n + m * n_size] : 0.f;
  for (iree_hal_dim_t k = 0; k < k_size; ++k) {
    int64_t rhs_index = transpose_rhs ? k + n * k_size : n + k * n_size;
    acc += iree_math_f16_to_f32(lhs_data[k + m * k_size]) *
           iree_math_f16_to_f32(rhs_data[rhs_index]);
  }
  result_data[n + m * n_size] = acc;
}

// Reference matmul for the bf16 input, bf16 accumlation, and bf16 result.
// [bf16 <= bf16 * bf16 + bf16]
static void reference_matmul_bf16_bf16_bf16_bf16(
    iree_hal_dim_t m_size, iree_hal_dim_t k_size, iree_hal_dim_t n_size,
    iree_hal_element_type_t lhs_type, iree_hal_element_type_t rhs_type,
    iree_hal_element_type_t acc_type, bool transpose_rhs,
    const uint16_t* lhs_data, const uint16_t* rhs_data,
    const uint16_t* acc_data, uint16_t* result_data, iree_hal_dim_t m,
    iree_hal_dim_t n) {
  float acc = acc_data ? iree_math_bf16_to_f32(acc_data[n + m * n_size]) : 0.f;
  for (iree_hal_dim_t k = 0; k < k_size; ++k) {
    int64_t rhs_index = transpose_rhs ? k + n * k_size : n + k * n_size;
    acc += iree_math_bf16_to_f32(lhs_data[k + m * k_size]) *
           iree_math_bf16_to_f32(rhs_data[rhs_index]);
  }
  result_data[n + m * n_size] = iree_math_f32_to_bf16(acc);
}

// Reference matmul for the bf16 input, f32 accumlation, and f32 result.
// [f32 <= bf16 * bf16 + f32]
static void reference_matmul_bf16_bf16_f32_f32(
    iree_hal_dim_t m_size, iree_hal_dim_t k_size, iree_hal_dim_t n_size,
    iree_hal_element_type_t lhs_type, iree_hal_element_type_t rhs_type,
    iree_hal_element_type_t acc_type, bool transpose_rhs,
    const uint16_t* lhs_data, const uint16_t* rhs_data, const float* acc_data,
    float* result_data, iree_hal_dim_t m, iree_hal_dim_t n) {
  float acc = acc_data ? acc_data[n + m * n_size] : 0.f;
  for (iree_hal_dim_t k = 0; k < k_size; ++k) {
    int64_t rhs_index = transpose_rhs ? k + n * k_size : n + k * n_size;
    acc += iree_math_bf16_to_f32(lhs_data[k + m * k_size]) *
           iree_math_bf16_to_f32(rhs_data[rhs_index]);
  }
  result_data[n + m * n_size] = acc;
}

// Computes one element in the result matrix.
static iree_status_t reference_matmul_element(
    iree_hal_dim_t m_size, iree_hal_dim_t k_size, iree_hal_dim_t n_size,
    iree_hal_element_type_t lhs_type, iree_hal_element_type_t rhs_type,
    iree_hal_element_type_t acc_type, bool transpose_rhs, void* lhs_data,
    void* rhs_data, void* acc_data, void* result_data, iree_hal_dim_t m,
    iree_hal_dim_t n) {
  if (lhs_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32 &&
      rhs_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32 &&
      acc_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32) {
    reference_matmul_float_float_float_float(
        m_size, k_size, n_size, lhs_type, rhs_type, acc_type, transpose_rhs,
        (const float*)lhs_data, (const float*)rhs_data, (const float*)acc_data,
        (float*)result_data, m, n);
  } else if (iree_hal_element_type_is_integer(lhs_type, 8) &&
             iree_hal_element_type_is_integer(rhs_type, 8) &&
             iree_hal_element_type_is_integer(acc_type, 32)) {
    reference_matmul_int8_t_int8_t_int32_t_int32_t(
        m_size, k_size, n_size, lhs_type, rhs_type, acc_type, transpose_rhs,
        (const int8_t*)lhs_data, (const int8_t*)rhs_data,
        (const int32_t*)acc_data, (int32_t*)result_data, m, n);
  } else if (iree_hal_element_type_is_integer(lhs_type, 32) &&
             iree_hal_element_type_is_integer(rhs_type, 32) &&
             iree_hal_element_type_is_integer(acc_type, 32)) {
    reference_matmul_int32_t_int32_t_int32_t_int32_t(
        m_size, k_size, n_size, lhs_type, rhs_type, acc_type, transpose_rhs,
        (const int32_t*)lhs_data, (const int32_t*)rhs_data,
        (const int32_t*)acc_data, (int32_t*)result_data, m, n);
  } else if (lhs_type == IREE_HAL_ELEMENT_TYPE_FLOAT_16 &&
             rhs_type == IREE_HAL_ELEMENT_TYPE_FLOAT_16 &&
             acc_type == IREE_HAL_ELEMENT_TYPE_FLOAT_16) {
    reference_matmul_f16_f16_f16_f16(
        m_size, k_size, n_size, lhs_type, rhs_type, acc_type, transpose_rhs,
        (const uint16_t*)lhs_data, (const uint16_t*)rhs_data,
        (const uint16_t*)acc_data, (uint16_t*)result_data, m, n);
  } else if (lhs_type == IREE_HAL_ELEMENT_TYPE_FLOAT_16 &&
             rhs_type == IREE_HAL_ELEMENT_TYPE_FLOAT_16 &&
             acc_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32) {
    reference_matmul_f16_f16_f32_f32(
        m_size, k_size, n_size, lhs_type, rhs_type, acc_type, transpose_rhs,
        (const uint16_t*)lhs_data, (const uint16_t*)rhs_data,
        (const float*)acc_data, (float*)result_data, m, n);
  } else if (lhs_type == IREE_HAL_ELEMENT_TYPE_BFLOAT_16 &&
             rhs_type == IREE_HAL_ELEMENT_TYPE_BFLOAT_16 &&
             acc_type == IREE_HAL_ELEMENT_TYPE_BFLOAT_16) {
    reference_matmul_bf16_bf16_bf16_bf16(
        m_size, k_size, n_size, lhs_type, rhs_type, acc_type, transpose_rhs,
        (const uint16_t*)lhs_data, (const uint16_t*)rhs_data,
        (const uint16_t*)acc_data, (uint16_t*)result_data, m, n);
  } else if (lhs_type == IREE_HAL_ELEMENT_TYPE_BFLOAT_16 &&
             rhs_type == IREE_HAL_ELEMENT_TYPE_BFLOAT_16 &&
             acc_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32) {
    reference_matmul_bf16_bf16_f32_f32(
        m_size, k_size, n_size, lhs_type, rhs_type, acc_type, transpose_rhs,
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
    iree_hal_dim_t m_repeat_period, iree_hal_dim_t n_repeat_period,
    iree_hal_element_type_t lhs_type, iree_hal_element_type_t rhs_type,
    iree_hal_element_type_t acc_type, bool transpose_rhs,
    iree_byte_span_t lhs_contents, iree_byte_span_t rhs_contents,
    iree_byte_span_t acc_contents, iree_byte_span_t result_contents,
    int compute_every) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, m_size);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, k_size);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, n_size);

  // This function computes:
  //
  // C = matmul(A, B) + D
  //
  // where the correspond to function arguments is:
  //        C - result_contents.
  //        A - lhs_contents.
  //        B - rhs_contents.
  //        D - acc_contents.
  //
  // Because the rows of A and the columns of B repeat,
  //
  // C[m0, n0] == C[m1, n1] if
  //  - m0 % m_repeat_period == m1 % m_repeat_period, and
  //  - n0 % n_repeat_period == n1 % n_repeat_period.
  //
  // This cyclical repetition of the values in C can be used to accelerate the
  // computation of C: By computing C[i, j] for all
  //   i <= m_repeat_period, and
  //   j <= n_repeat_period,
  //
  // all remaining values of C can be copied from these low-index values,
  // avoiding computing the O(k) inner-product and instead performing an O(1)
  // copy.
  //
  // Using the cyclical caching trick to compute only some of the results when
  // 'compute_every > 1' is too complicated, unless an additional buffer is used
  // to recored which elements have previously been computed (which we don't
  // want to do). We therefore use only one of the two techniques to reduce
  // computation.

  // Repeat period of 0 denotes no repeat.
  m_repeat_period = m_repeat_period > 0 ? m_repeat_period : m_size;
  n_repeat_period = n_repeat_period > 0 ? n_repeat_period : n_size;

  // The number of O(k) inner-products that will be computed using the caching
  // trick. Only inner-products for
  // (i,j) in {0, ..., m_repeat_period-1} x {0, ..., n_repeat_period-1}
  // computed.
  const int64_t caching_trick_cost = m_repeat_period * n_repeat_period;

  // The number of O(k) inner-products that will be computed by computing only
  // every compute_every-th element.
  const int64_t compute_every_cost = m_size * n_size / compute_every;

  // True if the repeat period in both the m- and n- dimensions are too large to
  // result in any caching.
  const bool no_cached_values =
      m_repeat_period >= m_size && n_repeat_period >= n_size;

  // Even if compute_every > 1, we use the caching trick to compute every value
  // if it is cheaper to do so.
  if (compute_every > 1 && compute_every_cost < caching_trick_cost) {
    iree_host_size_t count = 0;
    for (iree_hal_dim_t m = 0; m < m_size; ++m) {
      for (iree_hal_dim_t n = 0; n < n_size; ++n) {
        if (++count < compute_every) continue;
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, reference_matmul_element(
                    m_size, k_size, n_size, lhs_type, rhs_type, acc_type,
                    transpose_rhs, lhs_contents.data, rhs_contents.data,
                    acc_contents.data, result_contents.data, m, n));
      }
    }
  }

  else if (no_cached_values) {
    for (iree_hal_dim_t m = 0; m < m_size; ++m) {
      for (iree_hal_dim_t n = 0; n < n_size; ++n) {
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, reference_matmul_element(
                    m_size, k_size, n_size, lhs_type, rhs_type, acc_type,
                    transpose_rhs, lhs_contents.data, rhs_contents.data,
                    acc_contents.data, result_contents.data, m, n));
      }
    }
  }

  else {
    iree_host_size_t bytes_per_element =
        iree_hal_element_dense_byte_count(acc_type);
    for (iree_hal_dim_t m = 0; m < m_size; ++m) {
      for (iree_hal_dim_t n = 0; n < n_size; ++n) {
        const iree_hal_dim_t m_in_cache_zone = m % m_repeat_period;
        const iree_hal_dim_t n_in_cache_zone = n % n_repeat_period;
        const iree_hal_dim_t result_offset =
            bytes_per_element * (n + m * n_size);
        const iree_hal_dim_t cache_offset =
            bytes_per_element * (n_in_cache_zone + m_in_cache_zone * n_size);
        uint8_t* result_base = result_contents.data;

        const bool isCached = m != m_in_cache_zone || n != n_in_cache_zone;
        if (isCached) {
          memcpy(result_base + result_offset, result_base + cache_offset,
                 bytes_per_element);
        } else {
          IREE_RETURN_AND_END_ZONE_IF_ERROR(
              z0, reference_matmul_element(
                      m_size, k_size, n_size, lhs_type, rhs_type, acc_type,
                      transpose_rhs, lhs_contents.data, rhs_contents.data,
                      acc_contents.data, result_base, m, n));
        }
      }
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
  iree_hal_dim_t m_repeat_period;
  iree_hal_dim_t n_repeat_period;
  iree_hal_element_type_t lhs_type;
  iree_hal_element_type_t rhs_type;
  iree_hal_element_type_t acc_type;
  iree_hal_element_type_t result_type;
  bool transpose_rhs;
  iree_byte_span_t lhs_contents;
  iree_byte_span_t rhs_contents;
  iree_byte_span_t acc_contents;
  iree_byte_span_t actual_contents;
  iree_byte_span_t expected_contents;
} matmul_results_t;

static void matmul_results_deinitialize(matmul_results_t* results);

static iree_status_t matmul_results_initialize(
    iree_hal_device_t* device, iree_hal_dim_t m_size, iree_hal_dim_t k_size,
    iree_hal_dim_t n_size, iree_hal_dim_t m_repeat_period,
    iree_hal_dim_t n_repeat_period, uint32_t transpose_rhs,
    iree_hal_buffer_view_t* lhs, iree_hal_buffer_view_t* rhs,
    iree_hal_buffer_view_t* acc, iree_hal_buffer_view_t* result,
    iree_allocator_t host_allocator, matmul_results_t* out_results) {
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_results, 0, sizeof(*out_results));
  out_results->host_allocator = host_allocator;

  out_results->m = m_size;
  out_results->k = k_size;
  out_results->n = n_size;

  out_results->m_repeat_period = m_repeat_period;
  out_results->n_repeat_period = n_repeat_period;

  out_results->lhs_type = iree_hal_buffer_view_element_type(lhs);
  out_results->rhs_type = iree_hal_buffer_view_element_type(rhs);
  out_results->acc_type = iree_hal_buffer_view_element_type(result);
  out_results->result_type = iree_hal_buffer_view_element_type(result);

  out_results->transpose_rhs = transpose_rhs != 0;

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
      iree_hal_dim_t idx = col + row * cols;
      iree_test_utils_e2e_value_t elem =
          iree_test_utils_read_buffer_element(idx, element_type, matrix);
      // NOTE: iree_max is a macro and may evaluate its args twice.
      char buf[64];
      int this_elem_width =
          iree_test_utils_snprintf_value(buf, sizeof(buf), elem, precision);
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
      iree_hal_dim_t idx = col + row * cols;
      iree_test_utils_e2e_value_t element =
          iree_test_utils_read_buffer_element(idx, element_type, matrix);
      bool disagree = false;
      if (other_matrix) {
        iree_test_utils_e2e_value_t other_element =
            iree_test_utils_read_buffer_element(idx, element_type,
                                                other_matrix);
        disagree =
            !iree_test_utils_result_elements_agree(element, other_element);
      }
      char buf[64];
      iree_test_utils_snprintf_value(buf, sizeof(buf), element, precision);
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
static iree_status_t check_matmul_failure(
    FILE* file, const matmul_results_t* results,
    iree_test_utils_e2e_value_t actual_value,
    iree_test_utils_e2e_value_t expected_value, iree_hal_dim_t row,
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
  iree_test_utils_snprintf_value(actual_value_buf, sizeof(actual_value_buf),
                                 actual_value, PRECISION_HIGH);
  iree_test_utils_snprintf_value(expected_value_buf, sizeof(expected_value_buf),
                                 expected_value, PRECISION_HIGH);
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
               iree_test_utils_emoji(true));
  fprintf(file, "\n");
  print_matrix(file, "actual result", PRECISION_LOW, results->m, m_start, m_end,
               results->n, n_start, n_end, results->result_type,
               results->actual_contents.data, results->expected_contents.data,
               iree_test_utils_emoji(false));
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
      z0, reference_matmul(
              results->m, results->k, results->n, results->m_repeat_period,
              results->n_repeat_period, results->lhs_type, results->rhs_type,
              results->acc_type, results->transpose_rhs, results->lhs_contents,
              results->rhs_contents, results->acc_contents,
              results->expected_contents, check_every));

  int count = 0;

  for (iree_hal_dim_t m = 0; m < results->m; ++m) {
    for (iree_hal_dim_t n = 0; n < results->n; ++n) {
      if (++count < check_every) continue;
      count = 0;
      iree_hal_dim_t idx = m * results->n + n;

      iree_test_utils_e2e_value_t actual_value =
          iree_test_utils_read_buffer_element(idx, results->result_type,
                                              results->actual_contents.data);

      iree_test_utils_e2e_value_t expected_value =
          iree_test_utils_read_buffer_element(idx, results->result_type,
                                              results->expected_contents.data);

      if (!iree_test_utils_result_elements_agree(actual_value,
                                                 expected_value)) {
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
  int check_every = iree_test_utils_calculate_check_every(
      results->m * results->n, results->n);

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
// `matmul_test` custom module
//===----------------------------------------------------------------------===//
// This uses the C++ wrapper to keep things simple. Though easier to use it's
// got additional overhead/code-size bloat that doesn't matter in a test like
// this. Making a C module builder API that removes the boilerplate there is TBD
// so this file is written in C besides this module so that we can swap it back
// to being pure C in the future.

namespace iree {

class MatmulTestModuleState final {
 public:
  explicit MatmulTestModuleState(iree_allocator_t host_allocator)
      : host_allocator_(host_allocator) {}
  ~MatmulTestModuleState() = default;

  StatusOr<vm::ref<iree_hal_buffer_view_t>> GenerateRandomMatrix(
      const vm::ref<iree_hal_device_t> device, int64_t dim0, int64_t dim1,
      iree_hal_element_type_t element_type, int32_t seed) {
    // Generate a random matrix without any row repetition or column repetition.
    const int64_t dim0_repeat_period = 0;
    const int64_t dim1_repeat_period = 0;
    return GenerateRandomCyclicalMatrix(device, dim0, dim1, dim0_repeat_period,
                                        dim1_repeat_period, element_type, seed);
  }

  // Fills the destination span with pseudorandom values of the given
  // |element_type|. The given |seed| is passed to the pseudorandom generator.
  // The pseudorandom values are reproducible both across runs and across
  // machines.
  StatusOr<vm::ref<iree_hal_buffer_view_t>> GenerateRandomCyclicalMatrix(
      const vm::ref<iree_hal_device_t> device, int64_t dim0, int64_t dim1,
      int64_t dim0_repeat_period, int64_t dim1_repeat_period,
      iree_hal_element_type_t element_type, int32_t seed) {
    iree_hal_dim_t dims[2] = {
        (iree_hal_dim_t)dim0,
        (iree_hal_dim_t)dim1,
    };

    // Repeat period of 0 denotes no repeat.
    dim0_repeat_period = dim0_repeat_period <= 0 ? dim0 : dim0_repeat_period;
    dim1_repeat_period = dim1_repeat_period <= 0 ? dim1 : dim1_repeat_period;

    iree_hal_buffer_params_t buffer_params = {0};
    buffer_params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
    buffer_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    buffer_params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
    vm::ref<iree_hal_buffer_view_t> result_view;
    struct callback_state_t {
      iree_hal_element_type_t element_type;
      int32_t seed;
      int64_t dim0;
      int64_t dim1;
      int64_t dim0_repeat_period;
      int64_t dim1_repeat_period;
    } callback_state = {
        element_type, seed, dim0, dim1, dim0_repeat_period, dim1_repeat_period,
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
          iree_test_utils_get_min_max_for_element_type(
              callback_state.element_type, &min, &max);
          uint32_t range = (max - min + 1);
          iree_host_size_t element_byte_count =
              iree_hal_element_dense_byte_count(callback_state.element_type);
          uint32_t state = callback_state.seed;
          uint8_t* data = span.data;
          int64_t dim0 = callback_state.dim0;
          int64_t dim1 = callback_state.dim1;
          int64_t dim0_repeat_period = callback_state.dim0_repeat_period;
          int64_t dim1_repeat_period = callback_state.dim1_repeat_period;
          iree_hal_element_type_t element_type = callback_state.element_type;

          for (int64_t i = 0; i < dim0; i++) {
            for (int64_t j = 0; j < dim1; ++j) {
              uint64_t distance_to_previous = 0;

              // The row of the new element is greater than the repeat period
              // for rows, so get the value from a previous row rather than
              // generating a new value.
              if (i >= dim0_repeat_period) {
                distance_to_previous =
                    dim0_repeat_period * dim1 * element_byte_count;
              }

              // The column of the new element is greater than the repeat period
              // for columns, so get the value from a previous column rather
              // than generating a new value.
              else if (j >= dim1_repeat_period) {
                distance_to_previous = dim1_repeat_period * element_byte_count;
              }

              if (distance_to_previous) {
                uint64_t n_bytes =
                    iree_hal_element_dense_byte_count(element_type);
                memcpy(data, data - distance_to_previous, n_bytes);
              } else {
                int32_t nextValue =
                    (int32_t)iree_test_utils_pseudorandom_range(&state, range) +
                    min;
                iree_test_utils_write_element(element_type, nextValue, data);
              }

              data += element_byte_count;
            }
          }
          return iree_ok_status();
        },
        &callback_state, &result_view));
    return std::move(result_view);
  }

  Status CheckCyclicalMatmulResults(
      const vm::ref<iree_hal_device_t> device, int64_t m, int64_t k, int64_t n,
      int64_t m_repeat_period, int64_t n_repeat_period, int32_t transpose_rhs,
      const vm::ref<iree_hal_buffer_view_t> lhs,
      const vm::ref<iree_hal_buffer_view_t> rhs,
      const vm::ref<iree_hal_buffer_view_t> acc,
      const vm::ref<iree_hal_buffer_view_t> actual_result) {
    matmul_results_t results = {};
    IREE_RETURN_IF_ERROR(matmul_results_initialize(
        device.get(), (iree_hal_dim_t)m, (iree_hal_dim_t)k, (iree_hal_dim_t)n,
        (iree_hal_dim_t)m_repeat_period, (iree_hal_dim_t)n_repeat_period,
        transpose_rhs, lhs.get(), rhs.get(), acc.get(), actual_result.get(),
        host_allocator_, &results));
    iree_status_t status = check_matmul_results(stderr, &results);
    matmul_results_deinitialize(&results);
    return status;
  }

  Status CheckMatmulResults(
      const vm::ref<iree_hal_device_t> device, int64_t m, int64_t k, int64_t n,
      int32_t transpose_rhs, const vm::ref<iree_hal_buffer_view_t> lhs,
      const vm::ref<iree_hal_buffer_view_t> rhs,
      const vm::ref<iree_hal_buffer_view_t> acc,
      const vm::ref<iree_hal_buffer_view_t> actual_result,
      const vm::ref<iree_hal_buffer_view_t> expected_result) {
    // A repeat period of 0 denotes no repeat.
    int64_t m_repeat_period = 0;
    int64_t n_repeat_period = 0;
    return CheckCyclicalMatmulResults(device, m, k, n, m_repeat_period,
                                      n_repeat_period, transpose_rhs, lhs, rhs,
                                      acc, actual_result);
  }

 private:
  iree_allocator_t host_allocator_;
};

static const vm::NativeFunction<MatmulTestModuleState>
    kMatmulTestModuleFunctions[] = {
        vm::MakeNativeFunction("generate_random_matrix",
                               &MatmulTestModuleState::GenerateRandomMatrix),
        vm::MakeNativeFunction(
            "generate_random_cyclical_matrix",
            &MatmulTestModuleState::GenerateRandomCyclicalMatrix),
        vm::MakeNativeFunction(
            "check_cyclical_matmul_results",
            &MatmulTestModuleState::CheckCyclicalMatmulResults),
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

}  // namespace iree

static iree_status_t matmul_test_module_create(iree_vm_instance_t* instance,
                                               iree_allocator_t host_allocator,
                                               iree_vm_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;
  auto module = std::make_unique<iree::MatmulTestModule>(
      "matmul_test", /*version=*/0, instance, host_allocator,
      iree::span<const iree::vm::NativeFunction<iree::MatmulTestModuleState>>(
          iree::kMatmulTestModuleFunctions));
  *out_module = module.release()->interface();
  return iree_ok_status();
}

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();

  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (argc != 1) {
    fprintf(stderr, "use --module= flags to specify the modules to run\n");
    IREE_TRACE_APP_EXIT(EXIT_FAILURE);
    return EXIT_FAILURE;
  }

  iree_status_t status = iree_test_utils_load_and_run_e2e_tests(
      iree_allocator_system(), matmul_test_module_create);
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
