// Copyright 2021 The IREE Authors
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
#include "iree/tooling/device_util.h"
#include "iree/tooling/trace_replay.h"
#include "iree/tooling/vm_util.h"
#include "iree/tooling/yaml_util.h"
#include "iree/vm/api.h"

IREE_FLAG(bool, require_exact_results, true,
          "Requires floating point result elements to match exactly.");
IREE_FLAG(
    float, acceptable_fp_delta, 1e-5,
    "Maximum absolute difference allowed with inexact floating point results.");
IREE_FLAG(
    int32_t, max_elements_to_check, 10000,
    "Maximum number of matrix elements to check for each matmul. For larger "
    "matrices, only every n-th element will be checked for some n chosed to "
    "stay just under that threshold and to avoid being a divisor of the inner "
    "dimension size to avoid special patterns. As the check uses a slow "
    "reference implementation, this is a trade-off between test latency and "
    "coverage. The value 0 means check all elements.");

IREE_FLAG(bool, trace_execution, false, "Traces VM execution to stderr.");

static const char* emoji(bool good) { return good ? "🦄" : "🐞"; }

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

/*****************************************************************************
 *
 * Part 1:
 *
 * Generic helper functions to deal with buffer_view's.
 *
 *****************************************************************************/

// Get list[i] as a buffer_view.
static iree_status_t get_item_as_buffer_view(
    iree_vm_list_t* list, iree_host_size_t i,
    iree_hal_buffer_view_t** out_value) {
  iree_vm_variant_t variant = iree_vm_variant_empty();
  IREE_RETURN_IF_ERROR(iree_vm_list_get_variant_assign(list, i, &variant));
  if (!iree_vm_variant_is_ref(variant)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected list item %" PRIhsz " to be a ref", i);
  }
  return iree_hal_buffer_view_check_deref(variant.ref, out_value);
}

// Validates that |buffer_view|'s memory type satisfies |expected|.
static iree_status_t validate_memory_type(iree_hal_buffer_view_t* buffer_view,
                                          iree_hal_memory_type_t expected) {
  return iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(iree_hal_buffer_view_buffer(buffer_view)),
      expected);
}

// Map dense row-major data in a host-local buffer_view.
static iree_status_t map_host_local_row_major_data(
    iree_hal_buffer_view_t* buffer_view,
    enum iree_hal_memory_access_bits_t access,
    iree_hal_buffer_mapping_t* mapping) {
  IREE_RETURN_IF_ERROR(
      validate_memory_type(buffer_view, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
  if (iree_hal_buffer_view_encoding_type(buffer_view) !=
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer_view is not dense row major");
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      iree_hal_buffer_view_buffer(buffer_view), IREE_HAL_MAPPING_MODE_SCOPED,
      access, 0, IREE_WHOLE_BUFFER, mapping));
  return iree_ok_status();
}

// Allocates host-local |dst| to have the same shape as |src|.
// Implicitly zero-filled.
static iree_status_t allocate_host_buffer_view_like(
    iree_hal_device_t* device, iree_hal_allocator_t* hal_allocator,
    iree_hal_buffer_view_t* src, iree_hal_buffer_view_t** dst) {
  return iree_hal_buffer_view_allocate_buffer_copy(
      device, hal_allocator, iree_hal_buffer_view_shape_rank(src),
      iree_hal_buffer_view_shape_dims(src),
      iree_hal_buffer_view_element_type(src),
      iree_hal_buffer_view_encoding_type(src),
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL,
          .usage =
              IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING,
      },
      iree_const_byte_span_empty(), dst);
}

// Allocates device-local |dst| to have the same shape as |src|.
// Implicitly zero-filled.
static iree_status_t allocate_device_buffer_view_like(
    iree_hal_device_t* device, iree_hal_allocator_t* hal_allocator,
    iree_hal_buffer_view_t* src, iree_const_byte_span_t initial_data,
    iree_hal_buffer_view_t** dst) {
  return iree_hal_buffer_view_allocate_buffer_copy(
      device, hal_allocator, iree_hal_buffer_view_shape_rank(src),
      iree_hal_buffer_view_shape_dims(src),
      iree_hal_buffer_view_element_type(src),
      iree_hal_buffer_view_encoding_type(src),
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      },
      initial_data, dst);
}

// Performs a deep copy of device-local |src| into host-local |dst|.
// Allocates |dst|.
static iree_status_t copy_device_buffer_view_to_host(
    iree_hal_device_t* device, iree_hal_allocator_t* hal_allocator,
    iree_hal_buffer_view_t* src, iree_hal_buffer_view_t** dst) {
  IREE_RETURN_IF_ERROR(
      validate_memory_type(src, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL));
  IREE_RETURN_IF_ERROR(
      allocate_host_buffer_view_like(device, hal_allocator, src, dst));
  iree_hal_buffer_mapping_t dst_mapping;
  iree_status_t status = map_host_local_row_major_data(
      *dst, IREE_HAL_MEMORY_ACCESS_WRITE, &dst_mapping);
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_transfer_d2h(
        device, iree_hal_buffer_view_buffer(src), 0, dst_mapping.contents.data,
        iree_hal_buffer_view_byte_length(src),
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_unmap_range(&dst_mapping));
  return status;
}

// Performs a deep copy of device-local |src| into a device-local |dst|.
// Allocates |dst|.
static iree_status_t copy_device_buffer_view_to_device(
    iree_hal_device_t* device, iree_hal_allocator_t* hal_allocator,
    iree_hal_buffer_view_t* src, iree_hal_buffer_view_t** dst) {
  IREE_RETURN_IF_ERROR(
      validate_memory_type(src, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL));
  IREE_RETURN_IF_ERROR(allocate_device_buffer_view_like(
      device, hal_allocator, src, iree_const_byte_span_empty(), dst));
  iree_status_t status = iree_hal_device_transfer_d2d(
      device, iree_hal_buffer_view_buffer(src), 0,
      iree_hal_buffer_view_buffer(*dst), 0,
      iree_hal_buffer_view_byte_length(src),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_view_release(*dst);
  }
  return status;
}

// Deep-copy the list of device-local buffer-views |src| into |dst|.
// Allocates |dst|.
static iree_status_t copy_device_buffer_views_to_host(
    iree_hal_device_t* device, iree_hal_allocator_t* hal_allocator,
    iree_vm_list_t* src, iree_vm_list_t** dst) {
  iree_vm_type_def_t elem_type = iree_vm_list_element_type(src);
  iree_host_size_t size = iree_vm_list_size(src);
  iree_allocator_t allocator = iree_hal_allocator_host_allocator(hal_allocator);
  IREE_RETURN_IF_ERROR(iree_vm_list_create(elem_type, size, allocator, dst));
  IREE_RETURN_IF_ERROR(iree_vm_list_resize(*dst, size));
  for (iree_host_size_t i = 0; i < size; ++i) {
    iree_hal_buffer_view_t* src_elem = NULL;
    IREE_RETURN_IF_ERROR(get_item_as_buffer_view(src, i, &src_elem));
    iree_hal_buffer_view_t* dst_elem = NULL;
    IREE_RETURN_IF_ERROR(copy_device_buffer_view_to_host(device, hal_allocator,
                                                         src_elem, &dst_elem));
    iree_vm_ref_t dst_elem_ref = {0};
    IREE_RETURN_IF_ERROR(iree_vm_ref_wrap_assign(
        dst_elem, iree_hal_buffer_view_type(), &dst_elem_ref));
    IREE_RETURN_IF_ERROR(iree_vm_list_set_ref_move(*dst, i, &dst_elem_ref));
  }
  return iree_ok_status();
}

/*****************************************************************************
 *
 * Part 2:
 *
 * Helper functions to deal with matrices and matrix multiplications.
 *
 * Much of this is the |reference_matmul| function, a reference implementation
 * of matrix multiplication on host-mapped buffers, and helpers for it.
 *
 * Still generic in the sense that none of the high-level logic of this
 * particular test program is entrenched here.
 *
 *****************************************************************************/

// Reads an element from a mapped row-major matrix buffer.
static iree_e2e_test_value_t read_matrix_element(
    iree_hal_dim_t m_size, iree_hal_dim_t n_size,
    iree_hal_element_type_t result_type, void* data, iree_hal_dim_t m,
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

// Get the {m,k,n}_size values of the shape of a matmul
static iree_status_t get_matmul_sizes(
    iree_hal_buffer_view_t* lhs, iree_hal_buffer_view_t* rhs,
    iree_hal_buffer_view_t* acc, iree_hal_buffer_view_t* result,
    iree_hal_dim_t* m_size, iree_hal_dim_t* k_size, iree_hal_dim_t* n_size) {
  iree_hal_dim_t lhs_dims[2] = {0};
  iree_hal_dim_t rhs_dims[2] = {0};
  iree_hal_dim_t acc_dims[2] = {0};
  iree_hal_dim_t result_dims[2] = {0};
  IREE_RETURN_IF_ERROR(get_matrix_shape(lhs, lhs_dims));
  IREE_RETURN_IF_ERROR(get_matrix_shape(rhs, rhs_dims));
  IREE_RETURN_IF_ERROR(get_matrix_shape(result, result_dims));
  *m_size = lhs_dims[0];
  *k_size = lhs_dims[1];
  *n_size = rhs_dims[1];
  if (acc) {
    IREE_RETURN_IF_ERROR(get_matrix_shape(acc, acc_dims));
    if (!(lhs_dims[0] == *m_size && lhs_dims[1] == *k_size &&
          rhs_dims[0] == *k_size && rhs_dims[1] == *n_size &&
          acc_dims[0] == *m_size && acc_dims[1] == *n_size &&
          result_dims[0] == *m_size && result_dims[1] == *n_size)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "mismatched matrix shapes in matmul: %" PRIdim "x%" PRIdim
          " * %" PRIdim "x%" PRIdim " + %" PRIdim "x%" PRIdim " -> %" PRIdim
          "x%" PRIdim,
          lhs_dims[0], lhs_dims[1], rhs_dims[0], rhs_dims[1], acc_dims[0],
          acc_dims[1], result_dims[0], result_dims[1]);
    }
  } else {
    if (!(lhs_dims[0] == *m_size && lhs_dims[1] == *k_size &&
          rhs_dims[0] == *k_size && rhs_dims[1] == *n_size &&
          result_dims[0] == *m_size && result_dims[1] == *n_size)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "mismatched matrix shapes in matmul: %" PRIdim
                              "x%" PRIdim " * %" PRIdim "x%" PRIdim
                              " -> %" PRIdim "x%" PRIdim,
                              lhs_dims[0], lhs_dims[1], rhs_dims[0],
                              rhs_dims[1], result_dims[0], result_dims[1]);
    }
  }
  return iree_ok_status();
}

#define IREE_TRACE_REPLAY_REFERENCE_MATMUL(LHSTYPE, RHSTYPE, RESTYPE, ACCTYPE) \
  static void reference_matmul_##LHSTYPE##_##RHSTYPE##_##RESTYPE##_##ACCTYPE(  \
      iree_hal_dim_t m_size, iree_hal_dim_t k_size, iree_hal_dim_t n_size,     \
      iree_hal_element_type_t lhs_type, iree_hal_element_type_t rhs_type,      \
      iree_hal_element_type_t acc_type, const LHSTYPE* lhs_data,               \
      const RHSTYPE* rhs_data, const ACCTYPE* acc_data, RESTYPE* result_data,  \
      iree_hal_dim_t m, iree_hal_dim_t n) {                                    \
    ACCTYPE acc = acc_data ? acc_data[n + m * n_size] : 0;                     \
    for (iree_hal_dim_t k = 0; k < k_size; ++k) {                              \
      LHSTYPE lhs_value = lhs_data[k + m * k_size];                            \
      RHSTYPE rhs_value = rhs_data[n + k * n_size];                            \
      acc += (ACCTYPE)lhs_value * (ACCTYPE)rhs_value;                          \
    }                                                                          \
    result_data[n + m * n_size] = acc;                                         \
  }

// Reference mamtul instantiations from macro IREE_TRACE_REPLAY_REFERENCE_MATMUL
// for the f32 input, f32 accumlation, and f32 result.
// [float <= float * float + float]
IREE_TRACE_REPLAY_REFERENCE_MATMUL(float, float, float, float)

// Reference mamtul instantiations from macro IREE_TRACE_REPLAY_REFERENCE_MATMUL
// for the int8_t input, int32_t accumlation, and int32_t result.
// [i32 <= i8 * i8 + i32]
IREE_TRACE_REPLAY_REFERENCE_MATMUL(int8_t, int8_t, int32_t, int32_t)

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
    acc = iree_math_round_to_nearest_f16(
        iree_math_round_to_nearest_f16(
            (iree_math_f16_to_f32(lhs_data[k + m * k_size]) *
             iree_math_f16_to_f32(rhs_data[n + k * n_size]))) +
        acc);
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
    acc = iree_math_round_to_nearest_bf16(
        iree_math_round_to_nearest_bf16(
            (iree_math_bf16_to_f32(lhs_data[k + m * k_size]) *
             iree_math_bf16_to_f32(rhs_data[n + k * n_size]))) +
        acc);
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
static void reference_matmul_element(
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
    iree_status_abort(
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "unhandled combination of element types in matmul"));
  }
}

// Reference matmul implementation, used to compare matmul results against.
static iree_status_t reference_matmul(iree_vm_list_t* input_list,
                                      iree_hal_buffer_view_t* result,
                                      int compute_every) {
  iree_hal_buffer_view_t* lhs = NULL;
  iree_hal_buffer_view_t* rhs = NULL;
  iree_hal_buffer_view_t* acc = NULL;
  IREE_RETURN_IF_ERROR(get_item_as_buffer_view(input_list, 0, &lhs));
  IREE_RETURN_IF_ERROR(get_item_as_buffer_view(input_list, 1, &rhs));
  if (iree_vm_list_size(input_list) == 3) {
    IREE_RETURN_IF_ERROR(get_item_as_buffer_view(input_list, 2, &acc));
  }
  iree_hal_dim_t m_size, k_size, n_size;
  IREE_RETURN_IF_ERROR(
      get_matmul_sizes(lhs, rhs, acc, result, &m_size, &k_size, &n_size));
  iree_hal_buffer_mapping_t lhs_mapping;
  iree_hal_buffer_mapping_t rhs_mapping;
  iree_hal_buffer_mapping_t acc_mapping;
  iree_hal_buffer_mapping_t result_mapping;
  IREE_RETURN_IF_ERROR(map_host_local_row_major_data(
      lhs, IREE_HAL_MEMORY_ACCESS_READ, &lhs_mapping));
  IREE_RETURN_IF_ERROR(map_host_local_row_major_data(
      rhs, IREE_HAL_MEMORY_ACCESS_READ, &rhs_mapping));
  if (acc) {
    IREE_RETURN_IF_ERROR(map_host_local_row_major_data(
        acc, IREE_HAL_MEMORY_ACCESS_READ, &acc_mapping));
  }
  IREE_RETURN_IF_ERROR(map_host_local_row_major_data(
      result, IREE_HAL_MEMORY_ACCESS_WRITE, &result_mapping));
  iree_hal_element_type_t lhs_type = iree_hal_buffer_view_element_type(lhs);
  iree_hal_element_type_t rhs_type = iree_hal_buffer_view_element_type(rhs);
  iree_hal_element_type_t acc_type = iree_hal_buffer_view_element_type(result);
  int count = 0;
  for (iree_hal_dim_t m = 0; m < m_size; ++m) {
    for (iree_hal_dim_t n = 0; n < n_size; ++n) {
      if (++count < compute_every) continue;
      count = 0;
      reference_matmul_element(m_size, k_size, n_size, lhs_type, rhs_type,
                               acc_type, lhs_mapping.contents.data,
                               rhs_mapping.contents.data,
                               acc ? acc_mapping.contents.data : NULL,
                               result_mapping.contents.data, m, n);
    }
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_unmap_range(&lhs_mapping));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_unmap_range(&rhs_mapping));
  if (acc) {
    IREE_RETURN_IF_ERROR(iree_hal_buffer_unmap_range(&acc_mapping));
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_unmap_range(&result_mapping));
  return iree_ok_status();
}

/*****************************************************************************
 *
 * Part 3:
 *
 * Helper functions to validate matmul test results and pretty-print matrices.
 *
 * The only entry point in to this part is |check_matmul_results|, the other
 * functions are only helpers for it.
 *
 * Still generic in the sense that none of the high-level logic of this
 * particular test program is entrenched here.
 *
 *****************************************************************************/

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
      fprintf(stderr, "actual (%x) %g ; expected (%x) %g\n", actual.bf16_u16,
              iree_math_bf16_to_f32(actual.bf16_u16), expected.bf16_u16,
              iree_math_bf16_to_f32(expected.bf16_u16));
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
static int get_max_elem_width(precision_t precision, int row_start, int row_end,
                              int col_start, int col_end,
                              iree_hal_buffer_view_t* matrix) {
  iree_hal_dim_t dims[2] = {0};
  get_matrix_shape(matrix, dims);
  int rows = dims[0];
  int cols = dims[1];
  iree_hal_element_type_t elem_type = iree_hal_buffer_view_element_type(matrix);
  iree_hal_buffer_mapping_t mapping;
  IREE_CHECK_OK(map_host_local_row_major_data(
      matrix, IREE_HAL_MEMORY_ACCESS_READ, &mapping));
  int max_elem_width = 0;
  for (int row = row_start; row < row_end; row++) {
    for (int col = col_start; col < col_end; col++) {
      iree_e2e_test_value_t elem = read_matrix_element(
          rows, cols, elem_type, mapping.contents.data, row, col);
      char buf[64];
      int this_elem_width = snprintf_value(buf, sizeof buf, elem, precision);
      // iree_max is a macro, may evaluate its args twice. Give it plain ints.
      max_elem_width = iree_max(max_elem_width, this_elem_width);
    }
  }
  IREE_CHECK_OK(iree_hal_buffer_unmap_range(&mapping));
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
                         int row_start, int row_end, int col_start, int col_end,
                         iree_hal_buffer_view_t* matrix,
                         iree_hal_buffer_view_t* other_matrix,
                         const char* highlight) {
  assert((other_matrix == NULL) == (highlight == NULL));
  iree_hal_dim_t dims[2] = {0};
  get_matrix_shape(matrix, dims);
  int rows = dims[0];
  int cols = dims[1];
  iree_hal_element_type_t elem_type = iree_hal_buffer_view_element_type(matrix);
  iree_hal_buffer_mapping_t mapping;
  IREE_CHECK_OK(map_host_local_row_major_data(
      matrix, IREE_HAL_MEMORY_ACCESS_READ, &mapping));
  int max_elem_width = get_max_elem_width(precision, row_start, row_end,
                                          col_start, col_end, matrix);
  iree_hal_buffer_mapping_t other_mapping;
  if (other_matrix) {
    IREE_CHECK_OK(map_host_local_row_major_data(
        other_matrix, IREE_HAL_MEMORY_ACCESS_READ, &other_mapping));
    int other_matrix_max_elem_width = get_max_elem_width(
        precision, row_start, row_end, col_start, col_end, other_matrix);
    // iree_max is a macro, may evaluate its args twice. Give it plain ints.
    max_elem_width = iree_max(max_elem_width, other_matrix_max_elem_width);
  }

  fprintf(file,
          "%s (rows %d..%d out of %d..%d, columns %d..%d out of %d..%d)\n",
          label, row_start, row_end - 1, 0, rows - 1, col_start, col_end - 1, 0,
          cols - 1);
  for (int row = row_start; row < row_end; row++) {
    for (int col = col_start; col < col_end; col++) {
      iree_e2e_test_value_t elem = read_matrix_element(
          rows, cols, elem_type, mapping.contents.data, row, col);
      bool disagree = false;
      if (other_matrix) {
        iree_e2e_test_value_t other_elem = read_matrix_element(
            rows, cols, elem_type, other_mapping.contents.data, row, col);
        disagree = !matmul_result_elements_agree(elem, other_elem);
      }
      char buf[64];
      snprintf_value(buf, sizeof buf, elem, precision);
      fprintf(file, "%*s", max_elem_width, buf);
      // See comment on |highlight| function parameter for why 2 spaces.
      // A 3rd space is added unconditionally to make it clear that a highlight
      // concerns the matrix entry to its left.
      fprintf(file, "%s ", disagree ? highlight : "  ");
    }
    fprintf(file, "\n");
  }

  IREE_CHECK_OK(iree_hal_buffer_unmap_range(&mapping));
  if (other_matrix) {
    IREE_CHECK_OK(iree_hal_buffer_unmap_range(&mapping));
  }
}

// Helper for check_matmul_results: handler for the failure case.
// If |file| is not NULL, detailed logging is written to it.
static iree_status_t check_matmul_failure(
    FILE* file, iree_e2e_test_value_t actual_value,
    iree_e2e_test_value_t expected_value, iree_hal_dim_t row,
    iree_hal_dim_t col, iree_hal_buffer_view_t* lhs,
    iree_hal_buffer_view_t* rhs, iree_hal_buffer_view_t* acc,
    iree_hal_buffer_view_t* actual_result,
    iree_hal_buffer_view_t* expected_result, int check_every) {
  if (!file || check_every > 1) {
    // No logging of errors with check_every>1 as most of the reference matrix
    // elements have not been computed. The caller is expected to retry with
    // check_every=1.
    return iree_make_status(IREE_STATUS_ABORTED);
  }
  fprintf(file,
          "\n\nerror: the actual and expected result matrices disagree "
          "at row %" PRIdim ", column %" PRIdim ".\n\n",
          row, col);
  char actual_value_buf[32];
  char expected_value_buf[32];
  snprintf_value(actual_value_buf, sizeof actual_value_buf, actual_value,
                 PRECISION_HIGH);
  snprintf_value(expected_value_buf, sizeof expected_value_buf, expected_value,
                 PRECISION_HIGH);
  fprintf(file, "actual value: %s\n", actual_value_buf);
  fprintf(file, "expected value: %s\n", expected_value_buf);

  iree_hal_dim_t m_size, k_size, n_size;
  IREE_RETURN_IF_ERROR(get_matmul_sizes(lhs, rhs, acc, actual_result, &m_size,
                                        &k_size, &n_size));
  iree_hal_dim_t context = 8;
  const char* context_env = getenv("IREE_MATMUL_TEST_SHOW_CONTEXT");
  if (context_env) {
    if (1 != sscanf(context_env, "%" PRIdim, &context)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Failed to parse IREE_MATMUL_TEST_SHOW_CONTEXT "
                              "as \"%%" PRIdim "\". Got \"%s\"",
                              context_env);
    }
  }
  int m_start = iree_max(0, (int)row - (int)context);
  int m_end = iree_min(m_size, row + context);
  int n_start = iree_max(0, (int)col - (int)context);
  int n_end = iree_min(n_size, col + context);
  // We have a lot more freedom to pick k_start, k_end, since these parameters
  // only affect which regions of the input lhs and rhs matrices are printed.
  // If we were only testing random lhs and rhs, we would just pick
  // k_start = 0 and any reasonable k_end value.
  int k_start = iree_max(0, iree_min(m_start, n_start));
  int k_end = iree_min(k_size, iree_max(m_end, n_end));
  // [k_start, k_end) could be arbitrarily long at this point. Constrain it a
  // bit to avoid huge output.
  k_end = iree_min(k_end, k_start + 4 * context);

  fprintf(file, "\n");
  print_matrix(file, "left-hand side", PRECISION_LOW, m_start, m_end, k_start,
               k_end, lhs, NULL, NULL);
  fprintf(file, "\n");
  print_matrix(file, "right-hand side", PRECISION_LOW, k_start, k_end, n_start,
               n_end, rhs, NULL, NULL);
  fprintf(file, "\n");
  if (acc) {
    print_matrix(file, "input accumulator", PRECISION_LOW, m_start, m_end,
                 n_start, n_end, acc, NULL, NULL);
    fprintf(file, "\n");
  }
  print_matrix(file, "expected result", PRECISION_LOW, m_start, m_end, n_start,
               n_end, expected_result, actual_result, emoji(true));
  fprintf(file, "\n");
  print_matrix(file, "actual result", PRECISION_LOW, m_start, m_end, n_start,
               n_end, actual_result, expected_result, emoji(false));
  fprintf(file, "\n");
  return iree_make_status(IREE_STATUS_ABORTED);
}

// Helper for check_matmul_results: the actual interesting part once we've
// obtained and validated the {m,k,n}_size values. On error, detailed logging is
// written to |file| if it is not NULL.
static iree_status_t check_matmul_results_impl(
    FILE* file, iree_hal_dim_t m_size, iree_hal_dim_t k_size,
    iree_hal_dim_t n_size, iree_hal_buffer_view_t* lhs,
    iree_hal_buffer_view_t* rhs, iree_hal_buffer_view_t* acc,
    iree_hal_buffer_view_t* actual_result,
    iree_hal_buffer_view_t* expected_result, int check_every) {
  iree_hal_buffer_mapping_t actual_result_mapping;
  iree_hal_buffer_mapping_t expected_result_mapping;
  IREE_RETURN_IF_ERROR(map_host_local_row_major_data(
      actual_result, IREE_HAL_MEMORY_ACCESS_READ, &actual_result_mapping));
  IREE_RETURN_IF_ERROR(map_host_local_row_major_data(
      expected_result, IREE_HAL_MEMORY_ACCESS_READ, &expected_result_mapping));
  iree_hal_element_type_t result_type =
      iree_hal_buffer_view_element_type(actual_result);
  int count = 0;
  for (iree_hal_dim_t m = 0; m < m_size; ++m) {
    for (iree_hal_dim_t n = 0; n < n_size; ++n) {
      if (++count < check_every) continue;
      count = 0;
      iree_e2e_test_value_t actual_value =
          read_matrix_element(m_size, n_size, result_type,
                              actual_result_mapping.contents.data, m, n);
      iree_e2e_test_value_t expected_value =
          read_matrix_element(m_size, n_size, result_type,
                              expected_result_mapping.contents.data, m, n);
      if (!matmul_result_elements_agree(actual_value, expected_value)) {
        return check_matmul_failure(file, actual_value, expected_value, m, n,
                                    lhs, rhs, acc, actual_result,
                                    expected_result, check_every);
      }
    }
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_unmap_range(&actual_result_mapping));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_unmap_range(&expected_result_mapping));
  return iree_ok_status();
}

// Given an actual matmul's inputs and output (all host-local), uses a reference
// matmul implementation on the same inputs to check if the output is correct.
// On error, detailed logging is written to |file| if it is not NULL.
static iree_status_t check_matmul_results(
    FILE* file, iree_vm_list_t* input_list,
    iree_hal_buffer_view_t* actual_result,
    iree_hal_buffer_view_t* expected_result) {
  iree_hal_buffer_view_t* lhs = NULL;
  iree_hal_buffer_view_t* rhs = NULL;
  iree_hal_buffer_view_t* acc = NULL;
  IREE_RETURN_IF_ERROR(get_item_as_buffer_view(input_list, 0, &lhs));
  IREE_RETURN_IF_ERROR(get_item_as_buffer_view(input_list, 1, &rhs));
  if (iree_vm_list_size(input_list) == 3) {
    IREE_RETURN_IF_ERROR(get_item_as_buffer_view(input_list, 2, &acc));
  }

  iree_hal_dim_t m_size, k_size, n_size;
  IREE_RETURN_IF_ERROR(get_matmul_sizes(lhs, rhs, acc, actual_result, &m_size,
                                        &k_size, &n_size));

  int check_every = 1;
  if (FLAG_max_elements_to_check) {
    check_every = (iree_hal_buffer_view_element_count(actual_result) +
                   FLAG_max_elements_to_check - 1) /
                  FLAG_max_elements_to_check;
    if (check_every < 1) check_every = 1;
    if (check_every > 1)
      while ((n_size % check_every) == 0) ++check_every;
  }

  IREE_CHECK_OK(reference_matmul(input_list, expected_result, check_every));

  iree_status_t status =
      check_matmul_results_impl(file, m_size, k_size, n_size, lhs, rhs, acc,
                                actual_result, expected_result, check_every);

  if (!iree_status_is_ok(status) && check_every > 1) {
    // If we got a failure with check_every>1, that didn't log a useful
    // numerical summary, as most of the reference matrix entries hadn't been
    // computed. Rerun now with check_every=1 to get that numerical logging.
    status = check_matmul_results_impl(file, m_size, k_size, n_size, lhs, rhs,
                                       acc, actual_result, expected_result, 1);
  }

  return status;
}

/*****************************************************************************
 *
 * Part 4:
 *
 * Core matmul test logic.
 *
 * The entry point into this part is |replay_event_call_matmul|, which is the
 * handler for each matmul testcase in the trace. Other functions are only
 * helpers for it.
 *
 * |replay_event_call_matmul| calls |do_matmul_and_check_results| to actually
 * perform a matmul.
 *
 *****************************************************************************/

// Deep-copies device-local list of buffer_views |src| into |dst|.
static iree_status_t copy_device_buffer_views_to_device(
    iree_hal_device_t* device, iree_hal_allocator_t* hal_allocator,
    iree_vm_list_t* src_list, iree_vm_list_t** dst_list) {
  iree_vm_type_def_t elem_type = iree_vm_list_element_type(src_list);
  iree_host_size_t size = iree_vm_list_size(src_list);
  iree_allocator_t allocator = iree_hal_allocator_host_allocator(hal_allocator);
  IREE_RETURN_IF_ERROR(
      iree_vm_list_create(elem_type, size, allocator, dst_list));
  IREE_RETURN_IF_ERROR(iree_vm_list_resize(*dst_list, size));
  for (iree_host_size_t i = 0; i < size; ++i) {
    iree_hal_buffer_view_t* src = NULL;
    IREE_RETURN_IF_ERROR(get_item_as_buffer_view(src_list, i, &src));
    iree_hal_buffer_view_t* dst = NULL;
    IREE_RETURN_IF_ERROR(
        copy_device_buffer_view_to_device(device, hal_allocator, src, &dst));
    iree_vm_ref_t dst_ref = {0};
    IREE_RETURN_IF_ERROR(
        iree_vm_ref_wrap_assign(dst, iree_hal_buffer_view_type(), &dst_ref));
    IREE_RETURN_IF_ERROR(iree_vm_list_set_ref_move(*dst_list, i, &dst_ref));
  }
  return iree_ok_status();
}

// Performs one matmul test, on the device-local input matrices given in
// |original_device_inputs|.
//
// The contents of |original_device_inputs| are preserved, even if the
// |function| would overwrite input-output arguments (e.g. the accumulator).
static iree_status_t do_matmul_and_check_results(
    FILE* file, iree_trace_replay_t* replay, iree_vm_function_t function,
    iree_vm_list_t* original_device_inputs) {
  iree_hal_allocator_t* device_allocator =
      iree_hal_device_allocator(replay->device);

  // Perform a deep copy of the inputs to pass to the test function.
  // Needed as the test function may mutate some of the input list elements,
  // e.g. input-output parameters. For instance, the accumulator input of a
  // linalg.matmul. We need to preserve the original test inputs to perform
  // reruns on variants in the failure case (see |replay_event_call_matmul|).
  iree_vm_list_t* device_inputs = NULL;
  IREE_CHECK_OK(copy_device_buffer_views_to_device(
      replay->device, device_allocator, original_device_inputs,
      &device_inputs));

  // Perform a deep copy of the device-local inputs into host-local buffers.
  // Needed to pass to the reference matmul implementation and to logging
  // in the failure case.
  iree_vm_list_t* host_inputs = NULL;
  IREE_CHECK_OK(copy_device_buffer_views_to_host(
      replay->device, device_allocator, device_inputs, &host_inputs));

  // Invoke the function to produce the actual result.
  iree_vm_list_t* outputs = NULL;
  IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                    /*initial_capacity=*/8,
                                    replay->host_allocator, &outputs));
  IREE_CHECK_OK(iree_vm_invoke(
      replay->context, function, IREE_VM_INVOCATION_FLAG_NONE,
      /*policy=*/NULL, device_inputs, outputs, replay->host_allocator));
  iree_vm_list_release(device_inputs);

  // Transfer device buffers to host buffers.
  iree_hal_buffer_params_t host_params = {
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type =
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
      .min_alignment = 0,
  };
  IREE_CHECK_OK(iree_tooling_transfer_variant_list(
      replay->device, outputs, device_allocator, host_params,
      /*wait_fence=*/NULL, /*signal_fence=*/NULL));

  // Get the actual result computed by the program.
  iree_hal_buffer_view_t* actual_result;
  IREE_CHECK_OK(get_item_as_buffer_view(outputs, 0, &actual_result));

  // Allocate host_expected_result with same shape as actual_result.
  iree_hal_buffer_view_t* host_expected_result = NULL;
  IREE_CHECK_OK(allocate_host_buffer_view_like(
      replay->device, device_allocator, actual_result, &host_expected_result));

  // Check that actual_result and host_expected_result agree.
  iree_status_t status = check_matmul_results(file, host_inputs, actual_result,
                                              host_expected_result);

  iree_vm_list_release(outputs);  // releases actual_result
  iree_vm_list_release(host_inputs);
  iree_hal_buffer_view_release(host_expected_result);
  return status;
}

// Prints to |file| a message about the matmul shape. Useful as testcases
// otherwise only print the function name, and in the dynamic-shapes cases, that
// doesn't tell the actual shape.
static iree_status_t print_matmul_shape(FILE* file,
                                        iree_vm_list_t* input_list) {
  iree_hal_buffer_view_t* lhs = NULL;
  iree_hal_buffer_view_t* rhs = NULL;
  IREE_RETURN_IF_ERROR(get_item_as_buffer_view(input_list, 0, &lhs));
  IREE_RETURN_IF_ERROR(get_item_as_buffer_view(input_list, 1, &rhs));
  iree_hal_dim_t lhs_dims[2] = {0};
  iree_hal_dim_t rhs_dims[2] = {0};
  IREE_RETURN_IF_ERROR(get_matrix_shape(lhs, lhs_dims));
  IREE_RETURN_IF_ERROR(get_matrix_shape(rhs, rhs_dims));
  fprintf(file, "Matmul shape (MxKxN): %" PRIdim "x%" PRIdim "x%" PRIdim "\n",
          lhs_dims[0], lhs_dims[1], rhs_dims[1]);
  return iree_ok_status();
}

// Special handler for function calls in a e2e matmul test trace.
static iree_status_t replay_event_call_matmul(iree_trace_replay_t* replay,
                                              yaml_document_t* document,
                                              yaml_node_t* event_node) {
  yaml_node_t* function_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_find(
      document, event_node, iree_make_cstring_view("function"),
      &function_node));
  iree_string_view_t function_name = iree_yaml_node_as_string(function_node);
  fprintf(stderr, "--- CALL[%.*s] ---\n", (int)function_name.size,
          function_name.data);

  iree_vm_function_t function;
  iree_vm_list_t* device_inputs = NULL;
  IREE_RETURN_IF_ERROR(iree_trace_replay_event_call_prepare(
      replay, document, event_node, &function, &device_inputs));

  IREE_CHECK_OK(print_matmul_shape(stderr, device_inputs));

  iree_status_t status =
      do_matmul_and_check_results(stderr, replay, function, device_inputs);

  // Clean up.
  iree_vm_list_release(device_inputs);

  return status;
}

/*****************************************************************************
 *
 * Part 5:
 *
 * main function and high-level logic before one enters matmul test details.
 *
 *****************************************************************************/

// Helper for |replay_event_requirements|.
static iree_status_t iree_cpu_has_required_target_features(
    yaml_document_t* document, yaml_node_t* target_features_node) {
  for (yaml_node_item_t* item = target_features_node->data.sequence.items.start;
       item != target_features_node->data.sequence.items.top; ++item) {
    yaml_node_t* item_node = yaml_document_get_node(document, *item);
    iree_string_view_t required_feature = iree_yaml_node_as_string(item_node);
    if (iree_string_view_is_empty(required_feature)) continue;
    int64_t feature_is_supported = 0;
    IREE_RETURN_IF_ERROR(
        iree_cpu_lookup_data_by_key(required_feature, &feature_is_supported));
    if (!feature_is_supported) {
      return iree_make_status(
          // The error status matters. We distinguish "feature not supported",
          // which is a normal thing to happen, from actual errors.
          IREE_STATUS_UNAVAILABLE,
          "The target device does not have the required feature '%.*s'.\n",
          (int)required_feature.size, required_feature.data);
    }
  }
  return iree_ok_status();
}

// returns UNAVAILABLE if the required CPU feature is not supported by the CPU.
static iree_status_t replay_event_requirements(iree_trace_replay_t* replay,
                                               yaml_document_t* document,
                                               yaml_node_t* event_node) {
  yaml_node_t* target_features_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_find(
      document, event_node, iree_make_cstring_view("target_features"),
      &target_features_node));
  return iree_cpu_has_required_target_features(document, target_features_node);
}

static iree_status_t iree_e2e_matmul_test_trace_replay_event(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* event_node) {
  if (event_node->type != YAML_MAPPING_NODE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "(%zu): expected mapping node",
                            event_node->start_mark.line);
  }
  yaml_node_t* type_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_find(
      document, event_node, iree_make_cstring_view("type"), &type_node));
  if (iree_yaml_string_equal(type_node, iree_make_cstring_view("call"))) {
    return replay_event_call_matmul(replay, document, event_node);
  } else if (iree_yaml_string_equal(type_node,
                                    iree_make_cstring_view("requirements"))) {
    return replay_event_requirements(replay, document, event_node);
  } else {
    return iree_trace_replay_event(replay, document, event_node);
  }
}

// Runs the trace in |file| using |root_path| as the base for any path lookups
// required for external files referenced in |file|.
static iree_status_t run_trace_file(iree_string_view_t root_path, FILE* file,
                                    iree_vm_instance_t* instance) {
  iree_trace_replay_t replay;
  IREE_RETURN_IF_ERROR(iree_trace_replay_initialize(
      root_path, instance, IREE_TRACE_REPLAY_FLAG_NONE,
      FLAG_trace_execution ? IREE_VM_CONTEXT_FLAG_TRACE_EXECUTION
                           : IREE_VM_CONTEXT_FLAG_NONE,
      iree_hal_available_driver_registry(), iree_allocator_system(), &replay));

  // Query device overrides, if any. When omitted the devices from the trace
  // file will be used.
  // TODO(#5724): remove this and instead provide a device set on initialize.
  iree_host_size_t device_uri_count = 0;
  const iree_string_view_t* device_uris = NULL;
  iree_hal_get_devices_flag_list(&device_uri_count, &device_uris);
  iree_trace_replay_set_hal_devices_override(&replay, device_uri_count,
                                             device_uris);

  yaml_parser_t parser;
  if (!yaml_parser_initialize(&parser)) {
    iree_trace_replay_deinitialize(&replay);
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "yaml_parser_initialize failed");
  }
  yaml_parser_set_input_file(&parser, file);

  iree_status_t status = iree_ok_status();
  for (bool document_eof = false; !document_eof;) {
    yaml_document_t document;
    if (!yaml_parser_load(&parser, &document)) {
      status = iree_status_from_yaml_parser_error(&parser);
      break;
    }
    yaml_node_t* event_node = yaml_document_get_root_node(&document);
    if (event_node) {
      status = iree_e2e_matmul_test_trace_replay_event(&replay, &document,
                                                       event_node);
    } else {
      document_eof = true;
    }
    yaml_document_delete(&document);
    if (!iree_status_is_ok(status)) break;
  }

  yaml_parser_delete(&parser);
  iree_trace_replay_deinitialize(&replay);
  return status;
}

// Runs each of the given traces files sequentially in isolated contexts.
static iree_status_t run_trace_files(int file_count, char** file_paths,
                                     iree_vm_instance_t* instance) {
  for (int i = 0; i < file_count; ++i) {
    iree_string_view_t file_path = iree_make_cstring_view(file_paths[i]);
    iree_string_view_t root_path = iree_file_path_dirname(file_path);
    FILE* file = fopen(file_paths[i], "rb");
    if (!file) {
      return iree_make_status(iree_status_code_from_errno(errno),
                              "failed to open trace file '%.*s'",
                              (int)file_path.size, file_path.data);
    }
    iree_status_t status = run_trace_file(root_path, file, instance);
    fclose(file);
    IREE_RETURN_IF_ERROR(status, "replaying trace file '%.*s'",
                         (int)file_path.size, file_path.data);
  }
  return iree_ok_status();
}

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();

  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (argc <= 1) {
    fprintf(stderr,
            "no trace files provided; pass one or more yaml file paths\n");
    IREE_TRACE_APP_EXIT(EXIT_FAILURE);
    return EXIT_FAILURE;
  }

  iree_vm_instance_t* instance = NULL;
  iree_status_t status = iree_vm_instance_create(
      IREE_VM_TYPE_CAPACITY_DEFAULT, iree_allocator_system(), &instance);
  if (iree_status_is_ok(status)) {
    status = run_trace_files(argc - 1, argv + 1, instance);
  }
  iree_vm_instance_release(instance);
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
