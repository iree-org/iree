// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/file_path.h"
#include "iree/base/internal/flags.h"
#include "iree/base/target_platform.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/init.h"
#include "iree/modules/hal/module.h"
#include "iree/tools/utils/trace_replay.h"
#include "iree/tools/utils/yaml_util.h"
#include "iree/vm/api.h"

IREE_FLAG(bool, trace_execution, false, "Traces VM execution to stderr.");

IREE_FLAG(string, driver, "vmvx", "Backend driver to use.");

// Helper to get a list item as a buffer_view.
static iree_status_t iree_get_buffer_view_list_item(
    iree_vm_list_t* list, iree_host_size_t i,
    iree_hal_buffer_view_t** out_value) {
  iree_vm_variant_t variant = iree_vm_variant_empty();
  IREE_RETURN_IF_ERROR(iree_vm_list_get_variant(list, i, &variant));
  if (!iree_vm_variant_is_ref(variant)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected list item %zu to be a ref", i);
  }
  return iree_hal_buffer_view_check_deref(variant.ref, out_value);
}

// Helper to get the shape of a buffer_view that is a matrix, meaning
// has a 2D shape with positive dimensions.
static iree_status_t get_matrix_buffer_view_shape(
    iree_hal_buffer_view_t* buffer_view, iree_hal_dim_t* dims) {
  iree_host_size_t shape_rank = iree_hal_buffer_view_shape_rank(buffer_view);
  if (shape_rank != 2) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "expected a matrix (2D tensor) shape, got a %zu-dimensional shape",
        shape_rank);
  }
  dims[0] = iree_hal_buffer_view_shape_dim(buffer_view, 0);
  dims[1] = iree_hal_buffer_view_shape_dim(buffer_view, 1);
  if (!(dims[0] > 0 && dims[1] > 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected matrix dims to be positive, got %dx%d",
                            dims[0], dims[1]);
  }
  return iree_ok_status();
}

// Helper to get a pointer to dense row-major data in a buffer_view.
static iree_status_t get_buffer_view_dense_row_major_data(
    iree_hal_buffer_view_t* buffer_view, void** data) {
  if (iree_hal_buffer_view_encoding_type(buffer_view) !=
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer_view is not dense row major");
  }
  iree_hal_buffer_mapping_t mapping;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      iree_hal_buffer_view_buffer(buffer_view), IREE_HAL_MEMORY_ACCESS_READ, 0,
      IREE_WHOLE_BUFFER, &mapping));
  *data = mapping.contents.data;
  return iree_ok_status();
}

// Helper for iree_check_matmul and reference_matmul:
// obtain and validate the {m,k,n}_size values.
static iree_status_t get_matmul_sizes(
    iree_hal_buffer_view_t* lhs, iree_hal_buffer_view_t* rhs,
    iree_hal_buffer_view_t* acc, iree_hal_buffer_view_t* result,
    iree_hal_dim_t* m_size, iree_hal_dim_t* k_size, iree_hal_dim_t* n_size) {
  iree_hal_dim_t lhs_dims[2];
  iree_hal_dim_t rhs_dims[2];
  iree_hal_dim_t acc_dims[2];
  iree_hal_dim_t result_dims[2];
  IREE_RETURN_IF_ERROR(get_matrix_buffer_view_shape(lhs, lhs_dims));
  IREE_RETURN_IF_ERROR(get_matrix_buffer_view_shape(rhs, rhs_dims));
  IREE_RETURN_IF_ERROR(get_matrix_buffer_view_shape(acc, acc_dims));
  IREE_RETURN_IF_ERROR(get_matrix_buffer_view_shape(result, result_dims));
  *m_size = lhs_dims[0];
  *k_size = lhs_dims[1];
  *n_size = rhs_dims[1];
  if (!(lhs_dims[0] == *m_size && lhs_dims[1] == *k_size &&
        rhs_dims[0] == *k_size && rhs_dims[1] == *n_size &&
        acc_dims[0] == *m_size && acc_dims[1] == *n_size &&
        result_dims[0] == *m_size && result_dims[1] == *n_size)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "mismatched matrix shapes in matmul: %dx%d * %dx%d + %dx%d -> %dx%d",
        lhs_dims[0], lhs_dims[1], rhs_dims[0], rhs_dims[1], acc_dims[0],
        acc_dims[1], result_dims[0], result_dims[1]);
  }
  return iree_ok_status();
}

// Helper for reference_matmul_element. f32 case.
static void reference_matmul_element_f32(
    iree_hal_dim_t m_size, iree_hal_dim_t k_size, iree_hal_dim_t n_size,
    iree_hal_element_type_t lhs_type, iree_hal_element_type_t rhs_type,
    float* lhs_data, float* rhs_data, float* acc_data, float* result_data,
    iree_hal_dim_t m, iree_hal_dim_t n) {
  float acc = acc_data[n + m * n_size];
  for (iree_hal_dim_t k = 0; k < k_size; ++k) {
    float lhs_value = lhs_data[k + m * k_size];
    float rhs_value = rhs_data[n + k * n_size];
    acc += lhs_value * rhs_value;
  }
  result_data[n + m * n_size] = acc;
}

// Helper for reference_matmul_element. i8*i8->i32 case.
static void reference_matmul_element_i8_i8_i32(
    iree_hal_dim_t m_size, iree_hal_dim_t k_size, iree_hal_dim_t n_size,
    iree_hal_element_type_t lhs_type, iree_hal_element_type_t rhs_type,
    int8_t* lhs_data, int8_t* rhs_data, int32_t* acc_data, int32_t* result_data,
    iree_hal_dim_t m, iree_hal_dim_t n) {
  int32_t acc = acc_data[n + m * n_size];
  for (iree_hal_dim_t k = 0; k < k_size; ++k) {
    int8_t lhs_value = lhs_data[k + m * k_size];
    int8_t rhs_value = rhs_data[n + k * n_size];
    acc += ((int32_t)lhs_value) * ((int32_t)rhs_value);
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
    reference_matmul_element_f32(m_size, k_size, n_size, lhs_type, rhs_type,
                                 (float*)lhs_data, (float*)rhs_data,
                                 (float*)acc_data, (float*)result_data, m, n);
  } else if (iree_hal_element_type_is_integer(lhs_type, 8) &&
             iree_hal_element_type_is_integer(rhs_type, 8) &&
             iree_hal_element_type_is_integer(acc_type, 32)) {
    reference_matmul_element_i8_i8_i32(
        m_size, k_size, n_size, lhs_type, rhs_type, (int8_t*)lhs_data,
        (int8_t*)rhs_data, (int32_t*)acc_data, (int32_t*)result_data, m, n);
  } else {
    iree_status_abort(
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "unhandled combination of element types in matmul"));
  }
}

// Reference matmul implementation, used to compare matmul results against.
static iree_status_t reference_matmul(iree_vm_list_t* input_list,
                                      iree_hal_buffer_view_t* result) {
  iree_hal_buffer_view_t* lhs;
  iree_hal_buffer_view_t* rhs;
  iree_hal_buffer_view_t* acc;
  IREE_RETURN_IF_ERROR(iree_get_buffer_view_list_item(input_list, 0, &lhs));
  IREE_RETURN_IF_ERROR(iree_get_buffer_view_list_item(input_list, 1, &rhs));
  IREE_RETURN_IF_ERROR(iree_get_buffer_view_list_item(input_list, 2, &acc));

  iree_hal_dim_t m_size, k_size, n_size;
  IREE_RETURN_IF_ERROR(
      get_matmul_sizes(lhs, rhs, acc, result, &m_size, &k_size, &n_size));
  void* lhs_data;
  void* rhs_data;
  void* acc_data;
  void* result_data;
  IREE_RETURN_IF_ERROR(get_buffer_view_dense_row_major_data(lhs, &lhs_data));
  IREE_RETURN_IF_ERROR(get_buffer_view_dense_row_major_data(rhs, &rhs_data));
  IREE_RETURN_IF_ERROR(get_buffer_view_dense_row_major_data(acc, &acc_data));
  IREE_RETURN_IF_ERROR(
      get_buffer_view_dense_row_major_data(result, &result_data));
  iree_hal_element_type_t lhs_type = iree_hal_buffer_view_element_type(lhs);
  iree_hal_element_type_t rhs_type = iree_hal_buffer_view_element_type(rhs);
  iree_hal_element_type_t acc_type = iree_hal_buffer_view_element_type(acc);
  for (iree_hal_dim_t m = 0; m < m_size; ++m) {
    for (iree_hal_dim_t n = 0; n < n_size; ++n) {
      reference_matmul_element(m_size, k_size, n_size, lhs_type, rhs_type,
                               acc_type, lhs_data, rhs_data, acc_data,
                               result_data, m, n);
    }
  }
  return iree_ok_status();
}

// Reads an element from a (row-major) matrix.
static iree_vm_value_t read_matrix_element(iree_hal_dim_t m_size,
                                           iree_hal_dim_t n_size,
                                           iree_hal_element_type_t result_type,
                                           void* data, iree_hal_dim_t m,
                                           iree_hal_dim_t n) {
  iree_host_size_t index = n + m * n_size;
  (void)m_size;
  if (iree_hal_element_type_is_integer(result_type, 8)) {
    return iree_vm_value_make_i8(((int8_t*)data)[index]);
  } else if (iree_hal_element_type_is_integer(result_type, 16)) {
    return iree_vm_value_make_i16(((int16_t*)data)[index]);
  } else if (iree_hal_element_type_is_integer(result_type, 32)) {
    return iree_vm_value_make_i32(((int32_t*)data)[index]);
  } else if (result_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32) {
    return iree_vm_value_make_f32(((float*)data)[index]);
  }
  iree_status_abort(iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                     "unhandled matmul result type"));
  return iree_vm_value_make_none();
}

typedef enum precision_e {
  PRECISION_LOW,
  PRECISION_HIGH,
} precision_t;

// Prints a iree_vm_value_t to a string buffer. Returns the number of
// characters written. Like snprintf.
static int snprintf_value(char* buf, size_t bufsize, iree_vm_value_t value,
                          precision_t precision) {
  switch (value.type) {
    case IREE_VM_VALUE_TYPE_I8:
      return snprintf(buf, bufsize, "%" PRIi8, value.i8);
    case IREE_VM_VALUE_TYPE_I16:
      return snprintf(buf, bufsize, "%" PRIi16, value.i16);
    case IREE_VM_VALUE_TYPE_I32:
      return snprintf(buf, bufsize, "%" PRIi32, value.i32);
    case IREE_VM_VALUE_TYPE_I64:
      return snprintf(buf, bufsize, "%" PRIi64, value.i64);
    case IREE_VM_VALUE_TYPE_F32:
      return snprintf(buf, bufsize,
                      precision == PRECISION_HIGH ? "%.8g" : "%.4g", value.f32);
    case IREE_VM_VALUE_TYPE_F64:
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
static bool matmul_result_elements_agree(iree_vm_value_t expected,
                                         iree_vm_value_t actual) {
  if (expected.type != actual.type) {
    iree_status_abort(
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "mismatched types"));
    return false;
  }
  switch (expected.type) {
    case IREE_VM_VALUE_TYPE_I32:
      return actual.i32 == expected.i32;
    case IREE_VM_VALUE_TYPE_F32:
      // The absolute value difference comparison here is naive, bad.
      //
      // Why it's almost good enough: we are only testing matmuls here, not even
      // fused with any other op. Because of how matmul is defined (as a
      // polynomial expression with coefficients either 0 or 1), it's going to
      // be either correct or completely wrong. That wouldn't be
      // true if we were pursuing non-trivial accumulation strategies limiting
      // accumulation depth, but we are not doing that. Also, we are not testing
      // huge sizes, and all our test data is in the same order of magnitude.
      //
      // What would be the better thing to do here: adjust the tolerated
      // absolute value difference based on the magnitude of the matrix
      // elements, the accumulation depth (k_size) and the accumulator type's
      // epsilon. Floating-point calculations should be scale-invariant: matmul
      // tests should succeed or fail in the same way if we rescale all input
      // data by a constant factor (as long as we don't run out of exponents).
      return fabsf(actual.f32 - expected.f32) < 1e-3f;
    default:
      iree_status_abort(iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                         "unhandled value type"));
      return false;
  }
}

// Prints |matrix| to |file|, with |label| as caption.
// |precision| controls how many decimals are printed for float values.
//
// If |other_matrix| is not NULL, then any matrix entries that disagree
// between |matrix| and |other_matrix| (according to
// matmul_result_elements_agree) are highlighted.
static void print_matrix(FILE* file, const char* label, precision_t precision,
                         int row_start, int row_end, int col_start, int col_end,
                         iree_hal_buffer_view_t* matrix,
                         iree_hal_buffer_view_t* other_matrix) {
  iree_hal_dim_t dims[2];
  get_matrix_buffer_view_shape(matrix, dims);
  int rows = dims[0];
  int cols = dims[1];
  iree_hal_element_type_t elem_type = iree_hal_buffer_view_element_type(matrix);
  void* data = 0;
  get_buffer_view_dense_row_major_data(matrix, &data);
  void* other_data = 0;
  if (other_matrix) {
    get_buffer_view_dense_row_major_data(other_matrix, &other_data);
  }
  int max_elem_width = 0;
  for (int row = row_start; row < row_end; row++) {
    for (int col = col_start; col < col_end; col++) {
      iree_vm_value_t elem =
          read_matrix_element(rows, cols, elem_type, data, row, col);
      char buf[64];
      max_elem_width = iree_max(
          max_elem_width, snprintf_value(buf, sizeof buf, elem, precision));
    }
  }
  fprintf(file,
          "%s (rows %d..%d out of %d..%d, columns %d..%d out of %d..%d)\n",
          label, row_start, row_end - 1, 0, rows - 1, col_start, col_end - 1, 0,
          cols - 1);
  for (int row = row_start; row < row_end; row++) {
    for (int col = col_start; col < col_end; col++) {
      iree_vm_value_t elem =
          read_matrix_element(rows, cols, elem_type, (void*)data, row, col);
      bool bad_elem = false;
      if (other_matrix) {
        iree_vm_value_t other_elem = read_matrix_element(
            rows, cols, elem_type, (void*)other_data, row, col);
        bad_elem = !matmul_result_elements_agree(elem, other_elem);
      }
      char buf[64];
      snprintf_value(buf, sizeof buf, elem, precision);
      fprintf(file, "%*s", max_elem_width, buf);
      if (bad_elem) {
        fprintf(file, "💩");
      } else if (col < col_end - 1) {
        // two spaces per https://www.unicode.org/reports/tr11/#Recommendations
        fprintf(file, "  ");
      }
    }
    fprintf(file, "\n");
  }
}

// Helper for iree_check_matmul: handler for the failure case.
static iree_status_t iree_check_matmul_failure(
    FILE* file, iree_vm_value_t actual_value, iree_vm_value_t expected_value,
    iree_hal_dim_t row, iree_hal_dim_t col, iree_hal_buffer_view_t* lhs,
    iree_hal_buffer_view_t* rhs, iree_hal_buffer_view_t* acc,
    iree_hal_buffer_view_t* actual_result,
    iree_hal_buffer_view_t* expected_result) {
  fprintf(file,
          "\n\nerror: the actual and expected result matrices disagree "
          "at row %d, column %d.\n\n",
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
    if (1 != sscanf(context_env, "%d", &context)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Failed to parse IREE_MATMUL_TEST_SHOW_CONTEXT "
                              "as \"%%d\". Got \"%s\"",
                              context_env);
    }
  }
  int m_start = iree_max(0, row - context);
  int m_end = iree_min(m_size, row + context);
  int n_start = iree_max(0, col - context);
  int n_end = iree_min(n_size, col + context);
  // We have a lot more freedom to pick k_start, k_end, since these parameters
  // only affect which regions of the input lhs and rhs matrices are printed.
  // If we were only testing random lhs and rhs, we would just pick
  // k_start = 0 and any reasonable k_end value. Since we are often using
  // identity matrices for lhs and rhs, and we expect the majority of
  // test failures to occur with such identity matrices, we try to pick
  // k_start and k_end so that nontrivial regions of identity matrices will be
  // printed. That means that we try to have [k_start, k_end) intervals
  // overlap [m_start, m_end) and [n_start, n_end).
  int k_start = iree_max(0, iree_min(m_start, n_start));
  int k_end = iree_min(k_size, iree_max(m_end, n_end));
  // [k_start, k_end) could be arbitrarily long at this point. Constrain it a
  // bit to avoid huge output.
  k_end = iree_min(k_end, k_start + 4 * context);

  fprintf(file, "\n");
  print_matrix(file, "left-hand side", PRECISION_LOW, m_start, m_end, k_start,
               k_end, lhs, NULL);
  fprintf(file, "\n");
  print_matrix(file, "right-hand side", PRECISION_LOW, k_start, k_end, n_start,
               n_end, rhs, NULL);
  fprintf(file, "\n");
  print_matrix(file, "input accumulator", PRECISION_LOW, m_start, m_end,
               n_start, n_end, acc, NULL);
  fprintf(file, "\n");
  print_matrix(file, "expected result", PRECISION_LOW, m_start, m_end, n_start,
               n_end, expected_result, actual_result);
  fprintf(file, "\n");
  print_matrix(file, "actual result", PRECISION_LOW, m_start, m_end, n_start,
               n_end, actual_result, expected_result);
  fprintf(file, "\n");
  return iree_make_status(IREE_STATUS_ABORTED,
                          "matmul test failure, details logged above");
}

// Helper for iree_check_matmul: the actual interesting part once we've obtained
// and validated the {m,k,n}_size values.
static iree_status_t check_matmul_impl(
    iree_hal_dim_t m_size, iree_hal_dim_t k_size, iree_hal_dim_t n_size,
    iree_hal_buffer_view_t* lhs, iree_hal_buffer_view_t* rhs,
    iree_hal_buffer_view_t* acc, iree_hal_buffer_view_t* actual_result,
    iree_hal_buffer_view_t* expected_result) {
  void* actual_result_data;
  void* expected_result_data;
  IREE_RETURN_IF_ERROR(
      get_buffer_view_dense_row_major_data(actual_result, &actual_result_data));
  IREE_RETURN_IF_ERROR(get_buffer_view_dense_row_major_data(
      expected_result, &expected_result_data));
  iree_hal_element_type_t result_type =
      iree_hal_buffer_view_element_type(actual_result);
  for (iree_hal_dim_t m = 0; m < m_size; ++m) {
    for (iree_hal_dim_t n = 0; n < n_size; ++n) {
      iree_vm_value_t actual_value = read_matrix_element(
          m_size, n_size, result_type, actual_result_data, m, n);
      iree_vm_value_t expected_value = read_matrix_element(
          m_size, n_size, result_type, expected_result_data, m, n);
      if (!matmul_result_elements_agree(actual_value, expected_value)) {
        return iree_check_matmul_failure(stderr, actual_value, expected_value,
                                         m, n, lhs, rhs, acc, actual_result,
                                         expected_result);
      }
    }
  }
  return iree_ok_status();
}

// Given an actual matmul's inputs and output, uses a reference
// matmul implementation on the same inputs to check if the output
// is correct.
static iree_status_t iree_check_matmul(
    iree_vm_list_t* input_list, iree_hal_buffer_view_t* actual_result,
    iree_hal_buffer_view_t* expected_result) {
  iree_hal_buffer_view_t* lhs;
  iree_hal_buffer_view_t* rhs;
  iree_hal_buffer_view_t* acc;
  IREE_RETURN_IF_ERROR(iree_get_buffer_view_list_item(input_list, 0, &lhs));
  IREE_RETURN_IF_ERROR(iree_get_buffer_view_list_item(input_list, 1, &rhs));
  IREE_RETURN_IF_ERROR(iree_get_buffer_view_list_item(input_list, 2, &acc));

  iree_hal_dim_t m_size, k_size, n_size;
  IREE_RETURN_IF_ERROR(get_matmul_sizes(lhs, rhs, acc, actual_result, &m_size,
                                        &k_size, &n_size));

  return check_matmul_impl(m_size, k_size, n_size, lhs, rhs, acc, actual_result,
                           expected_result);
}

// Allocates |dst| to have the same shape as |src|, without copying contents.
static iree_status_t allocate_buffer_like(iree_hal_allocator_t* hal_allocator,
                                          iree_hal_buffer_view_t* src,
                                          iree_hal_buffer_view_t** dst) {
  return iree_hal_buffer_view_allocate_buffer(
      hal_allocator, iree_hal_buffer_view_shape_dims(src),
      iree_hal_buffer_view_shape_rank(src),
      iree_hal_buffer_view_element_type(src),
      iree_hal_buffer_view_encoding_type(src),
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      IREE_HAL_BUFFER_USAGE_ALL, iree_const_byte_span_empty(), dst);
}

// Performs a deep copy of |src| into |dst|. Takes care of allocating |dst|.
static iree_status_t copy_buffer(iree_hal_allocator_t* hal_allocator,
                                 iree_hal_buffer_view_t* src,
                                 iree_hal_buffer_view_t** dst) {
  // TODO(benvanik): change this to use iree_hal_buffer_copy_data. Or something.
  // I can't understand what all this code is doing.
  iree_hal_buffer_mapping_t src_mapping;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      iree_hal_buffer_view_buffer(src), IREE_HAL_MEMORY_ACCESS_READ, 0,
      IREE_WHOLE_BUFFER, &src_mapping));
  iree_const_byte_span_t src_span;
  src_span.data = src_mapping.contents.data;
  src_span.data_length = src_mapping.contents.data_length;
  return iree_hal_buffer_view_allocate_buffer(
      hal_allocator, iree_hal_buffer_view_shape_dims(src),
      iree_hal_buffer_view_shape_rank(src),
      iree_hal_buffer_view_element_type(src),
      iree_hal_buffer_view_encoding_type(src),
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      IREE_HAL_BUFFER_USAGE_ALL, src_span, dst);
}

static iree_status_t copy_list_of_buffer_views(
    iree_hal_allocator_t* hal_allocator, iree_vm_list_t* src,
    iree_vm_list_t** dst) {
  iree_vm_type_def_t elem_type;
  IREE_RETURN_IF_ERROR(iree_vm_list_element_type(src, &elem_type));
  iree_host_size_t size = iree_vm_list_size(src);
  iree_allocator_t allocator = iree_hal_allocator_host_allocator(hal_allocator);
  IREE_RETURN_IF_ERROR(iree_vm_list_create(&elem_type, size, allocator, dst));
  IREE_RETURN_IF_ERROR(iree_vm_list_resize(*dst, size));
  for (iree_host_size_t i = 0; i < size; ++i) {
    iree_hal_buffer_view_t* src_elem;
    IREE_RETURN_IF_ERROR(iree_get_buffer_view_list_item(src, i, &src_elem));
    iree_hal_buffer_view_t* dst_elem;
    IREE_RETURN_IF_ERROR(copy_buffer(hal_allocator, src_elem, &dst_elem));
    iree_vm_ref_t dst_elem_ref = {0};
    IREE_RETURN_IF_ERROR(iree_vm_ref_wrap_assign(
        dst_elem, iree_hal_buffer_view_type_id(), &dst_elem_ref));
    IREE_RETURN_IF_ERROR(iree_vm_list_set_ref_move(*dst, i, &dst_elem_ref));
  }
  return iree_ok_status();
}

// Special handler for function calls in a e2e matmul test trace.
// Assumes that all calls are to functions that take 3 inputs (lhs, rhs, acc)
// and return the result of a matmul (lhs*rhs+acc).
static iree_status_t replay_event_call(iree_trace_replay_t* replay,
                                       yaml_document_t* document,
                                       yaml_node_t* event_node) {
  yaml_node_t* function_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_find(
      document, event_node, iree_make_cstring_view("function"),
      &function_node));
  iree_string_view_t function_name = iree_yaml_node_as_string(function_node);
  fprintf(stderr, "--- CALL[%.*s] ---\n", (int)function_name.size,
          function_name.data);

  iree_hal_allocator_t* device_allocator =
      iree_hal_device_allocator(replay->device);

  iree_vm_function_t function;
  iree_vm_list_t* input_list = NULL;
  IREE_RETURN_IF_ERROR(iree_trace_replay_event_call_prepare(
      replay, document, event_node, &function, &input_list));

  // Perform a deep copy of the input list to pass to the test function.
  // Rationale: the test function may mutate some of the input list elements,
  // e.g. input-output parameters. For instance, the accumulator input of a
  // linalg.matmul. We need to preserve the original test inputs to run the
  // reference matmul on and to use in test failure logs.
  iree_vm_list_t* copy_of_input_list = NULL;
  copy_list_of_buffer_views(device_allocator, input_list, &copy_of_input_list);

  // Invoke the function to produce the actual result.
  iree_vm_list_t* output_list = NULL;
  IREE_CHECK_OK(iree_vm_list_create(/*element_type=*/NULL,
                                    /*initial_capacity=*/8,
                                    replay->host_allocator, &output_list));
  IREE_CHECK_OK(iree_vm_invoke(
      replay->context, function, IREE_VM_INVOCATION_FLAG_NONE, /*policy=*/NULL,
      copy_of_input_list, output_list, replay->host_allocator));

  // Get the actual_result buffer from the output_list.
  iree_hal_buffer_view_t* actual_result;
  IREE_RETURN_IF_ERROR(
      iree_get_buffer_view_list_item(output_list, 0, &actual_result));

  // Allocate an expected_result buffer, with same shape as actual_result.
  iree_hal_buffer_view_t* expected_result;
  IREE_RETURN_IF_ERROR(
      allocate_buffer_like(device_allocator, actual_result, &expected_result));

  // Use the reference matmul implementation to fill expected_result
  IREE_RETURN_IF_ERROR(reference_matmul(input_list, expected_result));

  // Check that actual_result and expected_result agree.
  IREE_CHECK_OK(iree_check_matmul(input_list, actual_result, expected_result));

  // Clean up.
  iree_vm_list_release(input_list);
  iree_vm_list_release(copy_of_input_list);
  iree_vm_list_release(output_list);  // releases actual_result
  iree_hal_buffer_view_release(expected_result);

  return iree_ok_status();
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
    return replay_event_call(replay, document, event_node);
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
      root_path, instance,
      FLAG_trace_execution ? IREE_VM_CONTEXT_FLAG_TRACE_EXECUTION
                           : IREE_VM_CONTEXT_FLAG_NONE,
      iree_allocator_system(), &replay));
  iree_trace_replay_set_hal_driver_override(
      &replay, iree_make_cstring_view(FLAG_driver));

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
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (argc <= 1) {
    fprintf(stderr,
            "no trace files provided; pass one or more yaml file paths");
    return 1;
  }

  iree_vm_instance_t* instance = NULL;
  iree_status_t status =
      iree_vm_instance_create(iree_allocator_system(), &instance);
  if (iree_status_is_ok(status)) {
    IREE_CHECK_OK(iree_hal_register_all_available_drivers(
        iree_hal_driver_registry_default()));
    status = run_trace_files(argc - 1, argv + 1, instance);
  }
  iree_vm_instance_release(instance);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return 1;
  }
  return 0;
}
