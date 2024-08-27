// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/base/internal/math.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/tooling/context_util.h"
#include "iree/tooling/device_util.h"
#include "iree/vm/api.h"
#include "iree/vm/native_module_cc.h"
#include "tools/testing/e2e/test_utils.h"

//===----------------------------------------------------------------------===//
// Reference conv2d (NCHW-FCHW)
//===----------------------------------------------------------------------===//

// Conversion from 4D indices in row major order to 1D index.
static int convert_to_1d_index(iree_hal_dim_t channels, iree_hal_dim_t height,
                               iree_hal_dim_t width, iree_hal_dim_t n,
                               iree_hal_dim_t c, iree_hal_dim_t h,
                               iree_hal_dim_t w) {
  return n * (channels * height * width) + c * (height * width) + h * width + w;
}

// [f16 <= f16 * f16 + f16]
static void reference_conv2d_f16_f16_f16_f16(
    iree_hal_dim_t n_size, iree_hal_dim_t c_size, iree_hal_dim_t h_size,
    iree_hal_dim_t w_size, iree_hal_dim_t f_size, iree_hal_dim_t kh_size,
    iree_hal_dim_t kw_size, iree_hal_dim_t sh_size, iree_hal_dim_t sw_size,
    iree_hal_dim_t dh_size, iree_hal_dim_t dw_size, iree_hal_dim_t oh_size,
    iree_hal_dim_t ow_size, const uint16_t* input_data,
    const uint16_t* kernel_data, const uint16_t* acc_data,
    uint16_t* result_data, iree_hal_dim_t n, iree_hal_dim_t oc,
    iree_hal_dim_t oh, iree_hal_dim_t ow) {
  iree_hal_dim_t out_idx =
      convert_to_1d_index(f_size, oh_size, ow_size, n, oc, oh, ow);

  float acc = acc_data ? iree_math_f16_to_f32(acc_data[out_idx]) : 0.f;

  for (iree_hal_dim_t ic = 0; ic < c_size; ++ic) {
    for (iree_hal_dim_t kh = 0; kh < kh_size; ++kh) {
      for (iree_hal_dim_t kw = 0; kw < kw_size; ++kw) {
        iree_hal_dim_t inp_idx = convert_to_1d_index(
            c_size, h_size, w_size, n, ic, (oh * sh_size + kh * dh_size),
            (ow * sw_size + kw * dw_size));
        iree_hal_dim_t krnl_idx =
            convert_to_1d_index(c_size, kh_size, kw_size, oc, ic, kh, kw);

        acc += iree_math_f16_to_f32(input_data[inp_idx]) *
               iree_math_f16_to_f32(kernel_data[krnl_idx]);
      }
    }
    result_data[out_idx] = iree_math_f32_to_f16(acc);
  }
}

static void reference_conv2d_f32_f32_f32_f32(
    iree_hal_dim_t n_size, iree_hal_dim_t c_size, iree_hal_dim_t h_size,
    iree_hal_dim_t w_size, iree_hal_dim_t f_size, iree_hal_dim_t kh_size,
    iree_hal_dim_t kw_size, iree_hal_dim_t sh_size, iree_hal_dim_t sw_size,
    iree_hal_dim_t dh_size, iree_hal_dim_t dw_size, iree_hal_dim_t oh_size,
    iree_hal_dim_t ow_size, const float* input_data, const float* kernel_data,
    const float* acc_data, float* result_data, iree_hal_dim_t n,
    iree_hal_dim_t oc, iree_hal_dim_t oh, iree_hal_dim_t ow) {
  iree_hal_dim_t out_idx =
      convert_to_1d_index(f_size, oh_size, ow_size, n, oc, oh, ow);

  float acc = acc_data ? acc_data[out_idx] : 0;

  for (iree_hal_dim_t ic = 0; ic < c_size; ++ic) {
    for (iree_hal_dim_t kh = 0; kh < kh_size; ++kh) {
      for (iree_hal_dim_t kw = 0; kw < kw_size; ++kw) {
        iree_hal_dim_t inp_idx = convert_to_1d_index(
            c_size, h_size, w_size, n, ic, (oh * sh_size + kh * dh_size),
            (ow * sw_size + kw * dw_size));
        iree_hal_dim_t krnl_idx =
            convert_to_1d_index(c_size, kh_size, kw_size, oc, ic, kh, kw);

        acc += input_data[inp_idx] * kernel_data[krnl_idx];
      }
    }
    result_data[out_idx] = acc;
  }
}

// Helper for reference_conv2d.
static iree_status_t reference_conv2d_element(
    iree_hal_dim_t n_size, iree_hal_dim_t c_size, iree_hal_dim_t h_size,
    iree_hal_dim_t w_size, iree_hal_dim_t f_size, iree_hal_dim_t kh_size,
    iree_hal_dim_t kw_size, iree_hal_dim_t sh_size, iree_hal_dim_t sw_size,
    iree_hal_dim_t dh_size, iree_hal_dim_t dw_size, iree_hal_dim_t oh_size,
    iree_hal_dim_t ow_size, iree_hal_element_type_t input_type,
    iree_hal_element_type_t kernel_type, iree_hal_element_type_t acc_type,
    void* input_data, void* kernel_data, void* acc_data, void* result_data,
    iree_hal_dim_t n, iree_hal_dim_t oc, iree_hal_dim_t oh, iree_hal_dim_t ow) {
  if (input_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32 &&
      kernel_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32 &&
      acc_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32) {
    reference_conv2d_f32_f32_f32_f32(
        n_size, c_size, h_size, w_size, f_size, kh_size, kw_size, sh_size,
        sw_size, dh_size, dw_size, oh_size, ow_size, (const float*)input_data,
        (const float*)kernel_data, (const float*)acc_data, (float*)result_data,
        n, oc, oh, ow);
  } else if (input_type == IREE_HAL_ELEMENT_TYPE_FLOAT_16 &&
             kernel_type == IREE_HAL_ELEMENT_TYPE_FLOAT_16 &&
             acc_type == IREE_HAL_ELEMENT_TYPE_FLOAT_16) {
    reference_conv2d_f16_f16_f16_f16(
        n_size, c_size, h_size, w_size, f_size, kh_size, kw_size, sh_size,
        sw_size, dh_size, dw_size, oh_size, ow_size,
        (const uint16_t*)input_data, (const uint16_t*)kernel_data,
        (const uint16_t*)acc_data, (uint16_t*)result_data, n, oc, oh, ow);
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unhandled combination of element types in conv2d");
  }
  return iree_ok_status();
}

// Calculate the output shape given the dilation and strides.
static iree_hal_dim_t out_shape_calc(iree_hal_dim_t i_shape,
                                     iree_hal_dim_t k_shape,
                                     iree_hal_dim_t stride,
                                     iree_hal_dim_t dilation) {
  iree_hal_dim_t x = (k_shape - 1) * (dilation - 1);
  x = i_shape - k_shape - x;
  return floor(x / stride) + 1;
}

// Reference conv2d-NCHW-FCHW implementation, used to compare conv2d results
// against.
static iree_status_t reference_conv2d(
    iree_hal_dim_t n_size, iree_hal_dim_t c_size, iree_hal_dim_t h_size,
    iree_hal_dim_t w_size, iree_hal_dim_t f_size, iree_hal_dim_t kh_size,
    iree_hal_dim_t kw_size, iree_hal_dim_t sh_size, iree_hal_dim_t sw_size,
    iree_hal_dim_t dh_size, iree_hal_dim_t dw_size,
    iree_hal_element_type_t input_type, iree_hal_element_type_t kernel_type,
    iree_hal_element_type_t acc_type, iree_byte_span_t input_contents,
    iree_byte_span_t kernel_contents, iree_byte_span_t acc_contents,
    iree_byte_span_t result_contents, int compute_every) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, n_size);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, c_size);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, h_size);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, w_size);

  iree_hal_dim_t oh_size = out_shape_calc(h_size, kh_size, sh_size, dh_size);
  iree_hal_dim_t ow_size = out_shape_calc(w_size, kw_size, sw_size, dw_size);

  for (iree_hal_dim_t n = 0; n < n_size; ++n) {
    for (iree_hal_dim_t oc = 0; oc < f_size; ++oc) {
      for (iree_hal_dim_t oh = 0; oh < oh_size; ++oh) {
        for (iree_hal_dim_t ow = 0; ow < ow_size; ++ow) {
          IREE_RETURN_AND_END_ZONE_IF_ERROR(
              z0, reference_conv2d_element(
                      n_size, c_size, h_size, w_size, f_size, kh_size, kw_size,
                      sh_size, sw_size, dh_size, dw_size, oh_size, ow_size,
                      input_type, kernel_type, acc_type, input_contents.data,
                      kernel_contents.data, acc_contents.data,
                      result_contents.data, n, oc, oh, ow));
        }
      }
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Conv2d comparison/logging
//===----------------------------------------------------------------------===//

typedef struct {
  iree_allocator_t host_allocator;
  iree_hal_dim_t n;   // batch dim
  iree_hal_dim_t c;   // input channels
  iree_hal_dim_t h;   // input height
  iree_hal_dim_t w;   // input width
  iree_hal_dim_t f;   // output channels
  iree_hal_dim_t kh;  // kernel height
  iree_hal_dim_t kw;  // kernel width
  iree_hal_dim_t sh;  // stride along height dim
  iree_hal_dim_t sw;  // stride along width dim
  iree_hal_dim_t dh;  // dilation along height dim
  iree_hal_dim_t dw;  // dilation along width dim
  iree_hal_element_type_t input_type;
  iree_hal_element_type_t kernel_type;
  iree_hal_element_type_t acc_type;
  iree_hal_element_type_t result_type;
  iree_byte_span_t input_contents;
  iree_byte_span_t kernel_contents;
  iree_byte_span_t acc_contents;
  iree_byte_span_t actual_contents;
  iree_byte_span_t expected_contents;
} conv2d_results_t;

static void conv2d_results_deinitialize(conv2d_results_t* results);

static iree_status_t conv2d_results_initialize(
    iree_hal_device_t* device, iree_hal_dim_t n_size, iree_hal_dim_t c_size,
    iree_hal_dim_t h_size, iree_hal_dim_t w_size, iree_hal_dim_t f_size,
    iree_hal_dim_t kh_size, iree_hal_dim_t kw_size, iree_hal_dim_t sh_size,
    iree_hal_dim_t sw_size, iree_hal_dim_t dh_size, iree_hal_dim_t dw_size,
    iree_hal_buffer_view_t* input, iree_hal_buffer_view_t* kernel,
    iree_hal_buffer_view_t* acc, iree_hal_buffer_view_t* result,
    iree_allocator_t host_allocator, conv2d_results_t* out_results) {
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_results, 0, sizeof(*out_results));
  out_results->host_allocator = host_allocator;

  out_results->n = n_size;
  out_results->c = c_size;
  out_results->h = h_size;
  out_results->w = w_size;
  out_results->f = f_size;
  out_results->kh = kh_size;
  out_results->kw = kw_size;
  out_results->sh = sh_size;
  out_results->sw = sw_size;
  out_results->dh = dh_size;
  out_results->dw = dw_size;

  out_results->input_type = iree_hal_buffer_view_element_type(input);
  out_results->kernel_type = iree_hal_buffer_view_element_type(kernel);
  out_results->acc_type = iree_hal_buffer_view_element_type(acc);
  out_results->result_type = iree_hal_buffer_view_element_type(result);

  iree_hal_buffer_t* input_buffer = iree_hal_buffer_view_buffer(input);
  iree_hal_buffer_t* kernel_buffer = iree_hal_buffer_view_buffer(kernel);
  iree_hal_buffer_t* acc_buffer = acc ? iree_hal_buffer_view_buffer(acc) : NULL;
  iree_hal_buffer_t* result_buffer = iree_hal_buffer_view_buffer(result);

  iree_status_t status = iree_ok_status();

  if (iree_status_is_ok(status)) {
    out_results->input_contents.data_length =
        iree_hal_buffer_byte_length(input_buffer);
    status = iree_allocator_malloc(host_allocator,
                                   out_results->input_contents.data_length,
                                   (void**)&out_results->input_contents.data);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_transfer_d2h(
        device, input_buffer, 0, out_results->input_contents.data,
        out_results->input_contents.data_length,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  }

  if (iree_status_is_ok(status)) {
    out_results->kernel_contents.data_length =
        iree_hal_buffer_byte_length(kernel_buffer);
    status = iree_allocator_malloc(host_allocator,
                                   out_results->kernel_contents.data_length,
                                   (void**)&out_results->kernel_contents.data);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_transfer_d2h(
        device, kernel_buffer, 0, out_results->kernel_contents.data,
        out_results->kernel_contents.data_length,
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
    conv2d_results_deinitialize(out_results);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void conv2d_results_deinitialize(conv2d_results_t* results) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(results->host_allocator, results->input_contents.data);
  iree_allocator_free(results->host_allocator, results->kernel_contents.data);
  if (!iree_byte_span_is_empty(results->acc_contents)) {
    iree_allocator_free(results->host_allocator, results->acc_contents.data);
  }
  iree_allocator_free(results->host_allocator, results->actual_contents.data);
  iree_allocator_free(results->host_allocator, results->expected_contents.data);

  IREE_TRACE_ZONE_END(z0);
}

// Helper for check_conv2d: the actual interesting part once we've
// obtained and validated the {n, f, oh, ow}_size values. On error, the first
// index is returned where the actual and expected value doesn't match. TODO:
// Add detailed logging to |file|.
static iree_status_t check_conv2d_results_impl(FILE* file,
                                               const conv2d_results_t* results,
                                               int check_every) {
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, reference_conv2d(results->n, results->c, results->h, results->w,
                           results->f, results->kh, results->kw, results->sh,
                           results->sw, results->dh, results->dw,
                           results->input_type, results->acc_type,
                           results->kernel_type, results->input_contents,
                           results->kernel_contents, results->acc_contents,
                           results->expected_contents, check_every));

  int count = 0;

  iree_hal_dim_t oh_size =
      out_shape_calc(results->h, results->kh, results->sh, results->dh);
  iree_hal_dim_t ow_size =
      out_shape_calc(results->w, results->kw, results->sw, results->dw);

  for (iree_hal_dim_t n = 0; n < results->n; ++n) {
    for (iree_hal_dim_t oc = 0; oc < results->f; ++oc) {
      for (iree_hal_dim_t oh = 0; oh < oh_size; ++oh) {
        for (iree_hal_dim_t ow = 0; ow < ow_size; ++ow) {
          if (++count < check_every) continue;
          count = 0;
          iree_hal_dim_t idx =
              convert_to_1d_index(results->f, oh_size, ow_size, n, oc, oh, ow);
          iree_test_utils_e2e_value_t actual_value =
              iree_test_utils_read_buffer_element(
                  idx, results->result_type, results->actual_contents.data);
          iree_test_utils_e2e_value_t expected_value =
              iree_test_utils_read_buffer_element(
                  idx, results->result_type, results->expected_contents.data);
          if (!iree_test_utils_result_elements_agree(actual_value,
                                                     expected_value)) {
            fprintf(
                file,
                "\n\nerror: the actual and expected result tensors disagree "
                "at n %" PRIdim ", oc %" PRIdim ", oh %" PRIdim ", ow %" PRIdim
                ".\n\n",
                n, oc, oh, ow);
            IREE_TRACE_ZONE_END(z0);
            return iree_make_status(IREE_STATUS_ABORTED);
          }
        }
      }
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Given an actual conv2d's inputs and output (all host-local), uses a
// reference conv2d implementation on the same inputs to check if the output
// is correct. On error, the first index is returned where the actual and
// expected value doesn't match. TODO: Add detailed logging to |file|.
static iree_status_t check_conv2d_results(FILE* file,
                                          const conv2d_results_t* results) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // TODO: Increase the check every param to reduce the number of comparisons.
  int check_every = 1;
  iree_status_t status = check_conv2d_results_impl(file, results, check_every);
  if (!iree_status_is_ok(status) && check_every > 1) {
    // If we got a failure with check_every>1, that didn't log a useful
    // numerical summary, as most of the reference tensor entries hadn't been
    // computed. Rerun now with check_every=1 to get that numerical logging.
    iree_status_ignore(status);
    status = check_conv2d_results_impl(file, results, 1);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// `conv2d_test` custom module
//===----------------------------------------------------------------------===//
// This uses the C++ wrapper to keep things simple. Though easier to use it's
// got additional overhead/code-size bloat that doesn't matter in a test like
// this. Making a C module builder API that removes the boilerplate there is
// TBD so this file is written in C besides this module so that we can swap it
// back to being pure C in the future.

namespace iree {

class Conv2dTestModuleState final {
 public:
  explicit Conv2dTestModuleState(iree_allocator_t host_allocator)
      : host_allocator_(host_allocator) {}
  ~Conv2dTestModuleState() = default;

  // Fills the destination span with pseudorandom values of the given
  // |element_type|. The given |seed| is passed to the pseudorandom generator.
  // The pseudorandom values are reproducible both across runs and across
  // machines.
  StatusOr<vm::ref<iree_hal_buffer_view_t>> GenerateRandom4dTensor(
      const vm::ref<iree_hal_device_t> device, int64_t dim0, int64_t dim1,
      int64_t dim2, int64_t dim3, iree_hal_element_type_t element_type,
      int32_t seed) {
    iree_hal_dim_t dims[4] = {
        (iree_hal_dim_t)dim0,
        (iree_hal_dim_t)dim1,
        (iree_hal_dim_t)dim2,
        (iree_hal_dim_t)dim3,
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
          // Generate "uniform" integer-valued numbers in the range [min,
          // max].
          int32_t min = 0;
          int32_t max = 0;
          iree_test_utils_get_min_max_for_element_type(
              callback_state.element_type, &min, &max);
          uint32_t range = (max - min + 1);
          iree_host_size_t element_byte_count =
              iree_hal_element_dense_byte_count(callback_state.element_type);
          uint8_t* data_end = span.data + span.data_length;
          uint32_t state = callback_state.seed;
          for (uint8_t* data = span.data; data < data_end;
               data += element_byte_count) {
            int32_t value =
                (int32_t)iree_test_utils_pseudorandom_range(&state, range) +
                min;
            iree_test_utils_write_element(callback_state.element_type, value,
                                          data);
          }
          return iree_ok_status();
        },
        &callback_state, &result_view));
    return std::move(result_view);
  }

  Status CheckConv2dResults(
      const vm::ref<iree_hal_device_t> device, int64_t n, int64_t c, int64_t h,
      int64_t w, int64_t f, int64_t kh, int64_t kw, int64_t sh, int64_t sw,
      int64_t dh, int64_t dw, const vm::ref<iree_hal_buffer_view_t> input,
      const vm::ref<iree_hal_buffer_view_t> kernel,
      const vm::ref<iree_hal_buffer_view_t> acc,
      const vm::ref<iree_hal_buffer_view_t> actual_result) {
    conv2d_results_t results = {};
    IREE_RETURN_IF_ERROR(conv2d_results_initialize(
        device.get(), (iree_hal_dim_t)n, (iree_hal_dim_t)c, (iree_hal_dim_t)h,
        (iree_hal_dim_t)w, (iree_hal_dim_t)f, (iree_hal_dim_t)kh,
        (iree_hal_dim_t)kw, (iree_hal_dim_t)sh, (iree_hal_dim_t)sw,
        (iree_hal_dim_t)dh, (iree_hal_dim_t)dw, input.get(), kernel.get(),
        acc.get(), actual_result.get(), host_allocator_, &results));
    iree_status_t status = check_conv2d_results(stderr, &results);
    conv2d_results_deinitialize(&results);
    return status;
  }

 private:
  iree_allocator_t host_allocator_;
};

static const vm::NativeFunction<Conv2dTestModuleState>
    kConv2dTestModuleFunctions[] = {
        vm::MakeNativeFunction("generate_random_tensor",
                               &Conv2dTestModuleState::GenerateRandom4dTensor),
        vm::MakeNativeFunction("check_conv2d_results",
                               &Conv2dTestModuleState::CheckConv2dResults),
};

struct Conv2dTestModule final : public vm::NativeModule<Conv2dTestModuleState> {
  using vm::NativeModule<Conv2dTestModuleState>::NativeModule;
  StatusOr<std::unique_ptr<Conv2dTestModuleState>> CreateState(
      iree_allocator_t host_allocator) override {
    return std::make_unique<Conv2dTestModuleState>(host_allocator);
  }
};

}  // namespace iree

static iree_status_t conv2d_test_module_create(iree_vm_instance_t* instance,
                                               iree_allocator_t host_allocator,
                                               iree_vm_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;
  auto module = std::make_unique<iree::Conv2dTestModule>(
      "conv2d_test", /*version=*/0, instance, host_allocator,
      iree::span<const iree::vm::NativeFunction<iree::Conv2dTestModuleState>>(
          iree::kConv2dTestModuleFunctions));
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

  // Run the tests. Note that some modules may be compiled for other platforms
  // and not have the required architectures for execution within them - to keep
  // the test runner dumber we gracefully fail those cases by returning success.
  iree_status_t status = iree_test_utils_load_and_run_e2e_tests(
      iree_allocator_system(), conv2d_test_module_create);
  int exit_code = EXIT_SUCCESS;
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    bool is_device_unavailable = iree_status_is_not_found(status);
    iree_status_free(status);
    exit_code = is_device_unavailable ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
