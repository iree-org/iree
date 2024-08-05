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
// Reference Attention
//===----------------------------------------------------------------------===//

// Helper for reference_attention.
// Computes one element in the result matrix.

static void reference_attention_f32_f32_f32_f32(iree_hal_dim_t seqLen_size,
                                                iree_hal_dim_t headDim_size,
                                                const float* query_data,
                                                const float* key_data,
                                                const float* value_data,
                                                float* result_data) {
  float** scores = (float**)malloc(seqLen_size * sizeof(float*));
  float** attention = (float**)malloc(seqLen_size * sizeof(float*));
  for (iree_hal_dim_t i = 0; i < seqLen_size; ++i) {
    scores[i] = (float*)malloc(seqLen_size * sizeof(float));
    attention[i] = (float*)malloc(seqLen_size * sizeof(float));
  }

  // Calculate the scores
  for (int i = 0; i < seqLen_size; ++i) {
    for (int j = 0; j < seqLen_size; ++j) {
      scores[i][j] = 0.0f;
      for (int k = 0; k < headDim_size; ++k) {
        scores[i][j] +=
            query_data[i * headDim_size + k] * key_data[j * headDim_size + k];
      }
      scores[i][j] /= sqrt(headDim_size);
    }
  }

  // Apply softmax
  for (int i = 0; i < seqLen_size; ++i) {
    float sum = 0.0f;
    for (int j = 0; j < seqLen_size; ++j) {
      attention[i][j] = exp(scores[i][j]);
      sum += attention[i][j];
    }
    for (int j = 0; j < seqLen_size; ++j) {
      attention[i][j] /= sum;
    }
  }

  // Calculate the output
  for (int i = 0; i < seqLen_size; ++i) {
    for (int j = 0; j < headDim_size; ++j) {
      result_data[i * headDim_size + j] = 0.0f;
      for (int k = 0; k < seqLen_size; ++k) {
        result_data[i * headDim_size + j] +=
            attention[i][k] * value_data[k * headDim_size + j];
      }
    }
  }

  for (iree_hal_dim_t i = 0; i < seqLen_size; ++i) {
    free(scores[i]);
    free(attention[i]);
  }
  free(scores);
  free(attention);
}

static iree_status_t reference_attention_element(
    iree_hal_dim_t numHeads_size, iree_hal_dim_t seqLen_size,
    iree_hal_dim_t headDim_size, iree_hal_element_type_t query_elem_type,
    iree_hal_element_type_t key_elem_type,
    iree_hal_element_type_t value_elem_type,
    iree_hal_element_type_t result_elem_type, void* query_data, void* key_data,
    void* value_data, void* actual_data, void* result_data, iree_hal_dim_t n) {
  if (query_elem_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32 &&
      key_elem_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32 &&
      value_elem_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32) {
    reference_attention_f32_f32_f32_f32(
        seqLen_size, headDim_size, (const float*)query_data,
        (const float*)key_data, (const float*)value_data, (float*)result_data);

  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unhandled combination of element types in attention");
  }
  return iree_ok_status();
}

// Reference attention implementation, used to compare attention results
// against.
static iree_status_t reference_attention(
    iree_hal_dim_t numHeads_size, iree_hal_dim_t seqLen_size,
    iree_hal_dim_t headDim_size, iree_hal_element_type_t query_elem_type,
    iree_hal_element_type_t key_elem_type,
    iree_hal_element_type_t value_elem_type,
    iree_hal_element_type_t result_elem_type, iree_byte_span_t query_contents,
    iree_byte_span_t key_contents, iree_byte_span_t value_contents,
    iree_byte_span_t actual_contents, iree_byte_span_t result_contents,
    int compute_every) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, numHeads_size);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, seqLen_size);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, headDim_size);

  iree_host_size_t count = 0;
  for (iree_hal_dim_t n = 0; n < numHeads_size; ++n) {
    if (++count < compute_every) continue;
    count = 0;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, reference_attention_element(
                numHeads_size, seqLen_size, headDim_size, query_elem_type,
                key_elem_type, value_elem_type, result_elem_type,
                query_contents.data, key_contents.data, value_contents.data,
                actual_contents.data, result_contents.data, n));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
//===----------------------------------------------------------------------===//
// Attention comparison/logging
//===----------------------------------------------------------------------===//

typedef struct {
  iree_allocator_t host_allocator;
  iree_hal_dim_t numHeads;
  iree_hal_dim_t seqLen;
  iree_hal_dim_t headDim;
  iree_hal_element_type_t query_elem_type;
  iree_hal_element_type_t key_elem_type;
  iree_hal_element_type_t value_elem_type;
  iree_hal_element_type_t result_elem_type;
  iree_byte_span_t query_contents;
  iree_byte_span_t key_contents;
  iree_byte_span_t value_contents;
  iree_byte_span_t actual_contents;
  iree_byte_span_t expected_contents;
} attention_results_t;

static void attention_results_deinitialize(attention_results_t* results);

static iree_status_t attention_results_initialize(
    iree_hal_device_t* device, iree_hal_dim_t numHeads_size,
    iree_hal_dim_t seqLen_size, iree_hal_dim_t headDim_size,
    iree_hal_buffer_view_t* query, iree_hal_buffer_view_t* key,
    iree_hal_buffer_view_t* value, iree_hal_buffer_view_t* result,
    iree_allocator_t host_allocator, attention_results_t* out_results) {
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_results, 0, sizeof(*out_results));
  out_results->host_allocator = host_allocator;

  out_results->numHeads = numHeads_size;
  out_results->seqLen = seqLen_size;

  out_results->query_elem_type = iree_hal_buffer_view_element_type(query);
  out_results->key_elem_type = iree_hal_buffer_view_element_type(key);
  out_results->value_elem_type = iree_hal_buffer_view_element_type(value);
  out_results->result_elem_type = iree_hal_buffer_view_element_type(result);

  iree_hal_buffer_t* query_buffer = iree_hal_buffer_view_buffer(query);
  iree_hal_buffer_t* key_buffer = iree_hal_buffer_view_buffer(key);
  iree_hal_buffer_t* value_buffer = iree_hal_buffer_view_buffer(value);
  iree_hal_buffer_t* result_buffer = iree_hal_buffer_view_buffer(result);

  iree_status_t status = iree_ok_status();

  if (iree_status_is_ok(status)) {
    out_results->query_contents.data_length =
        iree_hal_buffer_byte_length(query_buffer);
    status = iree_allocator_malloc(host_allocator,
                                   out_results->query_contents.data_length,
                                   (void**)&out_results->query_contents.data);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_transfer_d2h(
        device, query_buffer, 0, out_results->query_contents.data,
        out_results->query_contents.data_length,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  }
  if (iree_status_is_ok(status)) {
    out_results->key_contents.data_length =
        iree_hal_buffer_byte_length(key_buffer);
    status = iree_allocator_malloc(host_allocator,
                                   out_results->key_contents.data_length,
                                   (void**)&out_results->key_contents.data);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_transfer_d2h(
        device, key_buffer, 0, out_results->key_contents.data,
        out_results->key_contents.data_length,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  }
  if (iree_status_is_ok(status)) {
    out_results->value_contents.data_length =
        iree_hal_buffer_byte_length(value_buffer);
    status = iree_allocator_malloc(host_allocator,
                                   out_results->value_contents.data_length,
                                   (void**)&out_results->value_contents.data);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_device_transfer_d2h(
        device, value_buffer, 0, out_results->value_contents.data,
        out_results->value_contents.data_length,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
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
    attention_results_deinitialize(out_results);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void attention_results_deinitialize(attention_results_t* results) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_free(results->host_allocator, results->query_contents.data);
  iree_allocator_free(results->host_allocator, results->key_contents.data);
  iree_allocator_free(results->host_allocator, results->value_contents.data);
  iree_allocator_free(results->host_allocator, results->actual_contents.data);
  iree_allocator_free(results->host_allocator, results->expected_contents.data);
  IREE_TRACE_ZONE_END(z0);
}

// Helper for check_matmul_results: the actual interesting part once we've
// obtained and validated the {m,k,n}_size values. On error, detailed logging is
// written to |file| if it is not NULL.
static iree_status_t check_attention_results_impl(
    FILE* file, const attention_results_t* results, int check_every) {
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      reference_attention(results->numHeads, results->seqLen, results->headDim,
                          results->query_elem_type, results->key_elem_type,
                          results->value_elem_type, results->result_elem_type,
                          results->query_contents, results->key_contents,
                          results->value_contents, results->actual_contents,
                          results->expected_contents, check_every));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Given an actual attention's inputs and output (all host-local), uses a
// reference attention implementation on the same inputs to check if the output
// is correct. On error, detailed logging is written to |file| if it is not
// NULL.
static iree_status_t check_attention_results(
    FILE* file, const attention_results_t* results) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // TODO: Increase the check every param to reduce the number of comparisons.
  int check_every = 1;
  iree_status_t status =
      check_attention_results_impl(file, results, check_every);
  if (!iree_status_is_ok(status) && check_every > 1) {
    // If we got a failure with check_every>1, that didn't log a useful
    // numerical summary, as most of the reference matrix entries hadn't been
    // computed. Rerun now with check_every=1 to get that numerical logging.
    iree_status_ignore(status);
    status = check_attention_results_impl(file, results, 1);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// `attention_test` custom module
//===----------------------------------------------------------------------===//
// This uses the C++ wrapper to keep things simple. Though easier to use it's
// got additional overhead/code-size bloat that doesn't matter in a test like
// this. Making a C module builder API that removes the boilerplate there is TBD
// so this file is written in C besides this module so that we can swap it back
// to being pure C in the future.

namespace iree {

class AttentionTestModuleState final {
 public:
  explicit AttentionTestModuleState(iree_allocator_t host_allocator)
      : host_allocator_(host_allocator) {}
  ~AttentionTestModuleState() = default;

  // Fills the destination span with pseudorandom values of the given
  // |element_type|. The given |seed| is passed to the pseudorandom generator.
  // The pseudorandom values are reproducible both across runs and across
  // machines.
  StatusOr<vm::ref<iree_hal_buffer_view_t>> GenerateRandom3dTensor(
      const vm::ref<iree_hal_device_t> device, int64_t dim0, int64_t dim1,
      int64_t dim2, iree_hal_element_type_t element_type, int32_t seed) {
    iree_hal_dim_t dims[4] = {
        (iree_hal_dim_t)dim0,
        (iree_hal_dim_t)dim1,
        (iree_hal_dim_t)dim2,
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

  Status CheckAttentionResults(
      const vm::ref<iree_hal_device_t> device, int64_t numHeads, int64_t seqLen,
      int64_t headDim, const vm::ref<iree_hal_buffer_view_t> query,
      const vm::ref<iree_hal_buffer_view_t> key,
      const vm::ref<iree_hal_buffer_view_t> value,
      const vm::ref<iree_hal_buffer_view_t> actual_result) {
    attention_results_t results = {};
    IREE_RETURN_IF_ERROR(attention_results_initialize(
        device.get(), (iree_hal_dim_t)numHeads, (iree_hal_dim_t)seqLen,
        (iree_hal_dim_t)headDim, query.get(), key.get(), value.get(),
        actual_result.get(), host_allocator_, &results));
    iree_status_t status = check_attention_results(stderr, &results);
    attention_results_deinitialize(&results);
    return status;
  }

 private:
  iree_allocator_t host_allocator_;
};

static const vm::NativeFunction<AttentionTestModuleState>
    kAttentionTestModuleFunctions[] = {
        vm::MakeNativeFunction(
            "generate_random_tensor",
            &AttentionTestModuleState::GenerateRandom3dTensor),
        vm::MakeNativeFunction(
            "check_fa2_results",
            &AttentionTestModuleState::CheckAttentionResults),
};

struct AttentionTestModule final
    : public vm::NativeModule<AttentionTestModuleState> {
  using vm::NativeModule<AttentionTestModuleState>::NativeModule;
  StatusOr<std::unique_ptr<AttentionTestModuleState>> CreateState(
      iree_allocator_t host_allocator) override {
    return std::make_unique<AttentionTestModuleState>(host_allocator);
  }
};

}  // namespace iree

static iree_status_t fa2_test_module_create(iree_vm_instance_t* instance,
                                            iree_allocator_t host_allocator,
                                            iree_vm_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;
  auto module = std::make_unique<iree::AttentionTestModule>(
      "fa2_test", /*version=*/0, instance, host_allocator,
      iree::span<
          const iree::vm::NativeFunction<iree::AttentionTestModuleState>>(
          iree::kAttentionTestModuleFunctions));
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
      iree_allocator_system(), fa2_test_module_create);
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