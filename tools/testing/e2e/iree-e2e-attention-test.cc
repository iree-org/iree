// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <float.h>
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
// Function to allocate and initialize tensors
float* allocate_tensor(int dim1, int dim2, int dim3) {
  const int size = dim1 * dim2 * dim3;
  float* tensor = (float*)malloc(size * sizeof(float));
  for (int i = 0; i < size; ++i) {
    tensor[i] = 0.0f;
  }
  return tensor;
}

// Function to free allocated tensors
void free_tensor(float* tensor) {
  if (tensor != nullptr) free(tensor);
}

// Function to calculate 1D index for a 3D array
int index_3d(int i, int j, int k, int dim2, int dim3) {
  return i * dim2 * dim3 + j * dim3 + k;
}

static void reference_attention_f32_f32_f32_f32(
    iree_hal_dim_t M, iree_hal_dim_t K1, iree_hal_dim_t K2, iree_hal_dim_t N,
    iree_hal_dim_t B, const float* query_data, const float* key_data,
    const float* value_data, float* result_data, iree_hal_dim_t b,
    float* Attention) {
  // Compute Q * K^T
  for (int m = 0; m < M; ++m) {
    for (int k2 = 0; k2 < K2; ++k2) {
      float sum = 0.0;
      for (int k1 = 0; k1 < K1; ++k1) {
        int q_idx = index_3d(b, m, k1, M, K1);
        int k_idx = index_3d(b, k2, k1, K2, K1);

        sum += query_data[q_idx] * key_data[k_idx];
      }
      int att_idx = index_3d(0, m, k2, M, K2);
      Attention[att_idx] = sum / sqrt(K1);  // Scale by sqrt(K1)
    }
  }

  // Compute softmax on Attention
  for (int m = 0; m < M; ++m) {
    // Find the maximum value for the current sequence
    float max_val = -FLT_MAX;
    for (int k2 = 0; k2 < K2; ++k2) {
      int att_idx = index_3d(0, m, k2, M, K2);
      max_val = iree_max(max_val, Attention[att_idx]);
    }

    // Calculate the softmax denominator
    float sum = 0.0f;
    for (int k2 = 0; k2 < K2; ++k2) {
      int att_idx = index_3d(0, m, k2, M, K2);
      sum += exp(Attention[att_idx] - max_val);
    }

    // Apply softmax
    for (int k2 = 0; k2 < K2; ++k2) {
      int att_idx = index_3d(0, m, k2, M, K2);
      Attention[att_idx] = exp(Attention[att_idx]) / sum;
    }
  }

  // Compute Attention * V
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float sum = 0.0;
      for (int k2 = 0; k2 < K2; ++k2) {
        int att_idx = index_3d(0, m, k2, M, K2);
        int v_idx = index_3d(b, k2, n, K2, N);
        sum += Attention[att_idx] * value_data[v_idx];
      }
      int o_idx = index_3d(b, m, n, M, N);
      result_data[o_idx] = sum;
    }
  }
}

static iree_status_t reference_attention_element(
    iree_hal_dim_t M, iree_hal_dim_t K1, iree_hal_dim_t K2, iree_hal_dim_t N,
    iree_hal_dim_t B, iree_hal_element_type_t query_elem_type,
    iree_hal_element_type_t key_elem_type,
    iree_hal_element_type_t value_elem_type, void* query_data, void* key_data,
    void* value_data, void* actual_data, void* result_data, iree_hal_dim_t b,
    float* Attention) {
  if (query_elem_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32 &&
      key_elem_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32 &&
      value_elem_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32) {
    reference_attention_f32_f32_f32_f32(
        M, K1, K2, N, B, (const float*)query_data, (const float*)key_data,
        (const float*)value_data, (float*)result_data, b, Attention);

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
    iree_hal_dim_t B, iree_hal_dim_t M, iree_hal_dim_t K1, iree_hal_dim_t K2,
    iree_hal_dim_t N, iree_hal_element_type_t query_elem_type,
    iree_hal_element_type_t key_elem_type,
    iree_hal_element_type_t value_elem_type, iree_byte_span_t query_contents,
    iree_byte_span_t key_contents, iree_byte_span_t value_contents,
    iree_byte_span_t actual_contents, iree_byte_span_t result_contents,
    int compute_every) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, B);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, M);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, K1);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, K2);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, N);

  iree_host_size_t count = 0;
  float* Attention = allocate_tensor(1, M, K2);
  for (iree_hal_dim_t b = 0; b < B; ++b) {
    if (++count < compute_every) continue;
    count = 0;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        reference_attention_element(
            M, K1, K2, N, B, query_elem_type, key_elem_type, value_elem_type,
            query_contents.data, key_contents.data, value_contents.data,
            actual_contents.data, result_contents.data, b, Attention));
  }
  free_tensor(Attention);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
//===----------------------------------------------------------------------===//
// Attention comparison/logging
//===----------------------------------------------------------------------===//

typedef struct {
  iree_allocator_t host_allocator;
  iree_hal_dim_t b;
  iree_hal_dim_t m;
  iree_hal_dim_t k1;
  iree_hal_dim_t k2;
  iree_hal_dim_t n;
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
    iree_hal_device_t* device, iree_hal_dim_t b_size, iree_hal_dim_t m_size,
    iree_hal_dim_t k1_size, iree_hal_dim_t k2_size, iree_hal_dim_t n_size,
    iree_hal_buffer_view_t* query, iree_hal_buffer_view_t* key,
    iree_hal_buffer_view_t* value, iree_hal_buffer_view_t* result,
    iree_allocator_t host_allocator, attention_results_t* out_results) {
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_results, 0, sizeof(*out_results));
  out_results->host_allocator = host_allocator;

  out_results->b = b_size;
  out_results->m = m_size;
  out_results->k1 = k1_size;
  out_results->k2 = k2_size;
  out_results->n = n_size;

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

// Helper for check_attention_results: the actual interesting part once we've
// obtained and validated the {b,m,k1,k2,n}_size values. On error, detailed
// logging is written to |file| if it is not NULL.
static iree_status_t check_attention_results_impl(
    FILE* file, const attention_results_t* results, int check_every) {
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, reference_attention(results->b, results->m, results->k1, results->k2,
                              results->n, results->query_elem_type,
                              results->key_elem_type, results->value_elem_type,
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
    iree_hal_dim_t dims[3] = {
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
      const vm::ref<iree_hal_device_t> device, int64_t b, int64_t m, int64_t k1,
      int64_t k2, int64_t n, const vm::ref<iree_hal_buffer_view_t> query,
      const vm::ref<iree_hal_buffer_view_t> key,
      const vm::ref<iree_hal_buffer_view_t> value,
      const vm::ref<iree_hal_buffer_view_t> actual_result) {
    attention_results_t results = {};
    IREE_RETURN_IF_ERROR(attention_results_initialize(
        device.get(), (iree_hal_dim_t)b, (iree_hal_dim_t)m, (iree_hal_dim_t)k1,
        (iree_hal_dim_t)k2, (iree_hal_dim_t)n, query.get(), key.get(),
        value.get(), actual_result.get(), host_allocator_, &results));
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
            "check_attention_results",
            &AttentionTestModuleState::CheckAttentionResults),
};

struct AttentionTestModule final
    : public vm::NativeModule<AttentionTestModuleState> {
  using vm::NativeModule<AttentionTestModuleState>::NativeModule;
  StatusOr<std::unique_ptr<AttentionTestModuleState>> CreateState(
      iree_allocator_t host_allocator) override {
    return std::make_unique<AttentionTestModuleState>(host_allocator);
  }
  StatusOr<std::unique_ptr<AttentionTestModuleState>> ForkState(
      AttentionTestModuleState* parent_state,
      iree_allocator_t host_allocator) override {
    return std::make_unique<AttentionTestModuleState>(host_allocator);
  }
};

}  // namespace iree

static iree_status_t attention_test_module_create(
    iree_vm_instance_t* instance, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;
  auto module = std::make_unique<iree::AttentionTestModule>(
      "attention_test", /*version=*/0, instance, host_allocator,
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
      iree_allocator_system(), attention_test_module_create);
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
