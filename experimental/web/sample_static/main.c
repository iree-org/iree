// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <float.h>
#include <stdio.h>

#include "iree/hal/buffer_transfer.h"
#include "iree/runtime/api.h"
#include "iree/vm/bytecode/module.h"
#include "mnist_bytecode.h"

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

typedef struct iree_sample_state_t iree_sample_state_t;

// TODO(scotttodd): figure out error handling and state management
//     * out_state and return status would make sense, but emscripten...
iree_sample_state_t* setup_sample();
void cleanup_sample(iree_sample_state_t* state);

int run_sample(iree_sample_state_t* state, float* image_data);

//===----------------------------------------------------------------------===//
// Implementation
//===----------------------------------------------------------------------===//

extern iree_status_t create_device_with_static_loader(
    iree_allocator_t host_allocator, iree_hal_device_t** out_device);

typedef struct iree_sample_state_t {
  iree_runtime_instance_t* instance;
  iree_hal_device_t* device;
  iree_runtime_session_t* session;
  iree_vm_module_t* module;
  iree_runtime_call_t call;
} iree_sample_state_t;

iree_status_t create_bytecode_module(iree_vm_instance_t* instance,
                                     iree_vm_module_t** out_module) {
  const struct iree_file_toc_t* module_file_toc = iree_static_mnist_create();
  iree_const_byte_span_t module_data =
      iree_make_const_byte_span(module_file_toc->data, module_file_toc->size);
  return iree_vm_bytecode_module_create(instance, module_data,
                                        iree_allocator_null(),
                                        iree_allocator_system(), out_module);
}

iree_sample_state_t* setup_sample() {
  iree_sample_state_t* state = NULL;
  iree_status_t status = iree_allocator_malloc(
      iree_allocator_system(), sizeof(iree_sample_state_t), (void**)&state);

  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  // Note: no call to iree_runtime_instance_options_use_all_available_drivers().

  if (iree_status_is_ok(status)) {
    status = iree_runtime_instance_create(
        &instance_options, iree_allocator_system(), &state->instance);
  }

  if (iree_status_is_ok(status)) {
    status = create_device_with_static_loader(iree_allocator_system(),
                                              &state->device);
  }

  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_create_with_device(
        state->instance, &session_options, state->device,
        iree_runtime_instance_host_allocator(state->instance), &state->session);
  }

  if (iree_status_is_ok(status)) {
    status = create_bytecode_module(
        iree_runtime_instance_vm_instance(state->instance), &state->module);
  }
  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_append_module(state->session, state->module);
  }

  const char kMainFunctionName[] = "module.predict";
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_initialize_by_name(
        state->session, iree_make_cstring_view(kMainFunctionName),
        &state->call);
  }

  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    cleanup_sample(state);
    return NULL;
  }

  return state;
}

void cleanup_sample(iree_sample_state_t* state) {
  iree_runtime_call_deinitialize(&state->call);

  // Cleanup session and instance.
  iree_hal_device_release(state->device);
  iree_runtime_session_release(state->session);
  iree_runtime_instance_release(state->instance);
  iree_vm_module_release(state->module);

  free(state);
}

int run_sample(iree_sample_state_t* state, float* image_data) {
  iree_status_t status = iree_ok_status();

  iree_runtime_call_reset(&state->call);

  iree_hal_buffer_view_t* arg_buffer_view = NULL;
  iree_hal_dim_t buffer_shape[] = {1, 28, 28, 1};
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_view_allocate_buffer_copy(
        state->device, iree_hal_device_allocator(state->device),
        IREE_ARRAYSIZE(buffer_shape), buffer_shape,
        IREE_HAL_ELEMENT_TYPE_FLOAT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        (iree_hal_buffer_params_t){
            .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
            .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
        },
        iree_make_const_byte_span((void*)image_data, sizeof(float) * 28 * 28),
        &arg_buffer_view);
  }
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_inputs_push_back_buffer_view(&state->call,
                                                            arg_buffer_view);
  }
  iree_hal_buffer_view_release(arg_buffer_view);

  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_invoke(&state->call, /*flags=*/0);
  }

  // Get the result buffers from the invocation.
  iree_hal_buffer_view_t* ret_buffer_view = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_outputs_pop_front_buffer_view(&state->call,
                                                             &ret_buffer_view);
  }

  // Read back the results. The output of the mnist model is a 1x10 prediction
  // confidence values for each digit in [0, 9].
  float predictions[1 * 10] = {0.0f};
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_transfer_d2h(
        state->device, iree_hal_buffer_view_buffer(ret_buffer_view), 0,
        predictions, sizeof(predictions), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout());
  }
  iree_hal_buffer_view_release(ret_buffer_view);

  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return -1;
  }

  // Get the highest index from the output.
  float result_val = FLT_MIN;
  int result_idx = 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(predictions); ++i) {
    if (predictions[i] > result_val) {
      result_val = predictions[i];
      result_idx = i;
    }
  }
  fprintf(stdout,
          "Prediction: %d, confidences: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, "
          "%.2f, %.2f, %.2f, %.2f]\n",
          result_idx, predictions[0], predictions[1], predictions[2],
          predictions[3], predictions[4], predictions[5], predictions[6],
          predictions[7], predictions[8], predictions[9]);
  return result_idx;
}
