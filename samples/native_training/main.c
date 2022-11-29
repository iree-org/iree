// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/runtime/api.h"

struct State {
  float w[3];
  float b[1];
  float X[3];
  float y[1];
  float loss[1];
};

void print_state(struct State* state) {
  fprintf(stdout, "Weights:");
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(state->w); ++i) {
    fprintf(stdout, " %f", state->w[i]);
  }
  fprintf(stdout, ", Bias: %f", state->b[0]);
  fprintf(stdout, ", Loss: %f\n", state->loss[0]);
}

iree_status_t train(iree_runtime_session_t* session, struct State* state) {
  iree_status_t status = iree_ok_status();

  // Lookup the entry point function.
  iree_runtime_call_t call;
  status = iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view("module.forward"), &call);

  // Allocate buffers in device-local memory so that if the device has an
  // independent address space they live on the fast side of the fence.
  iree_hal_dim_t shape_w[1] = {IREE_ARRAYSIZE(state->w)};
  iree_hal_dim_t shape_b[0] = {};
  iree_hal_dim_t shape_X[2] = {1, IREE_ARRAYSIZE(state->X)};
  iree_hal_dim_t shape_y[1] = {IREE_ARRAYSIZE(state->y)};
  iree_hal_buffer_view_t* arg0 = NULL;
  iree_hal_buffer_view_t* arg1 = NULL;
  iree_hal_buffer_view_t* arg2 = NULL;
  iree_hal_buffer_view_t* arg3 = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_view_allocate_buffer(
        iree_runtime_session_device_allocator(session), IREE_ARRAYSIZE(shape_w),
        shape_w, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        (iree_hal_buffer_params_t){
            .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
            .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        },
        iree_make_const_byte_span(state->w, sizeof(state->w)), &arg0);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_view_allocate_buffer(
        iree_runtime_session_device_allocator(session), IREE_ARRAYSIZE(shape_b),
        shape_b, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        (iree_hal_buffer_params_t){
            .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
            .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        },
        iree_make_const_byte_span(state->b, sizeof(state->b)), &arg1);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_view_allocate_buffer(
        iree_runtime_session_device_allocator(session), IREE_ARRAYSIZE(shape_X),
        shape_X, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        (iree_hal_buffer_params_t){
            .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
            .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        },
        iree_make_const_byte_span(state->X, sizeof(state->X)), &arg2);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_view_allocate_buffer(
        iree_runtime_session_device_allocator(session), IREE_ARRAYSIZE(shape_y),
        shape_y, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        (iree_hal_buffer_params_t){
            .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
            .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        },
        iree_make_const_byte_span(state->y, sizeof(state->y)), &arg3);
  }

  // Setup call inputs with our buffers.
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_inputs_push_back_buffer_view(&call, arg0);
  }
  iree_hal_buffer_view_release(arg0);
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_inputs_push_back_buffer_view(&call, arg1);
  }
  iree_hal_buffer_view_release(arg1);
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_inputs_push_back_buffer_view(&call, arg2);
  }
  iree_hal_buffer_view_release(arg2);
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_inputs_push_back_buffer_view(&call, arg3);
  }
  iree_hal_buffer_view_release(arg3);

  // Invoke the function
  IREE_RETURN_IF_ERROR(iree_runtime_call_invoke(&call, /*flags=*/0));

  // Update weights
  iree_hal_buffer_view_t* result = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &result);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_transfer_d2h(
        iree_runtime_session_device(session),
        iree_hal_buffer_view_buffer(result), 0, &state->w, sizeof(state->w),
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  }

  // Update bias
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &result);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_transfer_d2h(
        iree_runtime_session_device(session),
        iree_hal_buffer_view_buffer(result), 0, &state->b, sizeof(state->b),
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  }

  // Update loss
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &result);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_transfer_d2h(iree_runtime_session_device(session),
                                          iree_hal_buffer_view_buffer(result),
                                          0, &state->loss, sizeof(state->loss),
                                          IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
                                          iree_infinite_timeout());
  }
  iree_hal_buffer_view_release(result);

  return status;
}

iree_status_t run_sample(iree_string_view_t bytecode_module_path,
                         iree_string_view_t driver_name, struct State* state) {
  iree_status_t status = iree_ok_status();

  //===-------------------------------------------------------------------===//
  // Instance configuration (this should be shared across sessions).
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  iree_runtime_instance_t* instance = NULL;
  if (iree_status_is_ok(status)) {
    fprintf(stdout, "Configuring IREE runtime instance and '%s' device\n",
            driver_name.data);
    status = iree_runtime_instance_create(&instance_options,
                                          iree_allocator_system(), &instance);
  }
  // TODO(#5724): move device selection into the compiled modules.
  iree_hal_device_t* device = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_runtime_instance_try_create_default_device(
        instance, driver_name, &device);
  }
  //===-------------------------------------------------------------------===//

  //===-------------------------------------------------------------------===//
  // Session configuration (one per loaded module to hold module state).
  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  iree_runtime_session_t* session = NULL;
  if (iree_status_is_ok(status)) {
    fprintf(stdout, "Creating IREE runtime session\n");
    status = iree_runtime_session_create_with_device(
        instance, &session_options, device,
        iree_runtime_instance_host_allocator(instance), &session);
  }
  iree_hal_device_release(device);

  if (iree_status_is_ok(status)) {
    fprintf(stdout, "Loading bytecode module at '%s'\n",
            bytecode_module_path.data);
    status = iree_runtime_session_append_bytecode_module_from_file(
        session, bytecode_module_path.data);
  }
  //===-------------------------------------------------------------------===//

  //===-------------------------------------------------------------------===//
  if (iree_status_is_ok(status)) {
    fprintf(stdout, "Training...\n");
    print_state(state);
    for (int i = 0; i < 10; i++) {
      status = train(session, state);
      print_state(state);
      if (!iree_status_is_ok(status)) {
        break;
      }
    }
  }
  //===-------------------------------------------------------------------===//

  //===-------------------------------------------------------------------===//
  // Cleanup.
  iree_runtime_session_release(session);
  iree_runtime_instance_release(instance);
  //===-------------------------------------------------------------------===//

  return status;
}

int main(int argc, char** argv) {
  // Parse args
  if (argc < 2) {
    fprintf(stderr,
            "Usage: native-training </path/to/native_training.vmfb> "
            "[<driver_name>]\n");
    fprintf(stderr, "  (See the README for this sample for details)\n ");
    return -1;
  }
  iree_string_view_t bytecode_module_path = iree_make_cstring_view(argv[1]);
  iree_string_view_t driver_name;
  if (argc >= 3) {
    driver_name = iree_make_cstring_view(argv[2]);
  } else {
    driver_name = iree_make_cstring_view("local-sync");
  }

  // Run training
  struct State state = {
      {4.0f, 4.0f, 5.0f},  // w
      {2.0f},              // b
      {1.0f, 1.0f, 1.0f},  // X
      {14.0f},             // y
      {1.0f},              // loss
  };
  iree_status_t result = run_sample(bytecode_module_path, driver_name, &state);
  if (!iree_status_is_ok(result)) {
    fprintf(stdout, "Failed!\n");
    iree_status_fprint(stderr, result);
    iree_status_ignore(result);
    return -1;
  }

  // Validate result
  if (*state.loss > 0.1f) {
    fprintf(stdout, "Loss unexpectedly high\n");
    return -1;
  }

  fprintf(stdout, "\nSuccess!\n");
  return 0;
}
