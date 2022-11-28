// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/runtime/api.h"

float w[] = {4.0f, 4.0f, 5.0f};
float b[] = {2.0f};
float X[] = {1.0f, 1.0f, 1.0f};
float y[] = {14.0f};
float loss[] = {1.0f};

void print_state() {
  fprintf(stdout, "Weights:");
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(w); ++i) {
    fprintf(stdout, " %f", w[i]);
  }
  fprintf(stdout, ", Bias: %f", b[0]);
  fprintf(stdout, ", Loss: %f\n", loss[0]);
}

iree_status_t train(iree_runtime_session_t* session) {
  // Lookup the entry point function.
  iree_runtime_call_t call;
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view("module.forward"), &call));

  // Allocate buffers in device-local memory so that if the device has an
  // independent address space they live on the fast side of the fence.
  iree_hal_dim_t shape_w[1] = {IREE_ARRAYSIZE(w)};
  iree_hal_dim_t shape_b[0] = {};
  iree_hal_dim_t shape_X[2] = {1, IREE_ARRAYSIZE(X)};
  iree_hal_dim_t shape_y[1] = {IREE_ARRAYSIZE(y)};
  iree_hal_buffer_view_t* arg0 = NULL;
  iree_hal_buffer_view_t* arg1 = NULL;
  iree_hal_buffer_view_t* arg2 = NULL;
  iree_hal_buffer_view_t* arg3 = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer(
      iree_runtime_session_device_allocator(session), IREE_ARRAYSIZE(shape_w),
      shape_w, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      },
      iree_make_const_byte_span(w, sizeof(w)), &arg0));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer(
      iree_runtime_session_device_allocator(session), IREE_ARRAYSIZE(shape_b),
      shape_b, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      },
      iree_make_const_byte_span(b, sizeof(b)), &arg1));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer(
      iree_runtime_session_device_allocator(session), IREE_ARRAYSIZE(shape_X),
      shape_X, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      },
      iree_make_const_byte_span(X, sizeof(X)), &arg2));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer(
      iree_runtime_session_device_allocator(session), IREE_ARRAYSIZE(shape_y),
      shape_y, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      },
      iree_make_const_byte_span(y, sizeof(y)), &arg3));

  // Setup call inputs with our buffers.
  IREE_RETURN_IF_ERROR(
      iree_runtime_call_inputs_push_back_buffer_view(&call, arg0));
  iree_hal_buffer_view_release(arg0);
  IREE_RETURN_IF_ERROR(
      iree_runtime_call_inputs_push_back_buffer_view(&call, arg1));
  iree_hal_buffer_view_release(arg1);
  IREE_RETURN_IF_ERROR(
      iree_runtime_call_inputs_push_back_buffer_view(&call, arg2));
  iree_hal_buffer_view_release(arg2);
  IREE_RETURN_IF_ERROR(
      iree_runtime_call_inputs_push_back_buffer_view(&call, arg3));
  iree_hal_buffer_view_release(arg3);

  // Invoke the function
  IREE_RETURN_IF_ERROR(iree_runtime_call_invoke(&call, /*flags=*/0));

  // Update weights
  iree_hal_buffer_view_t* result = NULL;
  IREE_RETURN_IF_ERROR(
      iree_runtime_call_outputs_pop_front_buffer_view(&call, &result));
  IREE_RETURN_IF_ERROR(iree_hal_device_transfer_d2h(
      iree_runtime_session_device(session), iree_hal_buffer_view_buffer(result),
      0, &w, sizeof(w), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));

  // Update bias
  IREE_RETURN_IF_ERROR(
      iree_runtime_call_outputs_pop_front_buffer_view(&call, &result));
  IREE_RETURN_IF_ERROR(iree_hal_device_transfer_d2h(
      iree_runtime_session_device(session), iree_hal_buffer_view_buffer(result),
      0, &b, sizeof(b), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));

  // Update loss
  IREE_RETURN_IF_ERROR(
      iree_runtime_call_outputs_pop_front_buffer_view(&call, &result));
  IREE_RETURN_IF_ERROR(iree_hal_device_transfer_d2h(
      iree_runtime_session_device(session), iree_hal_buffer_view_buffer(result),
      0, &loss, sizeof(loss), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));
  iree_hal_buffer_view_release(result);

  return iree_ok_status();
}

iree_status_t run_sample(iree_string_view_t bytecode_module_path,
                         iree_string_view_t driver_name) {
  iree_status_t status = iree_ok_status();

  //===-------------------------------------------------------------------===//
  // Instance configuration (this should be shared across sessions).
  fprintf(stdout, "Configuring IREE runtime instance and '%s' device\n",
          driver_name.data);
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  iree_runtime_instance_t* instance = NULL;
  if (iree_status_is_ok(status)) {
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
  fprintf(stdout, "Creating IREE runtime session\n");
  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  iree_runtime_session_t* session = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_create_with_device(
        instance, &session_options, device,
        iree_runtime_instance_host_allocator(instance), &session);
  }
  iree_hal_device_release(device);

  fprintf(stdout, "Loading bytecode module at '%s'\n",
          bytecode_module_path.data);
  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_append_bytecode_module_from_file(
        session, bytecode_module_path.data);
  }
  //===-------------------------------------------------------------------===//

  //===-------------------------------------------------------------------===//
  fprintf(stdout, "Training...\n");

  print_state();
  for (int i = 0; i < 10; i++) {
    IREE_RETURN_IF_ERROR(train(session));
    print_state();
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

  iree_status_t result = run_sample(bytecode_module_path, driver_name);
  if (!iree_status_is_ok(result)) {
    fprintf(stdout, "Failed!\n");
    iree_status_fprint(stderr, result);
    iree_status_ignore(result);
    return -1;
  }

  if (*loss > 0.1f) {
    fprintf(stdout, "Loss unexpectedly high\n");
    return -1;
  }

  fprintf(stdout, "\nSuccess!\n");
  return 0;
}
