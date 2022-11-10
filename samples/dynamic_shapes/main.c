// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/runtime/api.h"

iree_status_t reduce_sum_1d(iree_runtime_session_t* session, const int* values,
                            int values_length, int* out_result) {
  iree_runtime_call_t call;
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view("module.reduce_sum_1d"), &call));

  iree_hal_buffer_view_t* arg0 = NULL;
  const iree_hal_dim_t arg0_shape[1] = {(iree_hal_dim_t)values_length};

  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_view_allocate_buffer(
        iree_runtime_session_device_allocator(session),
        IREE_ARRAYSIZE(arg0_shape), arg0_shape, IREE_HAL_ELEMENT_TYPE_SINT_32,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        (iree_hal_buffer_params_t){
            .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
            .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        },
        iree_make_const_byte_span((void*)values, sizeof(int) * values_length),
        &arg0);
  }
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_inputs_push_back_buffer_view(&call, arg0);
  }
  iree_hal_buffer_view_release(arg0);
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_invoke(&call, /*flags=*/0);
  }

  iree_hal_buffer_view_t* buffer_view = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_runtime_call_outputs_pop_front_buffer_view(&call, &buffer_view);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_transfer_d2h(
        iree_runtime_session_device(session),
        iree_hal_buffer_view_buffer(buffer_view), 0, out_result,
        sizeof(*out_result), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout());
  }
  iree_hal_buffer_view_release(buffer_view);

  iree_runtime_call_deinitialize(&call);
  return status;
}

iree_status_t reduce_sum_2d(iree_runtime_session_t* session, const int* values,
                            size_t values_length,
                            iree_hal_buffer_view_t** out_buffer_view) {
  iree_runtime_call_t call;
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view("module.reduce_sum_2d"), &call));

  iree_hal_buffer_view_t* arg0 = NULL;
  const iree_hal_dim_t arg0_shape[2] = {values_length / 3, 3};

  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_view_allocate_buffer(
        iree_runtime_session_device_allocator(session),
        IREE_ARRAYSIZE(arg0_shape), arg0_shape, IREE_HAL_ELEMENT_TYPE_SINT_32,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        (iree_hal_buffer_params_t){
            .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
            .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        },
        iree_make_const_byte_span((void*)values, sizeof(int) * values_length),
        &arg0);
  }
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_inputs_push_back_buffer_view(&call, arg0);
  }
  iree_hal_buffer_view_release(arg0);
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_invoke(&call, /*flags=*/0);
  }

  if (iree_status_is_ok(status)) {
    status =
        iree_runtime_call_outputs_pop_front_buffer_view(&call, out_buffer_view);
  }

  iree_runtime_call_deinitialize(&call);
  return status;
}

iree_status_t add_one(iree_runtime_session_t* session, const int* values,
                      size_t values_length,
                      iree_hal_buffer_view_t** out_buffer_view) {
  iree_runtime_call_t call;
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view("module.add_one"), &call));

  iree_hal_buffer_view_t* arg0 = NULL;
  const iree_hal_dim_t arg0_shape[1] = {values_length};

  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_view_allocate_buffer(
        iree_runtime_session_device_allocator(session),
        IREE_ARRAYSIZE(arg0_shape), arg0_shape, IREE_HAL_ELEMENT_TYPE_SINT_32,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        (iree_hal_buffer_params_t){
            .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
            .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        },
        iree_make_const_byte_span((void*)values, sizeof(int) * values_length),
        &arg0);
  }
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_inputs_push_back_buffer_view(&call, arg0);
  }
  iree_hal_buffer_view_release(arg0);
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_invoke(&call, /*flags=*/0);
  }

  if (iree_status_is_ok(status)) {
    status =
        iree_runtime_call_outputs_pop_front_buffer_view(&call, out_buffer_view);
  }

  iree_runtime_call_deinitialize(&call);
  return status;
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
  // Call the exported sample functions with some test inputs
  fprintf(stdout, "Calling functions\n\n");

  // reduce_sum_1d([1, 10, 100])
  if (iree_status_is_ok(status)) {
    const int input[3] = {1, 10, 100};
    int result = -1;
    status = reduce_sum_1d(session, input, 3, &result);
    fprintf(stdout, "reduce_sum_1d([1, 10, 100]): %d\n", result);
  }

  // reduce_sum_2d([[1, 2, 3], [10, 20, 30]])
  if (iree_status_is_ok(status)) {
    const int input[6] = {1, 2, 3, 10, 20, 30};
    iree_hal_buffer_view_t* result_buffer_view = NULL;
    status = reduce_sum_2d(session, input, 6, &result_buffer_view);
    if (iree_status_is_ok(status)) {
      fprintf(stdout, "reduce_sum_2d([[1, 2, 3], [10, 20, 30]]): ");
      status = iree_hal_buffer_view_fprint(
          stdout, result_buffer_view,
          /*max_element_count=*/4096,
          iree_runtime_session_host_allocator(session));
      fprintf(stdout, "\n");
    }
    iree_hal_buffer_view_release(result_buffer_view);
  }

  // reduce_sum_2d([[1, 2, 3], [10, 20, 30], [100, 200, 300]])
  if (iree_status_is_ok(status)) {
    const int input[9] = {1, 2, 3, 10, 20, 30, 100, 200, 300};
    iree_hal_buffer_view_t* result_buffer_view = NULL;
    status = reduce_sum_2d(session, input, 9, &result_buffer_view);
    if (iree_status_is_ok(status)) {
      fprintf(stdout,
              "reduce_sum_2d([[1, 2, 3], [10, 20, 30], [100, 200, 300]]): ");
      status = iree_hal_buffer_view_fprint(
          stdout, result_buffer_view,
          /*max_element_count=*/4096,
          iree_runtime_session_host_allocator(session));
      fprintf(stdout, "\n");
    }
    iree_hal_buffer_view_release(result_buffer_view);
  }

  // add_one([1, 10, 100])
  if (iree_status_is_ok(status)) {
    const int input[3] = {1, 10, 100};
    iree_hal_buffer_view_t* result_buffer_view = NULL;
    status = add_one(session, input, 3, &result_buffer_view);
    if (iree_status_is_ok(status)) {
      fprintf(stdout, "add_one([1, 10, 100]): ");
      status = iree_hal_buffer_view_fprint(
          stdout, result_buffer_view,
          /*max_element_count=*/64,
          iree_runtime_session_host_allocator(session));
      fprintf(stdout, "\n");
    }
    iree_hal_buffer_view_release(result_buffer_view);
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
  if (argc != 3) {
    fprintf(
        stderr,
        "Usage: dynamic-shapes </path/to/dynamic_shapes.vmfb> <driver_name>\n");
    fprintf(stderr, "  (See the README for this sample for details)\n ");
    return -1;
  }

  iree_string_view_t bytecode_module_path = iree_make_cstring_view(argv[1]);
  iree_string_view_t driver_name = iree_make_cstring_view(argv[2]);

  iree_status_t result = run_sample(bytecode_module_path, driver_name);
  if (!iree_status_is_ok(result)) {
    fprintf(stdout, "Failed!\n");
    iree_status_fprint(stderr, result);
    iree_status_ignore(result);
    return -1;
  }
  fprintf(stdout, "\nSuccess!\n");
  return 0;
}
