// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/runtime/api.h"

iree_status_t counter_get_value(iree_runtime_session_t* session,
                                int* out_value) {
  iree_runtime_call_t call;
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view("module.get_value"), &call));

  iree_status_t status = iree_ok_status();
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
        iree_hal_buffer_view_buffer(buffer_view), 0, out_value,
        sizeof(*out_value), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout());
  }
  iree_hal_buffer_view_release(buffer_view);

  iree_runtime_call_deinitialize(&call);
  return status;
}

iree_status_t counter_set_value(iree_runtime_session_t* session,
                                int new_value) {
  iree_runtime_call_t call;
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view("module.set_value"), &call));

  iree_hal_buffer_view_t* arg0 = NULL;
  int arg0_data[1] = {new_value};

  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_view_allocate_buffer(
        iree_runtime_session_device_allocator(session),
        /*shape_rank=*/0, /*shape=*/NULL, IREE_HAL_ELEMENT_TYPE_SINT_32,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        (iree_hal_buffer_params_t){
            .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
            .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
        },
        iree_make_const_byte_span((void*)arg0_data, sizeof(arg0_data)), &arg0);
  }
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_inputs_push_back_buffer_view(&call, arg0);
  }
  iree_hal_buffer_view_release(arg0);
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_invoke(&call, /*flags=*/0);
  }

  iree_runtime_call_deinitialize(&call);
  return status;
}

iree_status_t counter_add_to_value(iree_runtime_session_t* session, int x) {
  iree_runtime_call_t call;
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view("module.add_to_value"), &call));

  iree_hal_buffer_view_t* arg0 = NULL;
  int arg0_data[1] = {x};

  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_view_allocate_buffer(
        iree_runtime_session_device_allocator(session), /*shape_rank=*/0,
        /*shape=*/NULL, IREE_HAL_ELEMENT_TYPE_SINT_32,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        (iree_hal_buffer_params_t){
            .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
            .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
        },
        iree_make_const_byte_span((void*)arg0_data, sizeof(arg0_data)), &arg0);
  }
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_inputs_push_back_buffer_view(&call, arg0);
  }
  iree_hal_buffer_view_release(arg0);
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_invoke(&call, /*flags=*/0);
  }

  iree_runtime_call_deinitialize(&call);
  return status;
}

iree_status_t counter_reset_value(iree_runtime_session_t* session) {
  iree_runtime_call_t call;
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view("module.reset_value"), &call));

  iree_status_t status = iree_runtime_call_invoke(&call, /*flags=*/0);

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
  // Call functions to manipulate the counter
  fprintf(stdout, "Calling functions\n\n");

  // 1. get_value() // initial value
  int value = -1;
  if (iree_status_is_ok(status)) {
    status = counter_get_value(session, &value);
    fprintf(stdout, "Initial get_value()    : %d\n", value);
  }

  // 2. set_value(101)
  if (iree_status_is_ok(status)) {
    status = counter_set_value(session, 101);
  }
  if (iree_status_is_ok(status)) {
    status = counter_get_value(session, &value);
    fprintf(stdout, "After set_value(101)   : %d\n", value);
  }

  // 3. add_to_value(20)
  if (iree_status_is_ok(status)) {
    status = counter_add_to_value(session, 20);
  }
  if (iree_status_is_ok(status)) {
    status = counter_get_value(session, &value);
    fprintf(stdout, "After add_to_value(20) : %d\n", value);
  }

  // 4. add_to_value(-50)
  if (iree_status_is_ok(status)) {
    status = counter_add_to_value(session, -50);
  }
  if (iree_status_is_ok(status)) {
    status = counter_get_value(session, &value);
    fprintf(stdout, "After add_to_value(-50): %d\n", value);
  }

  // 5. reset_value()
  if (iree_status_is_ok(status)) {
    status = counter_reset_value(session);
  }
  if (iree_status_is_ok(status)) {
    status = counter_get_value(session, &value);
    fprintf(stdout, "After reset_value()    : %d\n", value);
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
        "Usage: variables-and-state </path/to/counter.vmfb> <driver_name>\n");
    fprintf(stderr, "  (See the README for this sample for details)\n ");
    return -1;
  }

  iree_string_view_t bytecode_module_path = iree_make_cstring_view(argv[1]);
  iree_string_view_t driver_name = iree_make_cstring_view(argv[2]);

  iree_status_t result = run_sample(bytecode_module_path, driver_name);
  if (!iree_status_is_ok(result)) {
    iree_status_fprint(stderr, result);
    iree_status_ignore(result);
    return -1;
  }
  fprintf(stdout, "\nSuccess!\n");
  return 0;
}
