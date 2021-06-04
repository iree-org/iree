// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/runtime/api.h"

iree_status_t counter_get_value(iree_runtime_call_t* call_get_value,
                                iree_hal_buffer_view_t** out_value) {
  IREE_RETURN_IF_ERROR(iree_runtime_call_invoke(call_get_value, /*flags=*/0));
  IREE_RETURN_IF_ERROR(iree_runtime_call_outputs_pop_front_buffer_view(
      call_get_value, out_value));
  return iree_ok_status();
}

iree_status_t counter_print_current_value(iree_runtime_call_t* call_get_value) {
  iree_hal_buffer_view_t* ret = NULL;
  IREE_RETURN_IF_ERROR(counter_get_value(call_get_value, &ret));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_fprint(stdout, ret,
                                                   /*max_element_count=*/64));
  iree_hal_buffer_view_release(ret);
  return iree_ok_status();
}

iree_status_t counter_set_value(iree_runtime_call_t* call_set_value,
                                int new_value) {
  iree_runtime_call_reset(call_set_value);
  iree_hal_buffer_view_t* arg0 = NULL;
  static const iree_hal_dim_t arg0_shape[1] = {1};
  int arg0_data[1] = {new_value};

  // TODO(scotttodd): use iree_hal_buffer_view_wrap_or_clone_heap_buffer
  //   * debugging some apparent memory corruption with the stack-local value
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_clone_heap_buffer(
      iree_runtime_session_device_allocator(call_set_value->session),
      arg0_shape, IREE_ARRAYSIZE(arg0_shape), IREE_HAL_ELEMENT_TYPE_SINT_32,
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      IREE_HAL_BUFFER_USAGE_ALL,
      iree_make_const_byte_span((void*)arg0_data, sizeof(arg0_data)), &arg0));
  IREE_RETURN_IF_ERROR(
      iree_runtime_call_inputs_push_back_buffer_view(call_set_value, arg0));
  iree_hal_buffer_view_release(arg0);
  IREE_RETURN_IF_ERROR(iree_runtime_call_invoke(call_set_value, /*flags=*/0));
  return iree_ok_status();
}

iree_status_t counter_add_to_value(iree_runtime_call_t* call_add_to_value,
                                   int x) {
  iree_runtime_call_reset(call_add_to_value);
  iree_hal_buffer_view_t* arg0 = NULL;
  static const iree_hal_dim_t arg0_shape[1] = {1};
  int arg0_data[1] = {x};

  // TODO(scotttodd): use iree_hal_buffer_view_wrap_or_clone_heap_buffer
  //   * debugging some apparent memory corruption with the stack-local value
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_clone_heap_buffer(
      iree_runtime_session_device_allocator(call_add_to_value->session),
      arg0_shape, IREE_ARRAYSIZE(arg0_shape), IREE_HAL_ELEMENT_TYPE_SINT_32,
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      IREE_HAL_BUFFER_USAGE_ALL,
      iree_make_const_byte_span((void*)arg0_data, sizeof(arg0_data)), &arg0));
  IREE_RETURN_IF_ERROR(
      iree_runtime_call_inputs_push_back_buffer_view(call_add_to_value, arg0));
  iree_hal_buffer_view_release(arg0);
  IREE_RETURN_IF_ERROR(
      iree_runtime_call_invoke(call_add_to_value, /*flags=*/0));
  return iree_ok_status();
}

iree_status_t counter_reset_value(iree_runtime_call_t* call_reset_value) {
  return iree_runtime_call_invoke(call_reset_value, /*flags=*/0);
}

iree_status_t run_sample(iree_string_view_t bytecode_module_path,
                         iree_string_view_t driver_name) {
  //===-------------------------------------------------------------------===//
  // Instance configuration (this should be shared across sessions).
  fprintf(stdout, "Configuring IREE runtime instance and '%s' device\n",
          driver_name.data);
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(IREE_API_VERSION_LATEST,
                                           &instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  iree_runtime_instance_t* instance = NULL;
  IREE_RETURN_IF_ERROR(iree_runtime_instance_create(
      &instance_options, iree_allocator_system(), &instance));
  // TODO(#5724): move device selection into the compiled modules.
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_runtime_instance_try_create_default_device(
      instance, driver_name, &device));
  //===-------------------------------------------------------------------===//

  //===-------------------------------------------------------------------===//
  // Session configuration (one per loaded module to hold module state).
  fprintf(stdout, "Creating IREE runtime session\n");
  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  iree_runtime_session_t* session = NULL;
  IREE_RETURN_IF_ERROR(iree_runtime_session_create_with_device(
      instance, &session_options, device,
      iree_runtime_instance_host_allocator(instance), &session));
  iree_hal_device_release(device);

  fprintf(stdout, "Loading bytecode module at '%s'\n",
          bytecode_module_path.data);
  IREE_RETURN_IF_ERROR(iree_runtime_session_append_bytecode_module_from_file(
      session, bytecode_module_path.data));
  //===-------------------------------------------------------------------===//

  //===-------------------------------------------------------------------===//
  // Look up functions.
  iree_runtime_call_t call_get_value;
  iree_runtime_call_t call_set_value;
  iree_runtime_call_t call_add_to_value;
  iree_runtime_call_t call_reset_value;
  // Exported function names match what we used in Python.
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view("module.get_value"), &call_get_value));
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view("module.set_value"), &call_set_value));
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view("module.add_to_value"),
      &call_add_to_value));
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view("module.reset_value"),
      &call_reset_value));
  //===-------------------------------------------------------------------===//

  //===-------------------------------------------------------------------===//
  // Call functions to manipulate the counter
  fprintf(stdout, "Calling functions\n\n");

  // 1. get_value() // initial value
  fprintf(stdout, "Initial get_value():\n");
  IREE_RETURN_IF_ERROR(counter_print_current_value(&call_get_value));

  // 2. set_value(101)
  IREE_RETURN_IF_ERROR(counter_set_value(&call_set_value, 101));
  fprintf(stdout, "\nAfter set_value(101):\n");
  IREE_RETURN_IF_ERROR(counter_print_current_value(&call_get_value));

  // 3. add_to_value(20)
  IREE_RETURN_IF_ERROR(counter_add_to_value(&call_add_to_value, 20));
  fprintf(stdout, "\nAfter add_to_value(20):\n");
  IREE_RETURN_IF_ERROR(counter_print_current_value(&call_get_value));

  // 4. add_to_value(-50)
  IREE_RETURN_IF_ERROR(counter_add_to_value(&call_add_to_value, -50));
  fprintf(stdout, "\nAfter add_to_value(-50):\n");
  IREE_RETURN_IF_ERROR(counter_print_current_value(&call_get_value));

  // 5. reset_value()
  IREE_RETURN_IF_ERROR(counter_reset_value(&call_reset_value));
  fprintf(stdout, "\nAfter reset_value():\n");
  IREE_RETURN_IF_ERROR(counter_print_current_value(&call_get_value));
  //===-------------------------------------------------------------------===//

  //===-------------------------------------------------------------------===//
  // Cleanup.
  iree_runtime_call_deinitialize(&call_get_value);
  iree_runtime_call_deinitialize(&call_set_value);
  iree_runtime_call_deinitialize(&call_add_to_value);
  iree_runtime_call_deinitialize(&call_reset_value);

  iree_runtime_session_release(session);
  iree_runtime_instance_release(instance);
  //===-------------------------------------------------------------------===//

  return iree_ok_status();
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
