// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/runtime/api.h"

iree_status_t RunSample(iree_string_view_t bytecode_module_path,
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
  // Call functions.
  fprintf(stdout, "Calling functions\n\n");
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

  iree_hal_buffer_view_t* ret0 = NULL;

  // Call get_value() to check the initial value.
  IREE_RETURN_IF_ERROR(iree_runtime_call_invoke(&call_get_value, /*flags=*/0));
  IREE_RETURN_IF_ERROR(
      iree_runtime_call_outputs_pop_front_buffer_view(&call_get_value, &ret0));
  fprintf(stdout, "Initial value from get_value():\n");
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_view_fprint(stdout, ret0, /*max_element_count=*/64));

  // Call set_value() then get_value() again.
  iree_hal_buffer_view_t* arg0 = NULL;
  static const iree_hal_dim_t arg0_shape[1] = {1};
  static const int arg0_data[1] = {101};
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_wrap_or_clone_heap_buffer(
      iree_runtime_session_device_allocator(session), arg0_shape,
      IREE_ARRAYSIZE(arg0_shape), IREE_HAL_ELEMENT_TYPE_SINT_32,
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      IREE_HAL_MEMORY_ACCESS_READ, IREE_HAL_BUFFER_USAGE_ALL,
      iree_make_byte_span((void*)arg0_data, sizeof(arg0_data)),
      iree_allocator_null(), &arg0));
  IREE_RETURN_IF_ERROR(
      iree_runtime_call_inputs_push_back_buffer_view(&call_set_value, arg0));
  iree_hal_buffer_view_release(arg0);
  IREE_RETURN_IF_ERROR(iree_runtime_call_invoke(&call_set_value, /*flags=*/0));
  // Value should be set now, check by calling get_value().
  IREE_RETURN_IF_ERROR(iree_runtime_call_invoke(&call_get_value, /*flags=*/0));
  IREE_RETURN_IF_ERROR(
      iree_runtime_call_outputs_pop_front_buffer_view(&call_get_value, &ret0));
  fprintf(stdout, "\nNew value from get_value() after set_value():\n");
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_view_fprint(stdout, ret0, /*max_element_count=*/64));

  // Call add_to_value() a few times.
  iree_hal_buffer_view_t* arg1 = NULL;
  static const iree_hal_dim_t arg1_shape[1] = {1};
  static const int arg1_data[1] = {20};
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_wrap_or_clone_heap_buffer(
      iree_runtime_session_device_allocator(session), arg1_shape,
      IREE_ARRAYSIZE(arg1_shape), IREE_HAL_ELEMENT_TYPE_SINT_32,
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      IREE_HAL_MEMORY_ACCESS_READ, IREE_HAL_BUFFER_USAGE_ALL,
      iree_make_byte_span((void*)arg1_data, sizeof(arg1_data)),
      iree_allocator_null(), &arg1));
  IREE_RETURN_IF_ERROR(
      iree_runtime_call_inputs_push_back_buffer_view(&call_add_to_value, arg1));
  iree_hal_buffer_view_release(arg1);
  IREE_RETURN_IF_ERROR(
      iree_runtime_call_invoke(&call_add_to_value, /*flags=*/0));
  // Value should be set now, check by calling get_value().
  IREE_RETURN_IF_ERROR(iree_runtime_call_invoke(&call_get_value,
                                                /*flags=*/0));
  IREE_RETURN_IF_ERROR(
      iree_runtime_call_outputs_pop_front_buffer_view(&call_get_value, &ret0));
  fprintf(stdout, "\nNew value from get_value() after add_to_value():\n");
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_view_fprint(stdout, ret0, /*max_element_count=*/64));
  // Call add_to_value() again *without* a call to iree_runtime_call_reset.
  // This should retain the previous input.
  IREE_RETURN_IF_ERROR(
      iree_runtime_call_invoke(&call_add_to_value, /*flags=*/0));
  IREE_RETURN_IF_ERROR(iree_runtime_call_invoke(&call_get_value,
                                                /*flags=*/0));
  IREE_RETURN_IF_ERROR(
      iree_runtime_call_outputs_pop_front_buffer_view(&call_get_value, &ret0));
  fprintf(stdout, "\nNew value from get_value() after add_to_value():\n");
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_view_fprint(stdout, ret0, /*max_element_count=*/64));

  // Call reset_value().
  IREE_RETURN_IF_ERROR(
      iree_runtime_call_invoke(&call_reset_value, /*flags=*/0));
  IREE_RETURN_IF_ERROR(iree_runtime_call_invoke(&call_get_value, /*flags=*/0));
  IREE_RETURN_IF_ERROR(
      iree_runtime_call_outputs_pop_front_buffer_view(&call_get_value, &ret0));
  fprintf(stdout, "\nNew value from get_value() after reset_value():\n");
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_view_fprint(stdout, ret0, /*max_element_count=*/64));

  iree_hal_buffer_view_release(ret0);

  iree_runtime_call_deinitialize(&call_get_value);
  iree_runtime_call_deinitialize(&call_set_value);
  iree_runtime_call_deinitialize(&call_add_to_value);
  iree_runtime_call_deinitialize(&call_reset_value);
  //===-------------------------------------------------------------------===//

  //===-------------------------------------------------------------------===//
  // Cleanup.
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

  iree_status_t result = RunSample(bytecode_module_path, driver_name);
  if (!iree_status_is_ok(result)) {
    iree_status_fprint(stderr, result);
    iree_status_ignore(result);
    return -1;
  }
  fprintf(stdout, "\nSuccess!\n");
  return 0;
}
