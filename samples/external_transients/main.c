// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Demonstrates using external transient buffers with the IREE C API.
//
// This sample shows how to:
// 1. Query the transient buffer size needed for a function using reflection
//    attributes (iree.abi.transients.size.constant or iree.abi.transients.size)
// 2. Allocate the transient buffer from the device
// 3. Pass it to the function invocation along with inputs and outputs
//
// NOTE: this file does not properly handle error cases and will leak on
// failure. Applications that are just going to exit()/abort() on failure can
// probably get away with the same thing but really should prefer not to.

#include <stdio.h>
#include <stdlib.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"

// A function to create the HAL device from the different backend targets.
// The HAL device is returned based on the implementation, and it must be
// released by the caller.
extern iree_status_t create_sample_device(iree_allocator_t host_allocator,
                                          iree_hal_device_t** out_device);

// A function to load the vm bytecode module from the different backend targets.
// The bytecode module is generated for the specific backend and platform.
extern const iree_const_byte_span_t load_bytecode_module_data();

// Queries the size needed for transient storage for the given function.
// This checks for the iree.abi.transients.size.constant attribute first,
// and if not present, calls the function referenced by
// iree.abi.transients.size.
static iree_status_t query_transient_size(
    iree_vm_context_t* context, const iree_vm_function_t* main_function,
    iree_vm_list_t* main_inputs, iree_allocator_t host_allocator,
    iree_host_size_t* out_size) {
  // First check for a constant size attribute.
  iree_string_view_t size_constant_attr = iree_vm_function_lookup_attr_by_name(
      main_function, IREE_SV("iree.abi.transients.size.constant"));
  if (!iree_string_view_is_empty(size_constant_attr)) {
    // Constant size is specified - parse it directly.
    if (!iree_string_view_atoi_uint64(size_constant_attr,
                                      (uint64_t*)(out_size))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "failed to parse integer attribute");
    }
    return iree_ok_status();
  }

  // No constant size - need to call the size query function.
  iree_string_view_t size_func_name = iree_vm_function_lookup_attr_by_name(
      main_function, IREE_SV("iree.abi.transients.size"));
  if (iree_string_view_is_empty(size_func_name)) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "function has no transient size information (missing both "
        "iree.abi.transients.size.constant and iree.abi.transients.size)");
  }

  // Resolve the size query function.
  iree_vm_function_t size_function;
  IREE_RETURN_IF_ERROR(iree_vm_context_resolve_function(context, size_func_name,
                                                        &size_function));

  // Call the size function with the same inputs as the main function.
  iree_vm_list_t* size_outputs = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_list_create(iree_vm_make_undefined_type_def(),
                          /*capacity=*/1, host_allocator, &size_outputs),
      "can't allocate size query output list");
  iree_status_t status = iree_vm_invoke(
      context, size_function, IREE_VM_INVOCATION_FLAG_NONE,
      /*policy=*/NULL, main_inputs, size_outputs, host_allocator);
  if (iree_status_is_ok(status)) {
    // Extract the size from the output (should be an i64).
    iree_vm_value_t size_value;
    status = iree_vm_list_get_value_as(size_outputs, 0, IREE_VM_VALUE_TYPE_I64,
                                       &size_value);
    if (iree_status_is_ok(status)) {
      *out_size = (iree_host_size_t)size_value.i64;
    }
  }
  iree_vm_list_release(size_outputs);
  return status;
}

#define INPUT_SIZE 64

iree_status_t Run() {
  iree_allocator_t host_allocator = iree_allocator_system();

  iree_vm_instance_t* instance = NULL;
  IREE_CHECK_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                        host_allocator, &instance));
  IREE_CHECK_OK(iree_hal_module_register_all_types(instance));

  iree_hal_device_t* device = NULL;
  IREE_CHECK_OK(create_sample_device(host_allocator, &device), "create device");
  iree_hal_device_group_t* device_group = NULL;
  IREE_CHECK_OK(iree_hal_device_group_create_from_device(device, host_allocator,
                                                         &device_group));
  iree_vm_module_t* hal_module = NULL;
  IREE_CHECK_OK(iree_hal_module_create(
      instance, iree_hal_module_device_policy_default(), device_group,
      IREE_HAL_MODULE_FLAG_SYNCHRONOUS,
      iree_hal_module_debug_sink_stdio(stderr), host_allocator, &hal_module));
  iree_hal_device_group_release(device_group);

  // Load bytecode module from the embedded data.
  const iree_const_byte_span_t module_data = load_bytecode_module_data();
  iree_vm_module_t* bytecode_module = NULL;
  IREE_CHECK_OK(iree_vm_bytecode_module_create(
      instance, IREE_VM_BYTECODE_MODULE_FLAG_NONE, module_data,
      iree_allocator_null(), host_allocator, &bytecode_module));

  // Allocate a context that will hold the module state across invocations.
  iree_vm_context_t* context = NULL;
  iree_vm_module_t* modules[] = {hal_module, bytecode_module};
  IREE_CHECK_OK(iree_vm_context_create_with_modules(
      instance, IREE_VM_CONTEXT_FLAG_NONE, IREE_ARRAYSIZE(modules), &modules[0],
      host_allocator, &context));
  iree_vm_module_release(hal_module);
  iree_vm_module_release(bytecode_module);

  // Lookup the entry point function.
  const char kMainFunctionName[] = "module.in_place_computation";
  iree_vm_function_t main_function;
  IREE_CHECK_OK(iree_vm_context_resolve_function(
      context, iree_make_cstring_view(kMainFunctionName), &main_function));

  // Prepare input buffer: 64xf32 filled with 1.0.
  float input_data[INPUT_SIZE];
  for (iree_host_size_t i = 0; i < INPUT_SIZE; ++i) {
    input_data[i] = 1.0f;
  }
  iree_hal_dim_t input_shape[1] = {INPUT_SIZE};
  iree_hal_buffer_view_t* input_buffer_view = NULL;
  IREE_CHECK_OK(iree_hal_buffer_view_allocate_buffer_copy(
      device, iree_hal_device_allocator(device), IREE_ARRAYSIZE(input_shape),
      input_shape, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      },
      iree_make_const_byte_span(input_data, sizeof(input_data)),
      &input_buffer_view));

  // Prepare output buffer: 64xf32 of zeros (storage for the result).
  float output_data[INPUT_SIZE];
  for (iree_host_size_t i = 0; i < INPUT_SIZE; ++i) {
    output_data[i] = 0.0f;
  }
  iree_hal_buffer_view_t* output_buffer_view = NULL;
  IREE_CHECK_OK(iree_hal_buffer_view_allocate_buffer_copy(
      device, iree_hal_device_allocator(device), IREE_ARRAYSIZE(input_shape),
      input_shape, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      },
      iree_make_const_byte_span(output_data, sizeof(output_data)),
      &output_buffer_view));

  // Setup inputs list for size query (input + output, but no transient yet).
  iree_vm_list_t* size_query_inputs = NULL;
  IREE_CHECK_OK(
      iree_vm_list_create(iree_vm_make_undefined_type_def(),
                          /*capacity=*/3, host_allocator, &size_query_inputs),
      "can't allocate size query input list");
  iree_vm_ref_t input_ref = iree_hal_buffer_view_move_ref(input_buffer_view);
  iree_vm_ref_t output_ref = iree_hal_buffer_view_move_ref(output_buffer_view);
  IREE_CHECK_OK(iree_vm_list_push_ref_retain(size_query_inputs, &input_ref));
  IREE_CHECK_OK(iree_vm_list_push_ref_retain(size_query_inputs, &output_ref));

  // Query the transient buffer size.
  iree_host_size_t transient_size = 0;
  IREE_CHECK_OK(query_transient_size(context, &main_function, size_query_inputs,
                                     host_allocator, &transient_size),
                "failed to query transient size");
  fprintf(stdout, "Transient buffer size needed: %zu bytes\n", transient_size);

  // Allocate the transient buffer.
  iree_hal_buffer_t* transient_buffer = NULL;
  IREE_CHECK_OK(iree_hal_allocator_allocate_buffer(
                    iree_hal_device_allocator(device),
                    (iree_hal_buffer_params_t){
                        .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
                        .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
                    },
                    transient_size, &transient_buffer),
                "failed to allocate transient buffer");

  // Setup call inputs: input, output, transient.
  iree_vm_list_t* inputs = NULL;
  IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                    /*capacity=*/3, host_allocator, &inputs),
                "can't allocate input vm list");

  iree_vm_ref_t transient_ref = iree_hal_buffer_move_ref(transient_buffer);
  IREE_CHECK_OK(iree_vm_list_push_ref_move(inputs, &input_ref));
  IREE_CHECK_OK(iree_vm_list_push_ref_move(inputs, &output_ref));
  IREE_CHECK_OK(iree_vm_list_push_ref_move(inputs, &transient_ref));

  // Prepare outputs list to accept the results from the invocation.
  iree_vm_list_t* outputs = NULL;
  IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                    /*capacity=*/1, host_allocator, &outputs),
                "can't allocate output vm list");

  // Synchronously invoke the function.
  IREE_CHECK_OK(
      iree_vm_invoke(context, main_function, IREE_VM_INVOCATION_FLAG_NONE,
                     /*policy=*/NULL, inputs, outputs, host_allocator));

  // Get the result buffer from the invocation.
  iree_hal_buffer_view_t* ret_buffer_view =
      iree_vm_list_get_buffer_view_assign(outputs, 0);
  if (ret_buffer_view == NULL) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "can't find return buffer view");
  }

  // Read back the results and ensure we got the right values.
  // Expected: (input + 1.0) * 2.0 + input = (1.0 + 1.0) * 2.0 + 1.0 = 5.0
  float results[INPUT_SIZE];
  IREE_CHECK_OK(iree_hal_device_transfer_d2h(
      device, iree_hal_buffer_view_buffer(ret_buffer_view), 0, results,
      sizeof(results), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));
  fprintf(stdout, "Results: ");
  for (iree_host_size_t i = 0; i < INPUT_SIZE; ++i) {
    if (i > 0) fprintf(stdout, ", ");
    fprintf(stdout, "%.1f", results[i]);
    if (i >= 5) {
      fprintf(stdout, ", ...");
      break;
    }
  }
  fprintf(stdout, "\n");

  // Verify the results.
  for (iree_host_size_t i = 0; i < INPUT_SIZE; ++i) {
    if (results[i] != 5.0f) {
      return iree_make_status(
          IREE_STATUS_UNKNOWN,
          "result mismatches at index %zu: expected 5.0, got %.1f", i,
          results[i]);
    }
  }

  iree_vm_list_release(size_query_inputs);
  iree_vm_list_release(inputs);
  iree_vm_list_release(outputs);
  iree_hal_device_release(device);
  iree_vm_context_release(context);
  iree_vm_instance_release(instance);
  return iree_ok_status();
}

int main() {
  const iree_status_t result = Run();
  int ret = (int)iree_status_code(result);
  if (!iree_status_is_ok(result)) {
    iree_status_fprint(stderr, result);
    iree_status_free(result);
  } else {
    fprintf(stdout, "external_transients sample completed successfully\n");
  }
  return ret;
}
