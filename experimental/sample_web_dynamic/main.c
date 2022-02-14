// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>
#include <stdio.h>

#include "iree/runtime/api.h"
#include "iree/vm/bytecode_module.h"

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

int load_program(uint8_t* vmfb_data, size_t length);

//===----------------------------------------------------------------------===//
// Implementation
//===----------------------------------------------------------------------===//

// We're not really using cpuinfo here, but the linker complains about this
// function not being defined. It actually is defined though, _if_ you patch
// cpuinfo's CMake configuration: https://github.com/pytorch/cpuinfo/issues/34.
void __attribute__((weak)) cpuinfo_emscripten_init(void) {}

extern iree_status_t create_device_with_wasm_loader(
    iree_allocator_t host_allocator, iree_hal_device_t** out_device);

void inspect_module(iree_vm_module_t* module) {
  fprintf(stdout, "=== module properties ===\n");

  iree_string_view_t module_name = iree_vm_module_name(module);
  fprintf(stdout, "  module name: '%.*s'\n", (int)module_name.size,
          module_name.data);

  iree_vm_module_signature_t module_signature =
      iree_vm_module_signature(module);
  fprintf(stdout, "  module signature:\n");
  fprintf(stdout, "    %" PRIhsz " imported functions\n",
          module_signature.import_function_count);
  fprintf(stdout, "    %" PRIhsz " exported functions\n",
          module_signature.export_function_count);
  fprintf(stdout, "    %" PRIhsz " internal functions\n",
          module_signature.internal_function_count);

  fprintf(stdout, "  exported functions:\n");
  for (iree_host_size_t i = 0; i < module_signature.export_function_count;
       ++i) {
    iree_vm_function_t function;
    iree_status_t status = iree_vm_module_lookup_function_by_ordinal(
        module, IREE_VM_FUNCTION_LINKAGE_EXPORT, i, &function);

    iree_string_view_t function_name = iree_vm_function_name(&function);
    iree_vm_function_signature_t function_signature =
        iree_vm_function_signature(&function);
    iree_string_view_t calling_convention =
        function_signature.calling_convention;
    fprintf(stdout, "    function name: '%.*s', calling convention: %.*s'\n",
            (int)function_name.size, function_name.data,
            (int)calling_convention.size, calling_convention.data);
  }
}

int load_program(uint8_t* vmfb_data, size_t length) {
  fprintf(stdout, "load_program() received %zu bytes of data\n", length);

  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(IREE_API_VERSION_LATEST,
                                           &instance_options);
  // Note: no call to iree_runtime_instance_options_use_all_available_drivers().

  iree_runtime_instance_t* instance = NULL;
  iree_status_t status = iree_runtime_instance_create(
      &instance_options, iree_allocator_system(), &instance);

  iree_hal_device_t* device = NULL;
  if (iree_status_is_ok(status)) {
    status = create_device_with_wasm_loader(iree_allocator_system(), &device);
  }

  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  iree_runtime_session_t* session = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_create_with_device(
        instance, &session_options, device,
        iree_runtime_instance_host_allocator(instance), &session);
  }

  iree_vm_module_t* program_module = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_module_create(
        iree_make_const_byte_span(vmfb_data, length), iree_allocator_null(),
        iree_allocator_system(), &program_module);
  }

  if (iree_status_is_ok(status)) {
    inspect_module(program_module);
    status = iree_runtime_session_append_module(session, program_module);
  }

  // Call the 'abs' function in the module.
  iree_runtime_call_t call;
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_initialize_by_name(
        session, iree_make_cstring_view("module.abs"), &call);
  }

  iree_hal_buffer_view_t* arg0 = NULL;
  float arg0_data[1] = {-5.5};
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_view_allocate_buffer(
        iree_runtime_session_device_allocator(session), /*shape=*/NULL,
        /*shape_rank=*/0, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        IREE_HAL_BUFFER_USAGE_DISPATCH | IREE_HAL_BUFFER_USAGE_TRANSFER,
        iree_make_const_byte_span((void*)arg0_data, sizeof(arg0_data)), &arg0);
  }
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_inputs_push_back_buffer_view(&call, arg0);
  }
  iree_hal_buffer_view_release(arg0);

  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_invoke(&call, /*flags=*/0);
  }

  iree_hal_buffer_view_t* ret_buffer_view = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_outputs_pop_front_buffer_view(&call,
                                                             &ret_buffer_view);
  }
  float result[1] = {0.0f};
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_buffer_read_data(iree_hal_buffer_view_buffer(ret_buffer_view),
                                  0, result, sizeof(result));
  }
  iree_hal_buffer_view_release(ret_buffer_view);

  if (iree_status_is_ok(status)) {
    fprintf(stdout, "abs(%f) -> result: %f\n", arg0_data[0], result[0]);
  }

  iree_runtime_call_deinitialize(&call);

  iree_vm_module_release(program_module);
  iree_hal_device_release(device);
  iree_runtime_session_release(session);
  iree_runtime_instance_release(instance);

  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return -1;
  }

  return 0;
}
