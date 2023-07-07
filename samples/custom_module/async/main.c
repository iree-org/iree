// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

// IREE APIs:
#include "iree/modules/hal/types.h"
#include "iree/runtime/api.h"

// Custom native module used in the sample.
// Modules may be linked in from native code or other bytecode modules loaded at
// runtime: there's no difference.
#include "module.h"

// NOTE: CHECKs are dangerous but this is a sample; a real application would
// want to handle errors gracefully. We know in this constrained case that
// these won't fail unless something is catastrophically wrong (out of memory,
// solar flares, etc).
int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr,
            "Usage:\n"
            "  custom-module-async-run - <entry.point> # read from stdin\n"
            "  custom-module-async-run </path/to/say_hello.vmfb> "
            "<entry.point>\n");
    fprintf(stderr, "  (See the README for this sample for details)\n ");
    return -1;
  }

  // Internally IREE does not (in general) use malloc and instead uses the
  // provided allocator to allocate and free memory. Applications can integrate
  // their own allocator as-needed.
  iree_allocator_t host_allocator = iree_allocator_system();

  // Create and configure the instance shared across all sessions.
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  iree_runtime_instance_t* instance = NULL;
  IREE_CHECK_OK(iree_runtime_instance_create(&instance_options, host_allocator,
                                             &instance));

  // Try to create the device - it should always succeed as it's a CPU device.
  iree_hal_device_t* device = NULL;
  IREE_CHECK_OK(iree_runtime_instance_try_create_default_device(
      instance, iree_make_cstring_view("local-task"), &device));

  // Create one session per loaded module to hold the module state.
  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  // Useful to see the VM program flow:
  // session_options.context_flags = IREE_VM_CONTEXT_FLAG_TRACE_EXECUTION;
  iree_runtime_session_t* session = NULL;
  IREE_CHECK_OK(iree_runtime_session_create_with_device(
      instance, &session_options, device,
      iree_runtime_instance_host_allocator(instance), &session));

  // Create the custom module that can be reused across contexts.
  iree_vm_module_t* custom_module = NULL;
  IREE_CHECK_OK(iree_custom_module_async_create(
      iree_runtime_instance_vm_instance(instance), device, host_allocator,
      &custom_module));
  IREE_CHECK_OK(iree_runtime_session_append_module(session, custom_module));
  iree_vm_module_release(custom_module);

  // Load the module from stdin or a file on disk.
  const char* module_path = argv[1];
  if (strcmp(module_path, "-") == 0) {
    IREE_CHECK_OK(
        iree_runtime_session_append_bytecode_module_from_stdin(session));
  } else {
    IREE_CHECK_OK(iree_runtime_session_append_bytecode_module_from_file(
        session, module_path));
  }

  iree_vm_list_t* inputs = NULL;
  IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                    host_allocator, &inputs));
  iree_vm_list_t* outputs = NULL;
  IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                    host_allocator, &outputs));

  // Pass in the tensor<?xi32> arg:
  const int32_t input_data[5] = {1, 2, 3, 4, 5};
  const iree_hal_dim_t shape[1] = {IREE_ARRAYSIZE(input_data)};
  iree_hal_buffer_view_t* input_view = NULL;
  IREE_CHECK_OK(iree_hal_buffer_view_allocate_buffer(
      iree_runtime_session_device_allocator(session), IREE_ARRAYSIZE(shape),
      shape, IREE_HAL_ELEMENT_TYPE_INT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .access = IREE_HAL_MEMORY_ACCESS_READ,
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      },
      iree_make_const_byte_span(input_data, sizeof(input_data)), &input_view));
  iree_vm_ref_t input_view_ref = iree_hal_buffer_view_move_ref(input_view);
  IREE_CHECK_OK(iree_vm_list_push_ref_move(inputs, &input_view_ref));

  // Create our own timeline and set fences at T=1 and T=2.
  // We'll pass these in with the timeline at T=0 so that the runtime isn't
  // allowed to execute anything until we give it the go-ahead.
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_CHECK_OK(iree_hal_semaphore_create(device, 0ull, &semaphore));
  iree_hal_fence_t* fence_t1 = NULL;
  IREE_CHECK_OK(
      iree_hal_fence_create_at(semaphore, 1ull, host_allocator, &fence_t1));
  iree_hal_fence_t* fence_t2 = NULL;
  IREE_CHECK_OK(
      iree_hal_fence_create_at(semaphore, 2ull, host_allocator, &fence_t2));
  iree_hal_semaphore_release(semaphore);
  fprintf(stdout, "INITIALIZE T=0\n");
  fflush(stdout);

  // Add the (wait_fence, signal_fence) pair to the function call.
  // The --iree-execution-model=async-external flag adds these required
  // arguments to the functions exported by the module.
  iree_vm_ref_t fence_t1_ref = iree_hal_fence_retain_ref(fence_t1);
  IREE_CHECK_OK(iree_vm_list_push_ref_move(inputs, &fence_t1_ref));
  iree_vm_ref_t fence_t2_ref = iree_hal_fence_retain_ref(fence_t2);
  IREE_CHECK_OK(iree_vm_list_push_ref_move(inputs, &fence_t2_ref));

  // Let the call start executing by signaling the timeline to T=1.
  // TODO(benvanik): fix wait-before-signal on queue-ordered allocations.
  // For now we have to signal to T=1 before invoking the function but that's
  // only temporary. This should be moved down to after the VM invocation
  // returns so that we can show how all of the program execution can be
  // deferred. We could simulate this with another thread that signaled in the
  // future if we wanted.
  IREE_CHECK_OK(iree_hal_fence_signal(fence_t1));
  fprintf(stdout, "SIGNALED T=1\n");
  fflush(stdout);

  // Invoke the target function.
  // This will return immediately after scheduling work - including the custom
  // call - but will not actually execute anything until we say it's OK by
  // advancing the timeline to T=1.
  iree_string_view_t entry_point = iree_make_cstring_view(argv[2]);
  fprintf(stdout, "VM INVOKE BEGIN %.*s\n", (int)entry_point.size,
          entry_point.data);
  fflush(stdout);
  IREE_CHECK_OK(
      iree_runtime_session_call_by_name(session, entry_point, inputs, outputs));
  fprintf(stdout, "VM INVOKE END\n");
  fflush(stdout);

  // We could go do other things now while the async work progresses. Here we
  // just immediately wait.
  IREE_CHECK_OK(iree_hal_fence_wait(fence_t2, iree_infinite_timeout()));
  fprintf(stdout, "REACHED T=2\n");
  fflush(stdout);

  // Read back the tensor<?xi32> result:
  iree_hal_buffer_view_t* output_view =
      iree_vm_list_get_buffer_view_assign(outputs, 0);
  int32_t output_data[5] = {0};
  IREE_CHECK_OK(
      iree_hal_buffer_map_read(iree_hal_buffer_view_buffer(output_view), 0,
                               output_data, sizeof(output_data)));

  // Expecting (e^2 * 2)^2:
  bool did_match = true;
  for (size_t i = 0; i < IREE_ARRAYSIZE(input_data); ++i) {
    int32_t t0 = input_data[i];
    int32_t t1 = t0 * t0;
    int32_t t2 = t1 * 2;
    int32_t t3 = t2 * t2;
    if (t3 != output_data[i]) {
      fprintf(stdout, "MISMATCH [%zu] expected %d but actual %d\n", i, t3,
              output_data[i]);
      did_match = false;
      break;
    }
  }
  if (did_match) {
    fprintf(stdout, "MATCHED!\n");
  }

  iree_vm_list_release(inputs);
  iree_vm_list_release(outputs);
  iree_hal_fence_release(fence_t1);
  iree_hal_fence_release(fence_t2);
  iree_runtime_session_release(session);
  iree_hal_device_release(device);
  iree_runtime_instance_release(instance);
  return 0;
}
