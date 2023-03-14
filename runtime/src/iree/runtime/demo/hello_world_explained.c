// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/runtime/api.h"

static int iree_runtime_demo_main(void);
static iree_status_t iree_runtime_demo_run_session(
    iree_runtime_instance_t* instance);
static iree_status_t iree_runtime_demo_perform_mul(
    iree_runtime_session_t* session);

#if defined(IREE_RUNTIME_DEMO_LOAD_FILE_FROM_COMMAND_LINE_ARG)

static const char* demo_file_path = NULL;

// Takes the first argument on the command line as a file path and loads it.
int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: session_demo module_file.vmfb\n");
    return 1;
  }
  demo_file_path = argv[1];
  return iree_runtime_demo_main();
}

// Loads a compiled IREE module from the file system.
static iree_status_t iree_runtime_demo_load_module(
    iree_runtime_session_t* session) {
  return iree_runtime_session_append_bytecode_module_from_file(session,
                                                               demo_file_path);
}

#elif defined(IREE_RUNTIME_DEMO_LOAD_FILE_FROM_EMBEDDED_DATA)

#include "iree/runtime/testdata/simple_mul_module_c.h"

int main(int argc, char** argv) { return iree_runtime_demo_main(); }

// Loads the bytecode module directly from memory.
//
// Embedding the compiled output into your binary is not always possible (or
// recommended) but is a fairly painless way to get things working on a variety
// of targets without worrying about how to deploy files or pass flags.
//
// In cases like this the module file is in .rodata and does not need to be
// freed; if the memory needs to be released when the module is unloaded then a
// custom allocator can be provided to get a callback instead.
static iree_status_t iree_runtime_demo_load_module(
    iree_runtime_session_t* session) {
  const iree_file_toc_t* module_file =
      iree_runtime_testdata_simple_mul_module_create();
  return iree_runtime_session_append_bytecode_module_from_memory(
      session, iree_make_const_byte_span(module_file->data, module_file->size),
      iree_allocator_null());
}

#else
#error "must specify a way to load the module data"
#endif  // IREE_RUNTIME_DEMO_LOAD_FILE_FROM_*

//===----------------------------------------------------------------------===//
// 1. Entry point / shared iree_runtime_instance_t setup
//===----------------------------------------------------------------------===//
// Applications should create and share a single instance across all sessions.

// This would live in your application startup/shutdown code or scoped to the
// usage of IREE. Creating and destroying instances is expensive and should be
// avoided.
static int iree_runtime_demo_main(void) {
  // Set up the shared runtime instance.
  // An application should usually only have one of these and share it across
  // all of the sessions it has. The instance is thread-safe, while the
  // sessions are only thread-compatible (you need to lock if its required).
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  iree_runtime_instance_t* instance = NULL;
  iree_status_t status = iree_runtime_instance_create(
      &instance_options, iree_allocator_system(), &instance);

  // Run the demo.
  // A real application would load its models (at startup, on-demand, etc) and
  // retain them somewhere to be reused. Startup time and likelihood of failure
  // varies across different HAL backends; the synchronous CPU backend is nearly
  // instantaneous and will never fail (unless out of memory) while the Vulkan
  // backend may take significantly longer and fail if there are not supported
  // devices.
  if (iree_status_is_ok(status)) {
    status = iree_runtime_demo_run_session(instance);
  }

  // Release the shared instance - it will be deallocated when all sessions
  // using it have been released (here it is deallocated immediately).
  iree_runtime_instance_release(instance);

  int ret = (int)iree_status_code(status);
  if (!iree_status_is_ok(status)) {
    // Dump nice status messages to stderr on failure.
    // An application can route these through its own logging infrastructure as
    // needed. Note that the status is a handle and must be freed!
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
  }
  return ret;
}

//===----------------------------------------------------------------------===//
// 2. Load modules and initialize state in iree_runtime_session_t
//===----------------------------------------------------------------------===//
// Each instantiation of a module will live in its own session. Module state
// like variables will be retained across calls within the same session.

// Loads the demo module and uses it to perform some math.
// In a real application you'd want to hang on to the iree_runtime_session_t
// and reuse it for future calls - especially if it holds state internally.
static iree_status_t iree_runtime_demo_run_session(
    iree_runtime_instance_t* instance) {
  // TODO(#5724): move device selection into the compiled modules.
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_runtime_instance_try_create_default_device(
      instance, iree_make_cstring_view("local-task"), &device));

  // Set up the session to run the demo module.
  // Sessions are like OS processes and are used to isolate modules from each
  // other and hold runtime state such as the variables used within the module.
  // The same module loaded into two sessions will see their own private state.
  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  iree_runtime_session_t* session = NULL;
  iree_status_t status = iree_runtime_session_create_with_device(
      instance, &session_options, device,
      iree_runtime_instance_host_allocator(instance), &session);
  iree_hal_device_release(device);

  // Load the compiled user module in a demo-specific way.
  // Applications could specify files, embed the outputs directly in their
  // binaries, fetch them over the network, etc.
  if (iree_status_is_ok(status)) {
    status = iree_runtime_demo_load_module(session);
  }

  // Build and issue the call.
  if (iree_status_is_ok(status)) {
    status = iree_runtime_demo_perform_mul(session);
  }

  // Release the session and free all resources.
  iree_runtime_session_release(session);
  return status;
}

//===----------------------------------------------------------------------===//
// 3. Call a function within a module with buffer views
//===----------------------------------------------------------------------===//
// The inputs and outputs of a call are reusable across calls (and possibly
// across sessions depending on device compatibility) and can be setup by the
// application as needed. For example, an application could perform
// multi-threaded buffer view creation and then issue the call from a single
// thread when all inputs are ready. This simple demo just allocates them
// per-call and throws them away.

// Sets up and calls the simple_mul function and dumps the results:
// func.func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) ->
// tensor<4xf32>
//
// NOTE: this is a demo and as such this performs no memoization; a real
// application could reuse a lot of these structures and cache lookups of
// iree_vm_function_t to reduce the amount of per-call overhead.
static iree_status_t iree_runtime_demo_perform_mul(
    iree_runtime_session_t* session) {
  // Initialize the call to the function.
  iree_runtime_call_t call;
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view("module.simple_mul"), &call));

  // Append the function inputs with the HAL device allocator in use by the
  // session. The buffers will be usable within the session and _may_ be usable
  // in other sessions depending on whether they share a compatible device.
  iree_hal_allocator_t* device_allocator =
      iree_runtime_session_device_allocator(session);
  iree_allocator_t host_allocator =
      iree_runtime_session_host_allocator(session);
  iree_status_t status = iree_ok_status();
  {
    // %arg0: tensor<4xf32>
    iree_hal_buffer_view_t* arg0 = NULL;
    if (iree_status_is_ok(status)) {
      static const iree_hal_dim_t arg0_shape[1] = {4};
      static const float arg0_data[4] = {1.0f, 1.1f, 1.2f, 1.3f};
      status = iree_hal_buffer_view_allocate_buffer(
          device_allocator,
          // Shape rank and dimensions:
          IREE_ARRAYSIZE(arg0_shape), arg0_shape,
          // Element type:
          IREE_HAL_ELEMENT_TYPE_FLOAT_32,
          // Encoding type:
          IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
          (iree_hal_buffer_params_t){
              // Where to allocate (host or device):
              .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
              // Access to allow to this memory (this is .rodata so READ only):
              .access = IREE_HAL_MEMORY_ACCESS_READ,
              // Intended usage of the buffer (transfers, dispatches, etc):
              .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
          },
          // The actual heap buffer to wrap or clone and its allocator:
          iree_make_const_byte_span(arg0_data, sizeof(arg0_data)),
          // Buffer view + storage are returned and owned by the caller:
          &arg0);
    }
    if (iree_status_is_ok(status)) {
      IREE_IGNORE_ERROR(iree_hal_buffer_view_fprint(
          stdout, arg0, /*max_element_count=*/4096, host_allocator));
      // Add to the call inputs list (which retains the buffer view).
      status = iree_runtime_call_inputs_push_back_buffer_view(&call, arg0);
    }
    // Since the call retains the buffer view we can release it here.
    iree_hal_buffer_view_release(arg0);

    fprintf(stdout, "\n * \n");

    // %arg1: tensor<4xf32>
    iree_hal_buffer_view_t* arg1 = NULL;
    if (iree_status_is_ok(status)) {
      static const iree_hal_dim_t arg1_shape[1] = {4};
      static const float arg1_data[4] = {10.0f, 100.0f, 1000.0f, 10000.0f};
      status = iree_hal_buffer_view_allocate_buffer(
          device_allocator, IREE_ARRAYSIZE(arg1_shape), arg1_shape,
          IREE_HAL_ELEMENT_TYPE_FLOAT_32,
          IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
          (iree_hal_buffer_params_t){
              .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
              .access = IREE_HAL_MEMORY_ACCESS_READ,
              .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
          },
          iree_make_const_byte_span(arg1_data, sizeof(arg1_data)), &arg1);
    }
    if (iree_status_is_ok(status)) {
      IREE_IGNORE_ERROR(iree_hal_buffer_view_fprint(
          stdout, arg1, /*max_element_count=*/4096, host_allocator));
      status = iree_runtime_call_inputs_push_back_buffer_view(&call, arg1);
    }
    iree_hal_buffer_view_release(arg1);
  }

  // Synchronously perform the call.
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_invoke(&call, /*flags=*/0);
  }

  fprintf(stdout, "\n = \n");

  // Dump the function outputs.
  iree_hal_buffer_view_t* ret0 = NULL;
  if (iree_status_is_ok(status)) {
    // Try to get the first call result as a buffer view.
    status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &ret0);
  }
  if (iree_status_is_ok(status)) {
    // This prints the buffer view out but an application could read its
    // contents, pass it to another call, etc.
    status = iree_hal_buffer_view_fprint(
        stdout, ret0, /*max_element_count=*/4096, host_allocator);
  }
  iree_hal_buffer_view_release(ret0);

  iree_runtime_call_deinitialize(&call);
  return status;
}
