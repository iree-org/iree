// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/runtime/session.h"

#include <stddef.h>
#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/base/internal/file_io.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/runtime/instance.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"

//===----------------------------------------------------------------------===//
// iree_runtime_session_options_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_runtime_session_options_initialize(
    iree_runtime_session_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
  out_options->context_flags = IREE_VM_CONTEXT_FLAG_NONE;
  out_options->builtin_modules = IREE_RUNTIME_SESSION_BUILTIN_ALL;
}

//===----------------------------------------------------------------------===//
// iree_runtime_session_t
//===----------------------------------------------------------------------===//

struct iree_runtime_session_t {
  iree_atomic_ref_count_t ref_count;

  // Allocator used to allocate the session and all of its resources.
  // Independent sessions within the same instance can have unique allocators to
  // enable session-level tagging of allocations and pooling.
  iree_allocator_t host_allocator;

  // The instance this session is a part of; may be shared across many sessions.
  // Devices and pools are stored on the instance so that multiple sessions can
  // share resources. The session will keep the instance retained for its
  // lifetime to ensure that these resources remain available.
  iree_runtime_instance_t* instance;

  // VM context containing the loaded modules (both builtins and user).
  // Thread-compatible; a context carries state that must be externally
  // synchronized.
  iree_vm_context_t* context;

  // The HAL module state bound to the target devices.
  // This is used internally by the loaded modules to interact with the devices
  // but can also be used by the caller to perform allocation and custom device
  // execution.
  //
  // The state is owned by the context and we have it cached here for faster
  // lookup. An application directly using the API may never need this, or could
  // perform VM calls into HAL module exports to gain more portability.
  iree_vm_module_state_t* hal_module_state;
};

IREE_API_EXPORT iree_status_t iree_runtime_session_create_with_device(
    iree_runtime_instance_t* instance,
    const iree_runtime_session_options_t* options, iree_hal_device_t* device,
    iree_allocator_t host_allocator, iree_runtime_session_t** out_session) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_session);
  *out_session = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate the session state.
  iree_runtime_session_t* session = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*session),
                                (void**)&session));
  session->host_allocator = host_allocator;
  iree_atomic_ref_count_init(&session->ref_count);

  session->instance = instance;
  iree_runtime_instance_retain(session->instance);

  // Create the context empty so that we can add our modules to it.
  iree_status_t status = iree_vm_context_create(
      iree_runtime_instance_vm_instance(instance), options->context_flags,
      host_allocator, &session->context);

  // Add the HAL module; it is always required when using the runtime API.
  // Lower-level usage of the VM can avoid the HAL if it's not required.
  iree_vm_module_t* hal_module = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_module_create(iree_runtime_instance_vm_instance(instance),
                                    device, IREE_HAL_MODULE_FLAG_NONE,
                                    host_allocator, &hal_module);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_context_register_modules(
        session->context, /*module_count=*/1, /*modules=*/&hal_module);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_context_resolve_module_state(session->context, hal_module,
                                                  &session->hal_module_state);
  }
  iree_vm_module_release(hal_module);

  if (iree_status_is_ok(status)) {
    *out_session = session;
  } else {
    iree_runtime_session_release(session);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_runtime_session_destroy(iree_runtime_session_t* session) {
  IREE_ASSERT_ARGUMENT(session);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_vm_context_release(session->context);
  iree_runtime_instance_release(session->instance);

  iree_allocator_free(session->host_allocator, session);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT void iree_runtime_session_retain(
    iree_runtime_session_t* session) {
  if (session) {
    iree_atomic_ref_count_inc(&session->ref_count);
  }
}

IREE_API_EXPORT void iree_runtime_session_release(
    iree_runtime_session_t* session) {
  if (session && iree_atomic_ref_count_dec(&session->ref_count) == 1) {
    iree_runtime_session_destroy(session);
  }
}

IREE_API_EXPORT iree_allocator_t
iree_runtime_session_host_allocator(const iree_runtime_session_t* session) {
  IREE_ASSERT_ARGUMENT(session);
  return session->host_allocator;
}

IREE_API_EXPORT iree_runtime_instance_t* iree_runtime_session_instance(
    const iree_runtime_session_t* session) {
  IREE_ASSERT_ARGUMENT(session);
  return session->instance;
}

IREE_API_EXPORT iree_vm_context_t* iree_runtime_session_context(
    const iree_runtime_session_t* session) {
  IREE_ASSERT_ARGUMENT(session);
  return session->context;
}

IREE_API_EXPORT iree_hal_device_t* iree_runtime_session_device(
    const iree_runtime_session_t* session) {
  IREE_ASSERT_ARGUMENT(session);
  return iree_hal_module_state_device(session->hal_module_state);
}

IREE_API_EXPORT iree_hal_allocator_t* iree_runtime_session_device_allocator(
    const iree_runtime_session_t* session) {
  iree_hal_device_t* device = iree_runtime_session_device(session);
  if (!device) return NULL;
  return iree_hal_device_allocator(device);
}

IREE_API_EXPORT iree_status_t
iree_runtime_session_trim(iree_runtime_session_t* session) {
  IREE_ASSERT_ARGUMENT(session);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_vm_context_notify(
      iree_runtime_session_context(session), IREE_VM_SIGNAL_LOW_MEMORY);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_runtime_session_append_module(
    iree_runtime_session_t* session, iree_vm_module_t* module) {
  IREE_ASSERT_ARGUMENT(session);
  IREE_ASSERT_ARGUMENT(module);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, iree_vm_module_name(module).data,
                              iree_vm_module_name(module).size);

  iree_status_t status =
      iree_vm_context_register_modules(iree_runtime_session_context(session),
                                       /*module_count=*/1, /*modules=*/&module);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_runtime_session_append_bytecode_module_from_memory(
    iree_runtime_session_t* session, iree_const_byte_span_t flatbuffer_data,
    iree_allocator_t flatbuffer_allocator) {
  IREE_ASSERT_ARGUMENT(session);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_vm_module_t* module = NULL;
  iree_status_t status = iree_vm_bytecode_module_create(
      iree_runtime_instance_vm_instance(session->instance), flatbuffer_data,
      flatbuffer_allocator, iree_runtime_session_host_allocator(session),
      &module);
  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_append_module(session, module);
  }
  iree_vm_module_release(module);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_runtime_session_append_bytecode_module_from_file(
    iree_runtime_session_t* session, const char* file_path) {
  IREE_ASSERT_ARGUMENT(session);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, file_path);

  // TODO(#3909): actually map the memory here. For now we just load the
  // contents.
  iree_file_contents_t* flatbuffer_contents = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_file_read_contents(file_path,
                                  iree_runtime_session_host_allocator(session),
                                  &flatbuffer_contents));

  iree_status_t status =
      iree_runtime_session_append_bytecode_module_from_memory(
          session, flatbuffer_contents->const_buffer,
          iree_file_contents_deallocator(flatbuffer_contents));

  if (!iree_status_is_ok(status)) {
    iree_file_contents_free(flatbuffer_contents);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_runtime_session_append_bytecode_module_from_stdin(
    iree_runtime_session_t* session) {
  IREE_ASSERT_ARGUMENT(session);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_file_contents_t* flatbuffer_contents = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_stdin_read_contents(iree_runtime_session_host_allocator(session),
                                   &flatbuffer_contents));

  iree_status_t status =
      iree_runtime_session_append_bytecode_module_from_memory(
          session, flatbuffer_contents->const_buffer,
          iree_file_contents_deallocator(flatbuffer_contents));

  if (!iree_status_is_ok(status)) {
    iree_file_contents_free(flatbuffer_contents);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_runtime_session_lookup_function(
    const iree_runtime_session_t* session, iree_string_view_t full_name,
    iree_vm_function_t* out_function) {
  IREE_ASSERT_ARGUMENT(session);
  IREE_ASSERT_ARGUMENT(out_function);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_vm_context_resolve_function(
      iree_runtime_session_context(session), full_name, out_function);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_runtime_session_call(
    iree_runtime_session_t* session, const iree_vm_function_t* function,
    iree_vm_list_t* input_list, iree_vm_list_t* output_list) {
  IREE_ASSERT_ARGUMENT(session);
  IREE_ASSERT_ARGUMENT(function);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status =
      iree_vm_invoke(iree_runtime_session_context(session), *function,
                     IREE_VM_INVOCATION_FLAG_NONE,
                     /*policy=*/NULL, input_list, output_list,
                     iree_runtime_session_host_allocator(session));

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_runtime_session_call_by_name(
    iree_runtime_session_t* session, iree_string_view_t full_name,
    iree_vm_list_t* input_list, iree_vm_list_t* output_list) {
  IREE_ASSERT_ARGUMENT(session);
  iree_vm_function_t function;
  IREE_RETURN_IF_ERROR(
      iree_runtime_session_lookup_function(session, full_name, &function));
  return iree_runtime_session_call(session, &function, input_list, output_list);
}

IREE_API_EXPORT iree_status_t iree_runtime_session_call_direct(
    iree_runtime_session_t* session, const iree_vm_function_call_t call) {
  IREE_ASSERT_ARGUMENT(session);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate a VM stack on the host stack and initialize it.
  iree_vm_context_t* context = iree_runtime_session_context(session);
  IREE_VM_INLINE_STACK_INITIALIZE(stack, IREE_VM_INVOCATION_FLAG_NONE,
                                  iree_vm_context_state_resolver(context),
                                  iree_runtime_session_host_allocator(session));

  // Issue the call.
  iree_status_t status =
      call.function.module->begin_call(call.function.module->self, stack, call);

  // Cleanup the stack.
  iree_vm_stack_deinitialize(stack);

  IREE_TRACE_ZONE_END(z0);
  return status;
}
