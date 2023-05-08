// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/runtime/call.h"

#include <stddef.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/modules/hal/module.h"
#include "iree/runtime/session.h"

//===----------------------------------------------------------------------===//
// iree_runtime_call_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_runtime_call_initialize(
    iree_runtime_session_t* session, iree_vm_function_t function,
    iree_runtime_call_t* out_call) {
  IREE_ASSERT_ARGUMENT(session);
  IREE_ASSERT_ARGUMENT(out_call);
  memset(out_call, 0, sizeof(*out_call));

  // Query the signature of the function to determine the sizes of the lists.
  iree_vm_function_signature_t signature =
      iree_vm_function_signature(&function);
  iree_string_view_t arguments;
  iree_string_view_t results;
  IREE_RETURN_IF_ERROR(iree_vm_function_call_get_cconv_fragments(
      &signature, &arguments, &results));

  out_call->session = session;
  iree_runtime_session_retain(session);
  out_call->function = function;

  // Allocate the input and output lists with the required capacity.
  // A user wanting to avoid dynamic allocations could instead create on-stack
  // storage for these and use iree_vm_list_initialize instead. This high-level
  // API keeps things simple, though, and for the frequency of calls through
  // this interface a few small pooled malloc calls should be fine.
  iree_allocator_t host_allocator =
      iree_runtime_session_host_allocator(session);
  iree_status_t status =
      iree_vm_list_create(iree_vm_make_undefined_type_def(), arguments.size,
                          host_allocator, &out_call->inputs);
  if (iree_status_is_ok(status)) {
    status =
        iree_vm_list_create(iree_vm_make_undefined_type_def(), results.size,
                            host_allocator, &out_call->outputs);
  }

  if (!iree_status_is_ok(status)) {
    iree_runtime_call_deinitialize(out_call);
  }
  return status;
}

IREE_API_EXPORT iree_status_t iree_runtime_call_initialize_by_name(
    iree_runtime_session_t* session, iree_string_view_t full_name,
    iree_runtime_call_t* out_call) {
  iree_vm_function_t function;
  IREE_RETURN_IF_ERROR(
      iree_runtime_session_lookup_function(session, full_name, &function));
  return iree_runtime_call_initialize(session, function, out_call);
}

IREE_API_EXPORT void iree_runtime_call_deinitialize(iree_runtime_call_t* call) {
  IREE_ASSERT_ARGUMENT(call);
  iree_vm_list_release(call->inputs);
  iree_vm_list_release(call->outputs);
  iree_runtime_session_release(call->session);
}

IREE_API_EXPORT void iree_runtime_call_reset(iree_runtime_call_t* call) {
  IREE_ASSERT_ARGUMENT(call);
  iree_status_ignore(iree_vm_list_resize(call->inputs, 0));
  iree_status_ignore(iree_vm_list_resize(call->outputs, 0));
}

IREE_API_EXPORT iree_vm_list_t* iree_runtime_call_inputs(
    const iree_runtime_call_t* call) {
  IREE_ASSERT_ARGUMENT(call);
  return call->inputs;
}

IREE_API_EXPORT iree_vm_list_t* iree_runtime_call_outputs(
    const iree_runtime_call_t* call) {
  IREE_ASSERT_ARGUMENT(call);
  return call->outputs;
}

IREE_API_EXPORT iree_status_t iree_runtime_call_invoke(
    iree_runtime_call_t* call, iree_runtime_call_flags_t flags) {
  return iree_runtime_session_call(call->session, &call->function, call->inputs,
                                   call->outputs);
}

//===----------------------------------------------------------------------===//
// Helpers for defining call I/O
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_runtime_call_inputs_push_back_buffer_view(
    iree_runtime_call_t* call, iree_hal_buffer_view_t* buffer_view) {
  IREE_ASSERT_ARGUMENT(call);
  IREE_ASSERT_ARGUMENT(buffer_view);
  iree_vm_ref_t value = {0};
  IREE_RETURN_IF_ERROR(iree_vm_ref_wrap_assign(
      buffer_view, iree_hal_buffer_view_type(), &value));
  return iree_vm_list_push_ref_retain(call->inputs, &value);
}

// Pops a buffer view from the front of the call outputs list.
// Ownership of the buffer view transfers to the caller.
IREE_API_EXPORT iree_status_t iree_runtime_call_outputs_pop_front_buffer_view(
    iree_runtime_call_t* call, iree_hal_buffer_view_t** out_buffer_view) {
  IREE_ASSERT_ARGUMENT(call);
  IREE_ASSERT_ARGUMENT(out_buffer_view);
  *out_buffer_view = NULL;
  iree_vm_ref_t value = {0};
  IREE_RETURN_IF_ERROR(iree_vm_list_pop_front_ref_move(call->outputs, &value));
  return iree_hal_buffer_view_check_deref(value, out_buffer_view);
}
