// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/invocation.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/vm/ref.h"
#include "iree/vm/stack.h"
#include "iree/vm/value.h"

//===----------------------------------------------------------------------===//
// Invocation utilities for I/O
//===----------------------------------------------------------------------===//

static void iree_vm_invoke_release_io_storage(iree_string_view_t cconv_fragment,
                                              iree_byte_span_t storage) {
  if (!storage.data_length) return;
  if (cconv_fragment.size == 0 || cconv_fragment.data[0] != '0') return;
  uint8_t* p = storage.data;
  for (iree_host_size_t i = 1; i < cconv_fragment.size; ++i) {
    char c = cconv_fragment.data[i];
    switch (c) {
      default:
        IREE_ASSERT_UNREACHABLE("calling convention/FFI mismatch");
        break;
      case IREE_VM_CCONV_TYPE_VOID:
        break;
      case IREE_VM_CCONV_TYPE_I32:
      case IREE_VM_CCONV_TYPE_F32:
        p += sizeof(int32_t);
        break;
      case IREE_VM_CCONV_TYPE_I64:
      case IREE_VM_CCONV_TYPE_F64:
        p += sizeof(int64_t);
        break;
      case IREE_VM_CCONV_TYPE_REF:
        iree_vm_ref_release((iree_vm_ref_t*)p);
        p += sizeof(iree_vm_ref_t);
        break;
    }
  }
}

// Marshals caller arguments from the variant list to the ABI convention.
static iree_status_t iree_vm_invoke_marshal_inputs(
    iree_string_view_t cconv_arguments, const iree_vm_list_t* inputs,
    iree_byte_span_t arguments) {
  // We are 1:1 right now with no variadic args, so do a quick verification on
  // the input list.
  iree_host_size_t expected_input_count =
      cconv_arguments.size > 0
          ? (cconv_arguments.data[0] == 'v' ? 0 : cconv_arguments.size)
          : 0;
  if (IREE_UNLIKELY(!inputs)) {
    if (IREE_UNLIKELY(expected_input_count > 0)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "no input provided to a function that has inputs");
    }
    return iree_ok_status();
  } else if (IREE_UNLIKELY(expected_input_count != iree_vm_list_size(inputs))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "input list and function mismatch; expected %zu "
                            "arguments but passed %zu",
                            expected_input_count, iree_vm_list_size(inputs));
  }

  uint8_t* p = arguments.data;
  for (iree_host_size_t cconv_i = 0, arg_i = 0; cconv_i < cconv_arguments.size;
       ++cconv_i, ++arg_i) {
    switch (cconv_arguments.data[cconv_i]) {
      case IREE_VM_CCONV_TYPE_VOID:
        break;
      case IREE_VM_CCONV_TYPE_I32: {
        iree_vm_value_t value;
        IREE_RETURN_IF_ERROR(iree_vm_list_get_value_as(
            inputs, arg_i, IREE_VM_VALUE_TYPE_I32, &value));
        memcpy(p, &value.i32, sizeof(int32_t));
        p += sizeof(int32_t);
      } break;
      case IREE_VM_CCONV_TYPE_I64: {
        iree_vm_value_t value;
        IREE_RETURN_IF_ERROR(iree_vm_list_get_value_as(
            inputs, arg_i, IREE_VM_VALUE_TYPE_I64, &value));
        memcpy(p, &value.i64, sizeof(int64_t));
        p += sizeof(int64_t);
      } break;
      case IREE_VM_CCONV_TYPE_F32: {
        iree_vm_value_t value;
        IREE_RETURN_IF_ERROR(iree_vm_list_get_value_as(
            inputs, arg_i, IREE_VM_VALUE_TYPE_F32, &value));
        memcpy(p, &value.f32, sizeof(float));
        p += sizeof(float);
      } break;
      case IREE_VM_CCONV_TYPE_F64: {
        iree_vm_value_t value;
        IREE_RETURN_IF_ERROR(iree_vm_list_get_value_as(
            inputs, arg_i, IREE_VM_VALUE_TYPE_F64, &value));
        memcpy(p, &value.f64, sizeof(double));
        p += sizeof(double);
      } break;
      case IREE_VM_CCONV_TYPE_REF: {
        // TODO(benvanik): see if we can't remove this retain by instead relying
        // on the caller still owning the list.
        IREE_RETURN_IF_ERROR(
            iree_vm_list_get_ref_retain(inputs, arg_i, (iree_vm_ref_t*)p));
        p += sizeof(iree_vm_ref_t);
      } break;
    }
  }
  return iree_ok_status();
}

// Marshals callee results from the ABI convention to the variant list.
static iree_status_t iree_vm_invoke_marshal_outputs(
    iree_string_view_t cconv_results, iree_byte_span_t results,
    iree_vm_list_t* outputs) {
  iree_host_size_t expected_output_count =
      cconv_results.size > 0
          ? (cconv_results.data[0] == 'v' ? 0 : cconv_results.size)
          : 0;
  if (IREE_UNLIKELY(!outputs)) {
    if (IREE_UNLIKELY(expected_output_count > 0)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "no output provided to a function that has outputs");
    }
    return iree_ok_status();
  }

  // Resize the output list to hold all results (and kill anything that may
  // have been in there).
  // TODO(benvanik): list method for resetting to a new size.
  IREE_RETURN_IF_ERROR(iree_vm_list_resize(outputs, 0));
  IREE_RETURN_IF_ERROR(iree_vm_list_resize(outputs, expected_output_count));

  uint8_t* p = results.data;
  for (iree_host_size_t cconv_i = 0, arg_i = 0; cconv_i < cconv_results.size;
       ++cconv_i, ++arg_i) {
    switch (cconv_results.data[cconv_i]) {
      case IREE_VM_CCONV_TYPE_VOID:
        break;
      case IREE_VM_CCONV_TYPE_I32: {
        iree_vm_value_t value = iree_vm_value_make_i32(*(int32_t*)p);
        IREE_RETURN_IF_ERROR(iree_vm_list_set_value(outputs, arg_i, &value));
        p += sizeof(int32_t);
      } break;
      case IREE_VM_CCONV_TYPE_I64: {
        iree_vm_value_t value = iree_vm_value_make_i64(*(int64_t*)p);
        IREE_RETURN_IF_ERROR(iree_vm_list_set_value(outputs, arg_i, &value));
        p += sizeof(int64_t);
      } break;
      case IREE_VM_CCONV_TYPE_F32: {
        iree_vm_value_t value = iree_vm_value_make_f32(*(float*)p);
        IREE_RETURN_IF_ERROR(iree_vm_list_set_value(outputs, arg_i, &value));
        p += sizeof(float);
      } break;
      case IREE_VM_CCONV_TYPE_F64: {
        iree_vm_value_t value = iree_vm_value_make_f64(*(double*)p);
        IREE_RETURN_IF_ERROR(iree_vm_list_set_value(outputs, arg_i, &value));
        p += sizeof(double);
      } break;
      case IREE_VM_CCONV_TYPE_REF: {
        IREE_RETURN_IF_ERROR(
            iree_vm_list_set_ref_move(outputs, arg_i, (iree_vm_ref_t*)p));
        p += sizeof(iree_vm_ref_t);
      } break;
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Synchronous invocation
//===----------------------------------------------------------------------===//

static void iree_vm_invoke_fiber_enter(iree_vm_context_t* context) {
  IREE_TRACE_FIBER_ENTER(iree_vm_context_id(context));
}

static void iree_vm_invoke_fiber_reenter(iree_vm_context_t* context,
                                         iree_vm_stack_t* stack) {
  IREE_TRACE_FIBER_ENTER(iree_vm_context_id(context));
  iree_vm_stack_resume_trace_zones(stack);
}

static void iree_vm_invoke_fiber_leave(iree_vm_context_t* context,
                                       iree_vm_stack_t* stack) {
  if (stack) iree_vm_stack_suspend_trace_zones(stack);
  IREE_TRACE_FIBER_LEAVE();
}

IREE_API_EXPORT iree_status_t iree_vm_invoke(
    iree_vm_context_t* context, iree_vm_function_t function,
    iree_vm_invocation_flags_t flags, const iree_vm_invocation_policy_t* policy,
    const iree_vm_list_t* inputs, iree_vm_list_t* outputs,
    iree_allocator_t host_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Bound the synchronous invocation to the timeout specified by the user
  // regardless of what the target of the invocation wants when it waits.
  // TODO(benvanik): add a timeout arg to iree_vm_invoke.
  // For now we only use the timeouts specified on the wait operations.
  iree_timeout_t timeout = iree_infinite_timeout();
  iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);

  // Begin a zone outside the fiber to represent one tick of the loop.
  IREE_TRACE_ZONE_BEGIN_NAMED(zi, "iree_vm_invoke_tick");
  // Enter the fiber to start attributing zones to the context.
  iree_vm_invoke_fiber_enter(context);

  // Perform the initial invocation step, which if synchronous may fully
  // complete the invocation before returning. If it yields we'll need to resume
  // it, possibly after taking care of pending waits.
  iree_vm_invoke_state_t state;
  iree_status_t status = iree_vm_begin_invoke(&state, context, function, flags,
                                              policy, inputs, host_allocator);
  while (iree_status_is_deferred(status)) {
    // Grab the wait frame from the stack holding the wait parameters.
    // This is optional: if an invocation yields for cooperative scheduling
    // purposes there will not be a wait frame on the stack and we'll just
    // resume it below.
    iree_vm_stack_frame_t* current_frame =
        iree_vm_stack_current_frame(state.stack);
    if (IREE_UNLIKELY(!current_frame)) {
      // Unbalanced stack.
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "unbalanced stack after yield");
      break;  // bail and don't attempt a resume
    } else if (current_frame->type == IREE_VM_STACK_FRAME_WAIT) {
      // Perform the wait operation synchronously.
      // We do this outside of the fiber to match accounting with async
      // executors.
      iree_vm_invoke_fiber_leave(context, state.stack);
      IREE_TRACE_ZONE_END(zi);

      iree_vm_wait_frame_t* wait_frame =
          (iree_vm_wait_frame_t*)iree_vm_stack_frame_storage(current_frame);
      status = iree_vm_wait_invoke(&state, wait_frame, deadline_ns);

      // Restore tick zone and re-enter the fiber for the resume.
      IREE_TRACE_ZONE_BEGIN_NAMED(zi_next, "iree_vm_invoke_tick");
      zi = zi_next;
      iree_vm_invoke_fiber_reenter(context, state.stack);
      if (!iree_status_is_ok(status)) break;
    }

    // Resume the invocation after its wait completes (if it wasn't just a
    // simple yield for cooperation). This may yield again and require another
    // tick or complete with OK (or an error).
    status = iree_vm_resume_invoke(&state);
  }

  // If the invoke process itself was successful we can end the invocation
  // cleanly and get the invocation status as returned by the target function.
  iree_status_t invoke_status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    status = iree_vm_end_invoke(&state, outputs, &invoke_status);
  }

  // Otherwise if we failed to invoke we need to tear down the state to release
  // all resources retained by the stack.
  if (!iree_status_is_ok(status)) {
    // Cleanup the invocation state if the end wasn't able to.
    // This may leave the context in an unexpected state but the caller is
    // expected to tear down everything if this happens.
    iree_vm_abort_invoke(&state);
  }

  // Leave the fiber context now that execution has completed.
  iree_vm_invoke_fiber_leave(context, state.stack);
  IREE_TRACE_ZONE_END(zi);

  // If we succeeded at invoking the status will be OK and the invoke_status
  // will hold the status returned by the invokee. If we failed at invoking
  // the invoke_status won't be set.
  IREE_ASSERT(iree_status_is_ok(status) ||
              (!iree_status_is_ok(status) && iree_status_is_ok(invoke_status)));
  status = !iree_status_is_ok(invoke_status) ? invoke_status : status;

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Asynchronous invocation
//===----------------------------------------------------------------------===//

// WARNING: this function cannot have any trace markers that span the begin
// call; the begin may yield with zones still open.
IREE_API_EXPORT iree_status_t iree_vm_begin_invoke(
    iree_vm_invoke_state_t* state, iree_vm_context_t* context,
    iree_vm_function_t function, iree_vm_invocation_flags_t flags,
    const iree_vm_invocation_policy_t* policy, const iree_vm_list_t* inputs,
    iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Force tracing if specified on the context.
  if (iree_vm_context_flags(context) & IREE_VM_CONTEXT_FLAG_TRACE_EXECUTION) {
    flags |= IREE_VM_INVOCATION_FLAG_TRACE_EXECUTION;
  }

  // Grab function metadata used for marshaling inputs/outputs.
  iree_vm_function_signature_t signature =
      iree_vm_function_signature(&function);
  iree_string_view_t cconv_arguments = iree_string_view_empty();
  iree_string_view_t cconv_results = iree_string_view_empty();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_vm_function_call_get_cconv_fragments(
              &signature, &cconv_arguments, &cconv_results));

  // Allocate argument storage on the native stack. It only needs to survive the
  // begin call as it's consumed by the invokee.
  iree_byte_span_t arguments = iree_make_byte_span(NULL, 0);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_vm_function_call_compute_cconv_fragment_size(
          cconv_arguments, /*segment_size_list=*/NULL, &arguments.data_length));
  arguments.data = iree_alloca(arguments.data_length);
  memset(arguments.data, 0, arguments.data_length);

  // Allocate the result storage that will be populated by the invokee. This
  // must survive until the end() so we slice it off the bottom of the stack
  // storage. This reduces the overall available stack space but not by much,
  // and if the stack needs to dynamically grow the inlined storage will still
  // be available.
  iree_byte_span_t results = iree_make_byte_span(state->stack_storage, 0);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_vm_function_call_compute_cconv_fragment_size(
              cconv_results, /*segment_size_list=*/NULL, &results.data_length));
  memset(results.data, 0, results.data_length);
  iree_host_size_t result_storage_size =
      iree_host_align(results.data_length, iree_max_align_t);
  if (result_storage_size > sizeof(state->stack_storage)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "too many results for inlined storage (%" PRIhsz
                            ")",
                            result_storage_size);
  }

  iree_vm_function_call_t call = {
      .function = function,
      .arguments = arguments,
      .results = results,
  };

  // Marshal the input arguments into the VM ABI and preallocate the result
  // buffer. If marshaling fails we need to cleanup the arguments.
  // NOTE: today we don't support variadic arguments through this interface.
  iree_status_t status =
      iree_vm_invoke_marshal_inputs(cconv_arguments, inputs, arguments);
  if (!iree_status_is_ok(status)) {
    iree_vm_invoke_release_io_storage(cconv_arguments, arguments);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Initialize the stack with the inline storage.
  // We sliced off the head of the storage above to use for results and perform
  // an offset here to account for that.
  iree_vm_stack_t* stack = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_vm_stack_initialize(
              iree_make_byte_span(
                  state->stack_storage + result_storage_size,
                  sizeof(state->stack_storage) - result_storage_size),
              flags, iree_vm_context_state_resolver(context), host_allocator,
              &stack));

  // NOTE: at this point the stack must be properly deinitialized if we bail.

  // Initialize state now that we are confident we're returning OK.
  // If we return a failure the user won't know they have to end() and clean
  // these up.
  state->context = context;
  state->cconv_results = cconv_results;
  state->results = results;
  iree_vm_context_retain(context);
  state->stack = stack;

  // NOTE: we must end the zone here as the begin_call will return with
  // unbalanced zones if we yield.
  IREE_TRACE_ZONE_END(z0);

  // Execute the target function until the first yield point is reached or it
  // completes. A result of OK indicates successful completion while DEFERRED
  // indicates that the invocation needs to be resumed/waited again.
  iree_vm_execution_result_t result;
  state->status =
      function.module->begin_call(function.module->self, stack, &call, &result);

  // The call may have yielded, either for cooperative scheduling purposes or
  // for a wait operation (in which case the top of the stack will have a wait
  // frame).
  if (iree_status_is_deferred(state->status)) {
    return iree_status_from_code(IREE_STATUS_DEFERRED);
  } else if (IREE_UNLIKELY(!iree_status_is_ok(state->status))) {
    // If the call succeeded the arguments will have been consumed - otherwise
    // we're responsible for doing it.
    iree_vm_invoke_release_io_storage(cconv_arguments, call.arguments);
  }

  // NOTE: the begin-invoke was ok, but the operation itself may have failed.
  return iree_ok_status();
}

// WARNING: this function cannot have any trace markers that span the resume
// call; the resume may yield with zones still open.
IREE_API_EXPORT iree_status_t
iree_vm_resume_invoke(iree_vm_invoke_state_t* state) {
  IREE_ASSERT_ARGUMENT(state);
  if (iree_status_is_deferred(state->status)) {
    // Wait required; top of the stack should be a wait frame.
    IREE_ASSERT_EQ(iree_vm_stack_current_frame(state->stack)->type,
                   IREE_VM_STACK_FRAME_WAIT);
    return iree_status_from_code(IREE_STATUS_DEFERRED);
  } else if (!iree_status_is_ok(state->status)) {
    // Invocation previously failed so return immediately. The user should then
    // call end() to get the result. By returning OK here we are telling the
    // user the resume operation succeeded.
    return iree_ok_status();
  }

  // Get the top execution frame of the stack where we will resume execution.
  iree_vm_stack_frame_t* resume_frame = iree_vm_stack_top(state->stack);
  if (IREE_UNLIKELY(!resume_frame)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "resume called with no parent frame");
  }

  // Call into the VM to resume the function. It may complete (returning OK),
  // defer to be waited/resumed later, or fail.
  iree_vm_function_t resume_function = resume_frame->function;
  iree_vm_execution_result_t result;
  state->status = resume_function.module->resume_call(
      resume_function.module->self, state->stack, state->results, &result);

  // If the call yielded then return that so the user knows to resume again.
  if (iree_status_is_deferred(state->status)) {
    return iree_status_from_code(IREE_STATUS_DEFERRED);
  }

  // Stack resume: if the resume succeeded but the stack is not empty it means
  // we've got to resume the parent frame. When we do a full yield up to the
  // scheduler and then resume we're calling into the VM stack top from the host
  // stack bottom - to have the same behavior as a normal stack pop we've got to
  // continue running. We signal this as a DEFERRED so the same machinery for
  // normal yields can be used.
  // TODO(benvanik): use another result type in cases where we want to
  // differentiate?
  if (iree_status_is_ok(state->status) &&
      iree_vm_stack_current_frame(state->stack) != NULL) {
    return iree_status_from_code(IREE_STATUS_DEFERRED);
  }

  // We're indicating the resume operation was successful, not the result of the
  // VM call; the user will call end() to get that.
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_vm_wait_invoke(iree_vm_invoke_state_t* state,
                    iree_vm_wait_frame_t* wait_frame, iree_time_t deadline_ns) {
  IREE_ASSERT_ARGUMENT(state);
  if (IREE_UNLIKELY(!iree_status_is_deferred(state->status))) {
    // Can only wait if the invocation is actually waiting.
    // We could make this OK and act as a no-op but it can be useful for
    // ensuring scheduler implementations don't do extraneous work.
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "wait-invoke attempted on a non-waiting invocation");
  }

  // Combine the wait-invoke deadline with the one specified by the wait
  // operation itself. This allows schedulers to timeslice waits without
  // worrying whether user programs request to wait forever.
  iree_time_t min_deadline_ns = iree_min(deadline_ns, wait_frame->deadline_ns);

  // Perform the wait operation, blocking the calling thread until it completes,
  // fails, or hits the min_deadline_ns.
  if (wait_frame->wait_type == IREE_VM_WAIT_UNTIL) {
    wait_frame->wait_status = iree_wait_until(min_deadline_ns)
                                  ? iree_ok_status()
                                  : iree_status_from_code(IREE_STATUS_ABORTED);
  } else if (wait_frame->count == 1) {
    wait_frame->wait_status = iree_wait_source_wait_one(
        wait_frame->wait_sources[0], iree_make_deadline(min_deadline_ns));
  } else {
    // TODO(benvanik): multi-wait when running synchronously. This is already
    // supported by iree_loop_inline_t and maybe we can just reuse that. These
    // are not currently emitted by the compiler.
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "multi-wait in synchronous invocations not yet implemented");
  }

  // Reset status to OK - the next resume will pick back up in the waiter.
  iree_status_free(state->status);
  state->status = iree_ok_status();

  // OK here indicates we performed the wait and not the result of the wait.
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_end_invoke(iree_vm_invoke_state_t* state,
                                                 iree_vm_list_t* outputs,
                                                 iree_status_t* out_status) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(out_status);
  *out_status = iree_ok_status();

  // Suspend stack frame tracing zones; if the invocation failed the failing
  // frames will still be on the stack and thus also still have their trace
  // zones active. If we were to begin a zone here and then deinit the stack
  // we'd end up with unbalanced zones.
  iree_vm_stack_suspend_trace_zones(state->stack);

  IREE_TRACE_ZONE_BEGIN(z0);

  // Grab operation status. If this is not OK it's because the operation failed
  // or the user is calling this with a wait frame on the stack.
  iree_status_t invoke_status = state->status;
  if (IREE_UNLIKELY(iree_status_is_deferred(invoke_status))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "end-invoke attempted on a waiting invocation");
  } else if (IREE_UNLIKELY(!iree_status_is_ok(invoke_status))) {
    // Annotate failures with the stack trace (if compiled in).
    invoke_status = IREE_VM_STACK_ANNOTATE_BACKTRACE_IF_ENABLED(state->stack,
                                                                invoke_status);
  }

  // If the operation succeeded marshal the outputs from the stack buffers into
  // the user-provided storage. The outputs list will retain all results.
  if (iree_status_is_ok(invoke_status)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_vm_invoke_marshal_outputs(state->cconv_results, state->results,
                                           outputs));
  }

  // Cleanup the invocation resources.
  *out_status = invoke_status;  // takes ownership
  state->status = iree_ok_status();
  iree_vm_abort_invoke(state);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT void iree_vm_abort_invoke(iree_vm_invoke_state_t* state) {
  // We expect that the caller has already suspended the stack tracing zones,
  // but in failure cases we can end up here and want to ensure that things are
  // cleaned up. If we were to begin a zone now the stack deinit would lead to
  // unbalanced zones.
  if (state->stack) iree_vm_stack_suspend_trace_zones(state->stack);

  IREE_TRACE_ZONE_BEGIN(z0);

  if (state->stack) {
    iree_vm_stack_deinitialize(state->stack);
    state->stack = NULL;
  }

  if (!iree_byte_span_is_empty(state->results)) {
    iree_vm_invoke_release_io_storage(state->cconv_results, state->results);
    state->results = iree_byte_span_empty();
  }

  if (state->context) {
    iree_vm_context_release(state->context);
    state->context = NULL;
  }

  iree_status_free(state->status);
  state->status = iree_status_from_code(IREE_STATUS_INTERNAL);

  IREE_TRACE_ZONE_END(z0);
}
