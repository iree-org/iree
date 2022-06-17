// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// See iree/base/api.h for documentation on the API conventions used.

#ifndef IREE_VM_INVOCATION_H_
#define IREE_VM_INVOCATION_H_

#include "iree/base/api.h"
#include "iree/vm/context.h"
#include "iree/vm/list.h"
#include "iree/vm/module.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_vm_invocation_t iree_vm_invocation_t;
typedef struct iree_vm_invocation_policy_t iree_vm_invocation_policy_t;

//===----------------------------------------------------------------------===//
// Synchronous invocation
//===----------------------------------------------------------------------===//

// Synchronously invokes a function in the VM.
// The function will be run to completion and may block on external resources.
// If more control is required or callers want to have multiple invocations
// in-flight then iree_vm_invocation_t should be used.
//
// |policy| is used to schedule the invocation relative to other pending or
// in-flight invocations. It may be omitted to leave the behavior up to the
// implementation.
//
// |inputs| is used to pass values and objects into the target function and must
// match the signature defined by the compiled function. List ownership remains
// with the caller.
//
// |outputs| is populated after the function completes execution with the
// output values and objects of the function. List ownership remains with the
// caller.
IREE_API_EXPORT iree_status_t iree_vm_invoke(
    iree_vm_context_t* context, iree_vm_function_t function,
    iree_vm_invocation_flags_t flags, const iree_vm_invocation_policy_t* policy,
    const iree_vm_list_t* inputs, iree_vm_list_t* outputs,
    iree_allocator_t host_allocator);

//===----------------------------------------------------------------------===//
// Asynchronous invocation
//===----------------------------------------------------------------------===//

// Invocation storage for passing state across invocation stages.
// This is intended to be embedded within higher-level invocation objects or
// used directly on the stack. Users should prefer iree_vm_invoke (which wraps
// up this sequence) or iree_vm_invocation_t. Deep integrations into existing
// schedulers can use this to perform fine-grained interleaving of invocations
// with external workloads.
//
// Usage (ala iree_vm_invoke):
//   iree_vm_invoke_state_t state;
//   iree_vm_wait_result_t wait_result;
//   s = iree_vm_begin_invoke(&state, ..., function, ..., inputs, ...);
//   while (iree_status_is_deferred(s)) {
//      if (wait_frame) iree_vm_wait_invoke(&state, ...);
//      s = iree_vm_resume_invoke(&state, outputs);
//   }
//   iree_vm_end_invoke(&state, outputs, &status);
//
// iree_vm_*_invoke calls return the status of the specific invoke operation and
// not the result of the invocation itself. A failure in one of the calls is
// generally non-recoverable and iree_vm_abort_invoke must be used to clean up
// resources. Otherwise if all return OK iree_vm_end_invoke must be called to
// retrieve the invocation result.
//
// Thread-compatible: steps in the invocation sequence may be made from any
// thread so long as none are made concurrently (this structure is not
// thread-safe).
typedef struct iree_vm_invoke_state_t {
  // Retains the context the invocation is running within.
  iree_vm_context_t* context;
  // Status of the invocation. Returned by iree_vm_end_invoke.
  iree_status_t status;
  // Parsed calling convention results string for marshaling.
  iree_string_view_t cconv_results;
  // Pointer into stack storage containing the results.
  iree_byte_span_t results;
  // VM stack used during the invocation. Will retain required resources
  // across invocation stages.
  iree_vm_stack_t* stack;
  // Inlined stack storage. If the stack grows larger than this amount
  // additional storage will be allocated automatically.
  uint8_t stack_storage[IREE_VM_STACK_DEFAULT_SIZE];
} iree_vm_invoke_state_t;

// Begins an invocation of |function| in |context| with the given |inputs|.
//
// |state| is caller-provided uninitialized storage for the stack and other
// invocation resources and must always be passed to iree_vm_end_invoke to
// clean up resources from the sequence.
//
// |policy| is used to schedule the invocation relative to other pending or
// in-flight invocations. It may be omitted to leave the behavior up to the
// implementation.
//
// |inputs| is used to pass values and objects into the target function and must
// match the signature defined by the target |function|. List contents are
// captured and the caller can reuse it immediately upon return.
//
// Returns OK if the invocation began regardless of the invocation result.
// When OK iree_vm_end_invoke is used to retrieve the invocation result.
// If IREE_STATUS_DEFERRED is returned one of iree_vm_resume_invoke or
// iree_vm_wait_invoke (or other wait frame handling) must be performed by the
// caller.
IREE_API_EXPORT iree_status_t iree_vm_begin_invoke(
    iree_vm_invoke_state_t* state, iree_vm_context_t* context,
    iree_vm_function_t function, iree_vm_invocation_flags_t flags,
    const iree_vm_invocation_policy_t* policy, const iree_vm_list_t* inputs,
    iree_allocator_t host_allocator);

// Resumes an invocation previously began with iree_vm_begin_invoke.
// Only valid to call if a prior call to iree_vm_begin_invoke or
// iree_vm_resume_invoke returned IREE_STATUS_DEFERRED.
//
// Returns OK if the invocation resumed regardless of the invocation result.
// When OK iree_vm_end_invoke is used to retrieve the invocation result.
// If IREE_STATUS_DEFERRED is returned one of iree_vm_resume_invoke or
// iree_vm_wait_invoke (or other wait frame handling) must be performed by the
// caller.
IREE_API_EXPORT iree_status_t
iree_vm_resume_invoke(iree_vm_invoke_state_t* state);

// Synchronously performs the operation specified by |wait_frame|.
// |deadline_ns| will be combined with the deadline specified in the wait frame
// to bound the wait operation. If successful the caller must use
// iree_vm_resume_invoke to allow the invocation to process the wait results.
//
// Hosting schedulers that can more efficiently perform the wait should do so,
// either synchronously or asynchronously. Wait frames are stored on the stack
// and will remain valid until iree_vm_resume_invoke is used to complete the
// wait.
//
// Returns OK if the wait operation was performed regardless of the wait result.
// Wait errors are stored on the |wait_frame| for processing after the
// invocation is resumed.
IREE_API_EXPORT iree_status_t
iree_vm_wait_invoke(iree_vm_invoke_state_t* state,
                    iree_vm_wait_frame_t* wait_frame, iree_time_t deadline_ns);

// Ends the invocation sequence and appends the returned values to |outputs|.
// Invocation resources in |state| will be released and upon return the storage
// can be reused for another invocation. Note that the context the invocation is
// operating in may be released and all pointers to stack resources will be
// invalid if they had not been previously retained by the caller.
//
// Returns OK if the invocation has ended and |state| has been cleaned up.
// The status of the invocation will be stored in |out_status| and ownership of
// the status handle is transferred to the caller.
IREE_API_EXPORT iree_status_t iree_vm_end_invoke(iree_vm_invoke_state_t* state,
                                                 iree_vm_list_t* outputs,
                                                 iree_status_t* out_status);

// Aborts an invocation sequence prior to it having successfully ended.
// Invocation resources in |state| will be released and upon return the storage
// can be reused for another invocation. Note that the context the invocation is
// operating in may be released and all pointers to stack resources will be
// invalid if they had not been previously retained by the caller.
//
// This is not to be used for signaling; user-level cancellation of invocations
// must happen via user-level mechanisms (such as HAL semaphores). The function
// invocation will not be notified of the abort and instead will just wink out
// of existence and comes with whatever implications that has for the program.
//
// Only use this if the invocation sequence failed and calling
// iree_vm_end_invoke would be invalid. If all iree_vm_*_invoke calls have
// succeeded then iree_vm_end_invoke must be used instead.
IREE_API_EXPORT void iree_vm_abort_invoke(iree_vm_invoke_state_t* state);

//===----------------------------------------------------------------------===//
// Asynchronous stateful invocation
//===----------------------------------------------------------------------===//

// TODO(benvanik): document and implement.
IREE_API_EXPORT iree_status_t iree_vm_invocation_create(
    iree_vm_context_t* context, iree_vm_function_t function,
    iree_vm_invocation_flags_t flags, const iree_vm_invocation_policy_t* policy,
    const iree_vm_list_t* inputs, iree_allocator_t allocator,
    iree_vm_invocation_t** out_invocation);

// Retains the given |invocation| for the caller.
IREE_API_EXPORT iree_status_t
iree_vm_invocation_retain(iree_vm_invocation_t* invocation);

// Releases the given |invocation| from the caller.
IREE_API_EXPORT iree_status_t
iree_vm_invocation_release(iree_vm_invocation_t* invocation);

// Queries the completion status of the invocation.
// Returns one of the following:
//   IREE_STATUS_OK: the invocation completed successfully.
//   IREE_STATUS_DEFERRED: the invocation has not yet completed.
//   IREE_STATUS_CANCELLED: the invocation was cancelled by the user.
//   IREE_STATUS_ABORTED: the invocation was aborted by the executor.
//   IREE_STATUS_*: an error occurred during invocation.
IREE_API_EXPORT iree_status_t
iree_vm_invocation_query_status(iree_vm_invocation_t* invocation);

// Returns a reference to the outputs of the invocation.
// The returned structure is valid for the lifetime of the invocation and
// callers must retain any refs they want to outlive the invocation once
// released.
//
// Returns NULL if the invocation has not yet completed or if it failed.
IREE_API_EXPORT const iree_vm_list_t* iree_vm_invocation_outputs(
    iree_vm_invocation_t* invocation);

// Blocks the caller until the invocation completes (successfully or otherwise).
//
// Returns IREE_STATUS_DEADLINE_EXCEEDED if |deadline| elapses before the
// invocation completes and otherwise returns iree_vm_invocation_query_status.
IREE_API_EXPORT iree_status_t iree_vm_invocation_await(
    iree_vm_invocation_t* invocation, iree_time_t deadline);

// Attempts to cancel the invocation if it is in-flight.
// Cancellation is not guaranteed to work and should be considered a hint.
// A no-op if the invocation has already completed.
IREE_API_EXPORT void iree_vm_invocation_cancel(
    iree_vm_invocation_t* invocation);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_INVOCATION_H_
