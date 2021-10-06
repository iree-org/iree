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

// Synchronously invokes a function in the VM.
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
    iree_vm_list_t* inputs, iree_vm_list_t* outputs,
    iree_allocator_t allocator);

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
//   IREE_STATUS_UNAVAILABLE: the invocation has not yet completed.
//   IREE_STATUS_CANCELLED: the invocation was cancelled internally.
//   IREE_STATUS_ABORTED: the invocation was aborted.
//   IREE_STATUS_*: an error occurred during invocation.
IREE_API_EXPORT iree_status_t
iree_vm_invocation_query_status(iree_vm_invocation_t* invocation);

// Returns a reference to the output of the invocation.
// The returned structure is valid for the lifetime of the invocation and
// callers must retain any refs they want to outlive the invocation once
// released.
//
// Returns NULL if the invocation did not complete successfully.
IREE_API_EXPORT const iree_vm_list_t* iree_vm_invocation_output(
    iree_vm_invocation_t* invocation);

// Blocks the caller until the invocation completes (successfully or otherwise).
//
// Returns IREE_STATUS_DEADLINE_EXCEEDED if |deadline| elapses before the
// invocation completes and otherwise returns iree_vm_invocation_query_status.
IREE_API_EXPORT iree_status_t iree_vm_invocation_await(
    iree_vm_invocation_t* invocation, iree_time_t deadline);

// Attempts to abort the invocation if it is in-flight.
// A no-op if the invocation has already completed.
IREE_API_EXPORT iree_status_t
iree_vm_invocation_abort(iree_vm_invocation_t* invocation);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_INVOCATION_H_
