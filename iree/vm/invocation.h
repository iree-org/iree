// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

typedef struct iree_vm_invocation iree_vm_invocation_t;
typedef struct iree_vm_invocation_policy iree_vm_invocation_policy_t;

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
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_invoke(
    iree_vm_context_t* context, iree_vm_function_t function,
    const iree_vm_invocation_policy_t* policy, iree_vm_list_t* inputs,
    iree_vm_list_t* outputs, iree_allocator_t allocator);

// TODO(benvanik): document and implement.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_invocation_create(
    iree_vm_context_t* context, iree_vm_function_t function,
    const iree_vm_invocation_policy_t* policy, const iree_vm_list_t* inputs,
    iree_allocator_t allocator, iree_vm_invocation_t** out_invocation);

// Retains the given |invocation| for the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_invocation_retain(iree_vm_invocation_t* invocation);

// Releases the given |invocation| from the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_invocation_release(iree_vm_invocation_t* invocation);

// Queries the completion status of the invocation.
// Returns one of the following:
//   IREE_STATUS_OK: the invocation completed successfully.
//   IREE_STATUS_UNAVAILABLE: the invocation has not yet completed.
//   IREE_STATUS_CANCELLED: the invocation was cancelled internally.
//   IREE_STATUS_ABORTED: the invocation was aborted.
//   IREE_STATUS_*: an error occurred during invocation.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_invocation_query_status(iree_vm_invocation_t* invocation);

// Returns a reference to the output of the invocation.
// The returned structure is valid for the lifetime of the invocation and
// callers must retain any refs they want to outlive the invocation once
// released.
//
// Returns NULL if the invocation did not complete successfully.
IREE_API_EXPORT const iree_vm_list_t* IREE_API_CALL
iree_vm_invocation_output(iree_vm_invocation_t* invocation);

// Blocks the caller until the invocation completes (successfully or otherwise).
//
// Returns IREE_STATUS_DEADLINE_EXCEEDED if |deadline| elapses before the
// invocation completes and otherwise returns iree_vm_invocation_query_status.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_invocation_await(
    iree_vm_invocation_t* invocation, iree_time_t deadline);

// Attempts to abort the invocation if it is in-flight.
// A no-op if the invocation has already completed.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_invocation_abort(iree_vm_invocation_t* invocation);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_INVOCATION_H_
