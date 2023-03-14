// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_RUNTIME_CALL_H_
#define IREE_RUNTIME_CALL_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_runtime_session_t iree_runtime_session_t;

//===----------------------------------------------------------------------===//
// iree_runtime_call_t
//===----------------------------------------------------------------------===//

// TODO(benvanik): determine if we want to control behavior like non-blocking
// or whether to consume inputs like this or by having separate call types.
// For example, an async_call may make things more clear when using semaphores
// without having to pollute this interface.
enum iree_runtime_call_flag_bits_t {
  IREE_RUNTIME_CALL_FLAG_RESERVED = 0u,
};
typedef uint32_t iree_runtime_call_flags_t;

// A stateful VM function call builder.
//
// Applications that will be calling the same function repeatedly can reuse the
// call to avoid having to construct the inputs lists each time. Outputs of
// prior calls will be retained unless iree_runtime_call_reset is used and will
// be provided to the VM on subsequent calls to reuse (if able): when reusing a
// call like this callers are required to either reset the call, copy their
// data out, or reset the particular output they are consuming.
//
// Thread-compatible; these are designed to be stack-local or embedded in a user
// data structure that can provide synchronization when required.
typedef struct iree_runtime_call_t {
  iree_runtime_session_t* session;
  iree_vm_function_t function;
  iree_vm_list_t* inputs;
  iree_vm_list_t* outputs;
} iree_runtime_call_t;

// Initializes call state for a call to |function| within |session|.
IREE_API_EXPORT iree_status_t iree_runtime_call_initialize(
    iree_runtime_session_t* session, iree_vm_function_t function,
    iree_runtime_call_t* out_call);

// Initializes call state for a call to |full_name| within |session|.
//
// The function name matches the original MLIR module and function symbols.
// Example:
//   module @foo {
//     func.func @bar()
//   }
// The full name of '@bar' is 'foo.bar'.
// By default modules have the name 'module'.
IREE_API_EXPORT iree_status_t iree_runtime_call_initialize_by_name(
    iree_runtime_session_t* session, iree_string_view_t full_name,
    iree_runtime_call_t* out_call);

// Deinitializes a call by releasing its input and output lists.
IREE_API_EXPORT void iree_runtime_call_deinitialize(iree_runtime_call_t* call);

// Resets the input and output lists back to 0-length in preparation for
// construction of another call.
IREE_API_EXPORT void iree_runtime_call_reset(iree_runtime_call_t* call);

// Returns an initially-empty variant list for passing in function inputs.
// The list must be fully populated based on the required arguments of the
// function.
IREE_API_EXPORT iree_vm_list_t* iree_runtime_call_inputs(
    const iree_runtime_call_t* call);

// Returns an initially-empty variant list for passing in function outputs or
// for reading back the results of a call.
IREE_API_EXPORT iree_vm_list_t* iree_runtime_call_outputs(
    const iree_runtime_call_t* call);

// Synchronously invokes the call and returns the status.
// The inputs list will remain unchanged to allow for subsequent reuse and the
// output list will be populated with the results of the call.
IREE_API_EXPORT iree_status_t iree_runtime_call_invoke(
    iree_runtime_call_t* call, iree_runtime_call_flags_t flags);

//===----------------------------------------------------------------------===//
// Helpers for defining call I/O
//===----------------------------------------------------------------------===//
// NOTE: these are mostly useful for one-shot tests and samples. Applications
// that will be reusing the same inputs and outputs should prefer to track them
// themselves. If applications are able it's strongly recommended that they
// produce and consume the iree_hal_buffer_ts directly to avoid additional
// copies and allocations.

// Pushes |buffer_view| to the call inputs list.
// The value will be retained by the list.
IREE_API_EXPORT iree_status_t iree_runtime_call_inputs_push_back_buffer_view(
    iree_runtime_call_t* call, iree_hal_buffer_view_t* buffer_view);

// Pops a buffer view from the front of the call outputs list.
// Ownership of the buffer view transfers to the caller.
IREE_API_EXPORT iree_status_t iree_runtime_call_outputs_pop_front_buffer_view(
    iree_runtime_call_t* call, iree_hal_buffer_view_t** out_buffer_view);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_RUNTIME_CALL_H_
