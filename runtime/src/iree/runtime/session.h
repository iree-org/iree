// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_RUNTIME_SESSION_H_
#define IREE_RUNTIME_SESSION_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_runtime_instance_t iree_runtime_instance_t;

// A session containing a set of loaded VM modules and their runtime state.
// Each session has its own isolated module state and though multiple sessions
// may share the same device they will all see their own individual timelines.
// Think of a session like a process in an operating system: able to communicate
// and share syscalls but with a strict separation.
//
// Only sessions that share an instance may directly share resources as
// different instances may have different HAL devices and have incompatible
// memory. Import and export APIs must be used to transfer the resources across
// instances or incompatible devices within the same instance.
//
// As with all of iree/runtime/ this API is a higher-level wrapper for the
// low-level IREE HAL and VM. Using this may pull in additional dependencies and
// perform additional allocations compared to what you can get by directly going
// to the lower levels.
//
// Thread-compatible; only a single thread may use the session at any time and
// the caller must use external synchronization if they will be using it or any
// resource derived from it concurrently. Any two sessions may be executed
// concurrently without interference.
typedef struct iree_runtime_session_t iree_runtime_session_t;

//===----------------------------------------------------------------------===//
// iree_runtime_session_options_t
//===----------------------------------------------------------------------===//

// Builtin modules that are provided by the runtime.
enum iree_runtime_session_builtins_bits_t {
  // All built-in modules that are compiled into the runtime will be available.
  IREE_RUNTIME_SESSION_BUILTIN_ALL = UINT64_MAX,
};
typedef uint64_t iree_runtime_session_builtins_t;

// Options used to configure session creation.
typedef struct iree_runtime_session_options_t {
  // Flags controlling the execution environment.
  iree_vm_context_flags_t context_flags;

  // A bitmask identifying which IREE builtin modules should be enabled.
  // Session creation will fail if a requested module is not built into the
  // runtime binary.
  iree_runtime_session_builtins_t builtin_modules;
} iree_runtime_session_options_t;

// Initializes |out_options| to its default values.
IREE_API_EXPORT void iree_runtime_session_options_initialize(
    iree_runtime_session_options_t* out_options);

//===----------------------------------------------------------------------===//
// iree_runtime_session_t
//===----------------------------------------------------------------------===//

// Creates a new session forced to use the given |device|.
// This bypasses any device enumeration performed by the loaded modules but
// the loaded modules will still verify that the device matches their
// requirements.
//
// A base set of modules may be added by the runtime during creation based on
// |options| and users may load additional modules - such as the one containing
// their user code - by using the iree_vm_context_t provided by
// iree_runtime_session_context.
//
// |host_allocator| will be used to allocate the session and any associated
// resources. |out_session| must be released by the caller.
IREE_API_EXPORT iree_status_t iree_runtime_session_create_with_device(
    iree_runtime_instance_t* instance,
    const iree_runtime_session_options_t* options, iree_hal_device_t* device,
    iree_allocator_t host_allocator, iree_runtime_session_t** out_session);

// Retains the given |session| for the caller.
IREE_API_EXPORT void iree_runtime_session_retain(
    iree_runtime_session_t* session);

// Releases the given |session| from the caller.
IREE_API_EXPORT void iree_runtime_session_release(
    iree_runtime_session_t* session);

// Returns the host allocator used to allocate the session and its resources.
// Callers should use this to allocate resources so that any memory tracking
// being performed correctly attributes the allocations to the session.
IREE_API_EXPORT iree_allocator_t
iree_runtime_session_host_allocator(const iree_runtime_session_t* session);

// Returns the instance the session uses for shared resources.
IREE_API_EXPORT iree_runtime_instance_t* iree_runtime_session_instance(
    const iree_runtime_session_t* session);

// Returns the VM context used to load and link modules.
// The context can be used to perform additional reflection over the loaded
// modules or load additional modules (if supported).
IREE_API_EXPORT iree_vm_context_t* iree_runtime_session_context(
    const iree_runtime_session_t* session);

// Returns the HAL device being used for execution.
//
// NOTE: this device will not be available until initialized by a user module
// and will return NULL if queried prior.
IREE_API_EXPORT iree_hal_device_t* iree_runtime_session_device(
    const iree_runtime_session_t* session);

// Returns the device allocator used to allocate compatible buffers.
// Buffers from other allocators may not be compatible and require importing
// prior to being usable by the session.
//
// NOTE: this device allocator will not be available until initialized by a
// user module and will return NULL if queried prior.
IREE_API_EXPORT iree_hal_allocator_t* iree_runtime_session_device_allocator(
    const iree_runtime_session_t* session);

// Trims transient/cached resources used by the session.
// Upon resuming these resources may be expensive to rematerialize/reload and
// as such this should only be called when it is known the resources will not
// be needed soon.
IREE_API_EXPORT iree_status_t
iree_runtime_session_trim(iree_runtime_session_t* session);

// Appends the given |module| to the context.
// The module will be retained by the context.
//
// NOTE: only valid if the context is not yet frozen; see
// iree_vm_context_freeze for more information.
IREE_API_EXPORT iree_status_t iree_runtime_session_append_module(
    iree_runtime_session_t* session, iree_vm_module_t* module);

// Appends a bytecode module to the context loaded from the given memory blob.
// If the module exists as a file prefer instead to use
// iree_runtime_session_append_bytecode_module_from_file to use memory mapped
// I/O and reduce total memory consumption.
//
// The data must remain valid for the lifetime of the session. If a
// |flatbuffer_allocator| is provided then it will be used to free the
// |flatbuffer_data| when the module is destroyed. This call always consumes the
// data and even if the module fails to load or be registered with the session
// the |flatbuffer_allocator| will be used to release it.
//
// NOTE: only valid if the context is not yet frozen; see
// iree_vm_context_freeze for more information.
IREE_API_EXPORT iree_status_t
iree_runtime_session_append_bytecode_module_from_memory(
    iree_runtime_session_t* session, iree_const_byte_span_t flatbuffer_data,
    iree_allocator_t flatbuffer_allocator);

// Appends a bytecode module to the context loaded from the given |file_path|.
//
// NOTE: only valid if the context is not yet frozen; see
// iree_vm_context_freeze for more information.
IREE_API_EXPORT iree_status_t
iree_runtime_session_append_bytecode_module_from_file(
    iree_runtime_session_t* session, const char* file_path);

// Appends a bytecode module to the context loaded from stdin.
//
// NOTE: only valid if the context is not yet frozen; see
// iree_vm_context_freeze for more information.
IREE_API_EXPORT iree_status_t
iree_runtime_session_append_bytecode_module_from_stdin(
    iree_runtime_session_t* session);

// Sets |out_function| to an exported function with the fully-qualified name
// of |full_name| or returns IREE_STATUS_NOT_FOUND. The function reference is
// valid for the lifetime of |session|.
//
// The function name matches the original MLIR module and function symbols.
// Example:
//   module @foo {
//     func.func @bar()
//   }
// The full name of '@bar' is 'foo.bar'.
// By default modules have the name 'module'.
IREE_API_EXPORT iree_status_t iree_runtime_session_lookup_function(
    const iree_runtime_session_t* session, iree_string_view_t full_name,
    iree_vm_function_t* out_function);

// Synchronously issues a generic function call.
//
// |input_list| is used to pass values and objects into the target function and
// must match the signature defined by the compiled function. List ownership
// remains with the caller.
//
// |output_list| is populated after the function completes execution with the
// output values and objects of the function. List ownership remains with the
// caller.
//
// Functions with either no inputs or outputs may provide NULL for the
// respective list.
IREE_API_EXPORT iree_status_t iree_runtime_session_call(
    iree_runtime_session_t* session, const iree_vm_function_t* function,
    iree_vm_list_t* input_list, iree_vm_list_t* output_list);

// Synchronously issues a generic function call by fully-qualified name.
// This is equivalent to performing a iree_runtime_session_lookup_function
// followed by a iree_runtime_session_call. When calling the same function
// repeatedly callers should perform the lookup and cache the resulting function
// handle to avoid repeated lookups.
IREE_API_EXPORT iree_status_t iree_runtime_session_call_by_name(
    iree_runtime_session_t* session, iree_string_view_t full_name,
    iree_vm_list_t* input_list, iree_vm_list_t* output_list);

// Synchronously issues a direct function call.
// This bypasses signature verification and directly calls through the VM ABI.
// Though still safe(ish) the errors reported on a signature mismatch will be
// much less useful than a call performed via the more generic methods. Treat
// this as a low-level technique only to be used when the calling host code and
// callee modules are known to be compatible.
//
// See iree_vm_function_call_t for more information.
IREE_API_EXPORT iree_status_t iree_runtime_session_call_direct(
    iree_runtime_session_t* session, const iree_vm_function_call_t call);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_RUNTIME_SESSION_H_
