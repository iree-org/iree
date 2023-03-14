// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_CONTEXT_H_
#define IREE_VM_CONTEXT_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/vm/instance.h"
#include "iree/vm/module.h"
#include "iree/vm/ref.h"
#include "iree/vm/stack.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// An isolated execution context.
// Effectively a sandbox where modules can be loaded and run with restricted
// visibility and where they can maintain state.
//
// Modules have imports resolved automatically when registered by searching
// existing modules registered within the context and load order is used for
// resolution. Functions are resolved from the most recently registered module
// back to the first, such that modules can override implementations of
// functions in previously registered modules.
//
// Thread-compatible and must be externally synchronized.
typedef struct iree_vm_context_t iree_vm_context_t;

enum iree_vm_context_flag_bits_t {
  IREE_VM_CONTEXT_FLAG_NONE = 0u,

  // Enables tracing of execution to stderr (when available).
  // See iree/base/config.h for the flags that control whether this
  // functionality is available; specifically:
  //   -DIREE_VM_EXECUTION_TRACING_ENABLE=1
  // All invocations made to this context - including initializers - will be
  // traced. For fine-grained control use `iree_vm_invocation_flags_t`.
  IREE_VM_CONTEXT_FLAG_TRACE_EXECUTION = 1u << 0,

  // Context allows concurrent execution.
  // Multiple OS threads may call into the context concurrently. Synchronization
  // is not performed by the context and callers must ensure the executing
  // programs support concurrency.
  IREE_VM_CONTEXT_FLAG_CONCURRENT = 1u << 1,
};
typedef uint32_t iree_vm_context_flags_t;

typedef intptr_t iree_vm_context_id_t;

// Creates a new context that uses the given |instance| for device management.
// |out_context| must be released by the caller.
IREE_API_EXPORT iree_status_t iree_vm_context_create(
    iree_vm_instance_t* instance, iree_vm_context_flags_t flags,
    iree_allocator_t allocator, iree_vm_context_t** out_context);

// Creates a new context with the given static set of modules.
// This is equivalent to iree_vm_context_create+iree_vm_context_register_modules
// but may be more efficient to allocate. Contexts created in this way cannot
// have additional modules registered after creation.
// |out_context| must be released by the caller.
IREE_API_EXPORT iree_status_t iree_vm_context_create_with_modules(
    iree_vm_instance_t* instance, iree_vm_context_flags_t flags,
    iree_host_size_t module_count, iree_vm_module_t** modules,
    iree_allocator_t allocator, iree_vm_context_t** out_context);

// Retains the given |context| for the caller.
IREE_API_EXPORT void iree_vm_context_retain(iree_vm_context_t* context);

// Releases the given |context| from the caller.
IREE_API_EXPORT void iree_vm_context_release(iree_vm_context_t* context);

// Returns a process-unique ID for the |context|.
IREE_API_EXPORT iree_vm_context_id_t
iree_vm_context_id(const iree_vm_context_t* context);

// Returns |context| flags.
IREE_API_EXPORT iree_vm_context_flags_t
iree_vm_context_flags(const iree_vm_context_t* context);

// Returns the total number of modules registered in |context|.
IREE_API_EXPORT iree_host_size_t
iree_vm_context_module_count(const iree_vm_context_t* context);

// Returns the module registered at index |i| in |context|.
IREE_API_EXPORT iree_vm_module_t* iree_vm_context_module_at(
    const iree_vm_context_t* context, iree_host_size_t i);

// Registers a list of modules with the context and resolves imports in the
// order provided.
// The modules will be retained by the context until destruction.
IREE_API_EXPORT iree_status_t iree_vm_context_register_modules(
    iree_vm_context_t* context, iree_host_size_t module_count,
    iree_vm_module_t** modules);

// Freezes a context such that no more modules can be registered.
// This can be used to ensure that context contents cannot be modified by other
// code as the context is made available to other parts of the program.
// No-op if already frozen.
IREE_API_EXPORT iree_status_t
iree_vm_context_freeze(iree_vm_context_t* context);

// Returns a state resolver setup to use the |context| for resolving module
// state.
IREE_API_EXPORT iree_vm_state_resolver_t
iree_vm_context_state_resolver(const iree_vm_context_t* context);

// Sets |out_module_state| to the context-specific state for the given |module|.
// The state is owned by the context and will only be live for as long as the
// context is.
IREE_API_EXPORT iree_status_t iree_vm_context_resolve_module_state(
    const iree_vm_context_t* context, iree_vm_module_t* module,
    iree_vm_module_state_t** out_module_state);

// Sets |out_function| to an exported function with the fully-qualified name
// of |full_name| or returns IREE_STATUS_NOT_FOUND. The function reference is
// valid for the lifetime of |context|.
IREE_API_EXPORT iree_status_t iree_vm_context_resolve_function(
    const iree_vm_context_t* context, iree_string_view_t full_name,
    iree_vm_function_t* out_function);

// Notifies all modules in the context of a system signal.
IREE_API_EXPORT iree_status_t iree_vm_context_notify(iree_vm_context_t* context,
                                                     iree_vm_signal_t signal);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

IREE_VM_DECLARE_CC_TYPE_ADAPTERS(iree_vm_context, iree_vm_context_t);

#endif  // IREE_VM_CONTEXT_H_
