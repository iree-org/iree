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

#ifndef IREE_VM_CONTEXT_H_
#define IREE_VM_CONTEXT_H_

#include "iree/base/api.h"
#include "iree/vm/instance.h"
#include "iree/vm/module.h"
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
typedef struct iree_vm_context iree_vm_context_t;

// Creates a new context that uses the given |instance| for device management.
// |out_context| must be released by the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_context_create(iree_vm_instance_t* instance, iree_allocator_t allocator,
                       iree_vm_context_t** out_context);

// Creates a new context with the given static set of modules.
// This is equivalent to iree_vm_context_create+iree_vm_context_register_modules
// but may be more efficient to allocate. Contexts created in this way cannot
// have additional modules registered after creation.
// |out_context| must be released by the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_context_create_with_modules(
    iree_vm_instance_t* instance, iree_vm_module_t** modules,
    iree_host_size_t module_count, iree_allocator_t allocator,
    iree_vm_context_t** out_context);

// Retains the given |context| for the caller.
IREE_API_EXPORT void IREE_API_CALL
iree_vm_context_retain(iree_vm_context_t* context);

// Releases the given |context| from the caller.
IREE_API_EXPORT void IREE_API_CALL
iree_vm_context_release(iree_vm_context_t* context);

// Returns a process-unique ID for the |context|.
IREE_API_EXPORT intptr_t IREE_API_CALL
iree_vm_context_id(const iree_vm_context_t* context);

// Returns a state resolver setup to use the |context| for resolving module
// state.
IREE_API_EXPORT iree_vm_state_resolver_t IREE_API_CALL
iree_vm_context_state_resolver(const iree_vm_context_t* context);

// Sets |out_module_state| to the context-specific state for the given |module|.
// The state is owned by the context and will only be live for as long as the
// context is.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_context_resolve_module_state(const iree_vm_context_t* context,
                                     iree_vm_module_t* module,
                                     iree_vm_module_state_t** out_module_state);

// Registers a list of modules with the context and resolves imports in the
// order provided.
// The modules will be retained by the context until destruction.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_context_register_modules(
    iree_vm_context_t* context, iree_vm_module_t** modules,
    iree_host_size_t module_count);

// Sets |out_function| to to an exported function with the fully-qualified name
// of |full_name| or returns IREE_STATUS_NOT_FOUND. The function reference is
// valid for the lifetime of |context|.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_context_resolve_function(
    const iree_vm_context_t* context, iree_string_view_t full_name,
    iree_vm_function_t* out_function);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_CONTEXT_H_
