// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_INSTANCE_H_
#define IREE_VM_INSTANCE_H_

#include "iree/base/api.h"
#include "iree/vm/ref.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Default maximum registered types value for iree_vm_instance_create.
// Users wanting to conserve memory can reduce this to the minimum their
// instance requires.
#define IREE_VM_TYPE_CAPACITY_DEFAULT 32

// Shared runtime instance responsible for routing iree_vm_context_ events,
// enumerating and creating hardware device interfaces, and managing device
// resource pools.
//
// A single runtime instance can service multiple contexts and hosting
// applications should try to reuse instances as much as possible. This ensures
// that resource allocation across contexts is handled and extraneous device
// interaction is avoided. For devices that may have exclusive access
// restrictions it is mandatory to share instances, so plan accordingly.
//
// Thread-safe.
typedef struct iree_vm_instance_t iree_vm_instance_t;

// Creates a new instance. This should be shared with all contexts in an
// application to ensure that resources are tracked properly and threads are
// managed correctly.
//
// |type_capacity| specifies the maximum number of types that can be
// simultaneously registered with the instance. Callers can use
// IREE_VM_TYPE_CAPACITY_DEFAULT if they have a reasonable number of types.
//
// |out_instance| must be released by the caller.
IREE_API_EXPORT iree_status_t iree_vm_instance_create(
    iree_host_size_t type_capacity, iree_allocator_t allocator,
    iree_vm_instance_t** out_instance);

// Retains the given |instance| for the caller.
IREE_API_EXPORT void iree_vm_instance_retain(iree_vm_instance_t* instance);

// Releases the given |instance| from the caller.
IREE_API_EXPORT void iree_vm_instance_release(iree_vm_instance_t* instance);

// Returns the host allocator the instance was created with.
IREE_API_EXPORT iree_allocator_t
iree_vm_instance_allocator(iree_vm_instance_t* instance);

// Registers a user-defined type with the IREE C ref system.
// The provided destroy function will be used to destroy objects when their
// reference count goes to 0. NULL can be used to no-op the destruction if the
// type is not owned by the VM.
//
// Descriptors registered multiple times will be deduplicated and counted to
// ensure a matching number of unregisters are required to fully unregister
// the type. Descriptors do not need to be unregistered before instance
// destruction and are only required if the memory defining the descriptor is
// invalid (shared library unload, etc).
//
// Once registered the descriptor must stay valid until all ref types created
// using it have expired and the type is unregistered from the instance.
//
// Upon successful registration |out_registration| will be set to the canonical
// iree_vm_ref_type_t that should be used when interacting with the type.
//
// NOTE: the name is not retained and must be kept live by the caller. Ideally
// it is stored in static read-only memory in the binary.
IREE_API_EXPORT iree_status_t
iree_vm_instance_register_type(iree_vm_instance_t* instance,
                               const iree_vm_ref_type_descriptor_t* descriptor,
                               iree_vm_ref_type_t* out_registration);

// Unregisters a user-defined type with the IREE C ref system.
// No iree_vm_ref_t instances must be live in the program referencing the type.
IREE_API_EXPORT void iree_vm_instance_unregister_type(
    iree_vm_instance_t* instance,
    const iree_vm_ref_type_descriptor_t* descriptor);

// Returns the registered type descriptor for the given type, if found.
IREE_API_EXPORT iree_vm_ref_type_t iree_vm_instance_lookup_type(
    iree_vm_instance_t* instance, iree_string_view_t full_name);

// Resolves all builtin VM types by looking them up on the instance.
// This should only be called in dynamically-loaded libraries.
IREE_API_EXPORT iree_status_t
iree_vm_resolve_builtin_types(iree_vm_instance_t* instance);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

IREE_VM_DECLARE_CC_TYPE_ADAPTERS(iree_vm_instance, iree_vm_instance_t);

#endif  // IREE_VM_INSTANCE_H_
