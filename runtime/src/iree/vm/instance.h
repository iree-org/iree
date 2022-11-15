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
// |out_instance| must be released by the caller.
IREE_API_EXPORT iree_status_t iree_vm_instance_create(
    iree_allocator_t allocator, iree_vm_instance_t** out_instance);

// Retains the given |instance| for the caller.
IREE_API_EXPORT void iree_vm_instance_retain(iree_vm_instance_t* instance);

// Releases the given |instance| from the caller.
IREE_API_EXPORT void iree_vm_instance_release(iree_vm_instance_t* instance);

// Returns the host allocator the instance was created with.
IREE_API_EXPORT iree_allocator_t
iree_vm_instance_allocator(iree_vm_instance_t* instance);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

IREE_VM_DECLARE_CC_TYPE_ADAPTERS(iree_vm_instance, iree_vm_instance_t);

#endif  // IREE_VM_INSTANCE_H_
