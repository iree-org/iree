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

#ifndef IREE_VM_INSTANCE_H_
#define IREE_VM_INSTANCE_H_

#include "iree/base/api.h"

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
typedef struct iree_vm_instance iree_vm_instance_t;

// Creates a new instance. This should be shared with all contexts in an
// application to ensure that resources are tracked properly and threads are
// managed correctly.
// |out_instance| must be released by the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_instance_create(
    iree_allocator_t allocator, iree_vm_instance_t** out_instance);

// Retains the given |instance| for the caller.
IREE_API_EXPORT void IREE_API_CALL
iree_vm_instance_retain(iree_vm_instance_t* instance);

// Releases the given |instance| from the caller.
IREE_API_EXPORT void IREE_API_CALL
iree_vm_instance_release(iree_vm_instance_t* instance);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_INSTANCE_H_
