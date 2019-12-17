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

#include "iree/vm2/instance.h"

#include <stdatomic.h>

struct iree_vm_instance {
  atomic_intptr_t ref_count;
  iree_allocator_t allocator;
};

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_instance_create(
    iree_allocator_t allocator, iree_vm_instance_t** out_instance) {
  if (!out_instance) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_instance = NULL;

  iree_vm_instance_t* instance = NULL;
  IREE_API_RETURN_IF_API_ERROR(iree_allocator_malloc(
      allocator, sizeof(iree_vm_instance_t), (void**)&instance));
  instance->allocator = allocator;
  atomic_store(&instance->ref_count, 1);

  *out_instance = instance;
  return IREE_STATUS_OK;
}

static iree_status_t iree_vm_instance_destroy(iree_vm_instance_t* instance) {
  iree_allocator_free(instance->allocator, instance);
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_instance_retain(iree_vm_instance_t* instance) {
  if (!instance) return IREE_STATUS_INVALID_ARGUMENT;
  atomic_fetch_add(&instance->ref_count, 1);
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_instance_release(iree_vm_instance_t* instance) {
  if (instance) {
    if (atomic_fetch_sub(&instance->ref_count, 1) == 1) {
      return iree_vm_instance_destroy(instance);
    }
  }
  return IREE_STATUS_OK;
}
