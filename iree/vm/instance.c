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

#include "iree/vm/instance.h"

#include "iree/base/internal/atomics.h"
#include "iree/base/tracing.h"
#include "iree/vm/builtin_types.h"

struct iree_vm_instance {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t allocator;
};

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_instance_create(
    iree_allocator_t allocator, iree_vm_instance_t** out_instance) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_instance);
  *out_instance = NULL;

  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, iree_vm_register_builtin_types());

  iree_vm_instance_t* instance = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(iree_vm_instance_t),
                                (void**)&instance));
  instance->allocator = allocator;
  iree_atomic_ref_count_init(&instance->ref_count);

  *out_instance = instance;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_vm_instance_destroy(iree_vm_instance_t* instance) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(instance);
  iree_allocator_free(instance->allocator, instance);
  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT void IREE_API_CALL
iree_vm_instance_retain(iree_vm_instance_t* instance) {
  if (instance) {
    iree_atomic_ref_count_inc(&instance->ref_count);
  }
}

IREE_API_EXPORT void IREE_API_CALL
iree_vm_instance_release(iree_vm_instance_t* instance) {
  if (instance && iree_atomic_ref_count_dec(&instance->ref_count) == 1) {
    iree_vm_instance_destroy(instance);
  }
}
