// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/instance.h"

#include <stddef.h>

#include "iree/base/internal/atomics.h"
#include "iree/base/tracing.h"

// Defined in their respective files:
iree_status_t iree_vm_buffer_register_types(iree_vm_instance_t* instance);
iree_status_t iree_vm_list_register_types(iree_vm_instance_t* instance);

// Registers the builtin VM types. This must be called on startup. Safe to call
// multiple times.
static iree_status_t iree_vm_register_builtin_types(
    iree_vm_instance_t* instance) {
  IREE_RETURN_IF_ERROR(iree_vm_buffer_register_types(instance));
  IREE_RETURN_IF_ERROR(iree_vm_list_register_types(instance));
  return iree_ok_status();
}

struct iree_vm_instance_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t allocator;
};

IREE_API_EXPORT iree_status_t iree_vm_instance_create(
    iree_allocator_t allocator, iree_vm_instance_t** out_instance) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_instance);
  *out_instance = NULL;

  iree_vm_instance_t* instance = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(allocator, sizeof(*instance), (void**)&instance));
  instance->allocator = allocator;
  iree_atomic_ref_count_init(&instance->ref_count);

  iree_status_t status = iree_vm_register_builtin_types(instance);

  if (iree_status_is_ok(status)) {
    *out_instance = instance;
  } else {
    iree_vm_instance_release(instance);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_vm_instance_destroy(iree_vm_instance_t* instance) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(instance);
  iree_allocator_free(instance->allocator, instance);
  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT void iree_vm_instance_retain(iree_vm_instance_t* instance) {
  if (instance) {
    iree_atomic_ref_count_inc(&instance->ref_count);
  }
}

IREE_API_EXPORT void iree_vm_instance_release(iree_vm_instance_t* instance) {
  if (instance && iree_atomic_ref_count_dec(&instance->ref_count) == 1) {
    iree_vm_instance_destroy(instance);
  }
}

IREE_API_EXPORT iree_allocator_t
iree_vm_instance_allocator(iree_vm_instance_t* instance) {
  IREE_ASSERT_ARGUMENT(instance);
  return instance->allocator;
}
