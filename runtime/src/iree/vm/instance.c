// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/instance.h"

#include <stddef.h>

#include "iree/base/internal/atomics.h"
#include "iree/base/internal/synchronization.h"
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

typedef struct {
  // Pointer to the externally-defined/unowned descriptor.
  const iree_vm_ref_type_descriptor_t* descriptor;
  // Number of times the type has been registered. Only to be used by the
  // instance and while holding the instance lock.
  uint32_t registration_count;
} iree_vm_registered_type_t;

struct iree_vm_instance_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t allocator;

  iree_slim_mutex_t type_mutex;
  uint16_t type_capacity;
  uint16_t type_count;
  iree_vm_registered_type_t types[];
};

IREE_API_EXPORT iree_status_t iree_vm_instance_create(
    iree_host_size_t type_capacity, iree_allocator_t allocator,
    iree_vm_instance_t** out_instance) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_instance);
  *out_instance = NULL;

  iree_vm_instance_t* instance = NULL;
  iree_host_size_t total_size =
      sizeof(*instance) + type_capacity * sizeof(instance->types[0]);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, total_size, (void**)&instance));
  instance->allocator = allocator;
  iree_atomic_ref_count_init(&instance->ref_count);
  iree_slim_mutex_initialize(&instance->type_mutex);
  instance->type_capacity = type_capacity;

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
  iree_slim_mutex_deinitialize(&instance->type_mutex);
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

IREE_API_EXPORT iree_status_t
iree_vm_instance_register_type(iree_vm_instance_t* instance,
                               const iree_vm_ref_type_descriptor_t* descriptor,
                               iree_vm_ref_type_t* out_registration) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(descriptor);
  IREE_ASSERT_ARGUMENT(out_registration);
  *out_registration = 0;

  if ((((uintptr_t)descriptor) & IREE_VM_REF_TYPE_TAG_BIT_MASK) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "type descriptors must be aligned to %d bytes",
                            (1 << IREE_VM_REF_TYPE_TAG_BITS));
  }

  if (descriptor->offsetof_counter & ~IREE_VM_REF_TYPE_TAG_BIT_MASK) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "types must have offsets within the first few "
                            "words of their structures");
  }

  iree_slim_mutex_lock(&instance->type_mutex);

  // Scan to see if there are any existing types registered with this
  // descriptor.
  for (iree_host_size_t i = 0; i < instance->type_count; ++i) {
    iree_vm_registered_type_t* type = &instance->types[i];
    if (type->descriptor == descriptor) {
      // Already registered - increment count so that we have a balanced
      // register/unregister set.
      ++type->registration_count;
      iree_slim_mutex_unlock(&instance->type_mutex);
      *out_registration = (iree_vm_ref_type_t)descriptor |
                          (iree_vm_ref_type_t)descriptor->offsetof_counter;
      return iree_ok_status();
    }
  }

  // Ensure there's capacity.
  if (instance->type_count + 1 > instance->type_capacity) {
    iree_slim_mutex_unlock(&instance->type_mutex);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "too many user-defined types registered; new type "
                            "%.*s would exceed capacity of %d",
                            (int)descriptor->type_name.size,
                            descriptor->type_name.data,
                            instance->type_capacity);
  }

  // Append to the list.
  instance->types[instance->type_count++] = (iree_vm_registered_type_t){
      .descriptor = descriptor,
      .registration_count = 1,
  };

  iree_slim_mutex_unlock(&instance->type_mutex);

  *out_registration = (iree_vm_ref_type_t)descriptor |
                      (iree_vm_ref_type_t)descriptor->offsetof_counter;
  return iree_ok_status();
}

IREE_API_EXPORT void iree_vm_instance_unregister_type(
    iree_vm_instance_t* instance,
    const iree_vm_ref_type_descriptor_t* descriptor) {
  iree_slim_mutex_lock(&instance->type_mutex);
  for (iree_host_size_t i = 0; i < instance->type_count; ++i) {
    // NOTE: descriptor pointers must be stable so we can just compare that
    // instead of each field.
    iree_vm_registered_type_t* type = &instance->types[i];
    if (type->descriptor == descriptor) {
      if (--type->registration_count == 0) {
        // Last registration reference, remove from the list.
        memmove(&instance->types[i], &instance->types[i + 1],
                instance->type_count - i - 1);
        instance->types[--instance->type_count] = (iree_vm_registered_type_t){
            .descriptor = NULL,
            .registration_count = 0,
        };
      }
      break;
    }
  }
  iree_slim_mutex_unlock(&instance->type_mutex);
}

// NOTE: this does a linear scan over the type descriptors even though they are
// likely in a random order. Type lookup should be done once and reused so this
// shouldn't really matter.
IREE_API_EXPORT iree_vm_ref_type_t iree_vm_instance_lookup_type(
    iree_vm_instance_t* instance, iree_string_view_t full_name) {
  const iree_vm_ref_type_descriptor_t* descriptor = NULL;
  iree_slim_mutex_lock(&instance->type_mutex);
  for (iree_host_size_t i = 0; i < instance->type_count; ++i) {
    const iree_vm_registered_type_t* type = &instance->types[i];
    if (iree_string_view_equal(type->descriptor->type_name, full_name)) {
      descriptor = type->descriptor;
      break;
    }
  }
  iree_slim_mutex_unlock(&instance->type_mutex);
  if (!descriptor) return 0;
  return (iree_vm_ref_type_t)descriptor |
         (iree_vm_ref_type_t)descriptor->offsetof_counter;
}
