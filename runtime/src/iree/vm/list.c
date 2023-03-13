// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/list.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/tracing.h"
#include "iree/vm/instance.h"

static uint8_t iree_vm_value_type_size(iree_vm_value_type_t type) {
  // Size of each iree_vm_value_type_t in bytes. We bitpack these so that we
  // can do a simple shift and mask to get the size.
  const uint32_t kValueTypeSizes = (0u << 0) |   // IREE_VM_VALUE_TYPE_NONE
                                   (1u << 4) |   // IREE_VM_VALUE_TYPE_I8
                                   (2u << 8) |   // IREE_VM_VALUE_TYPE_I16
                                   (4u << 12) |  // IREE_VM_VALUE_TYPE_I32
                                   (8u << 16) |  // IREE_VM_VALUE_TYPE_I64
                                   (4u << 20) |  // IREE_VM_VALUE_TYPE_F32
                                   (8u << 24) |  // IREE_VM_VALUE_TYPE_F64
                                   (0u << 28);   // unused
  return (kValueTypeSizes >> ((type & 0x7) * 4)) & 0xF;
}

// Defines how the iree_vm_list_t storage is allocated and what elements are
// interpreted as.
typedef enum iree_vm_list_storage_mode_e {
  // Each element is a primitive value and stored as a dense array.
  IREE_VM_LIST_STORAGE_MODE_VALUE = 0,
  // Each element is an iree_vm_ref_t of some type.
  IREE_VM_LIST_STORAGE_MODE_REF,
  // Each element is a variant of any type (possibly all different).
  IREE_VM_LIST_STORAGE_MODE_VARIANT,
} iree_vm_list_storage_mode_t;

// A list able to hold either flat primitive elements or ref values.
struct iree_vm_list_t {
  iree_vm_ref_object_t ref_object;
  iree_allocator_t allocator;

  // Current capacity of the list storage, in elements.
  iree_host_size_t capacity;
  // Current count of elements in the list.
  iree_host_size_t count;

  // Element type stored within the list.
  iree_vm_type_def_t element_type;
  // Size of each element in the storage in bytes.
  iree_host_size_t element_size;

  // Storage mode defining how the storage array is managed.
  iree_vm_list_storage_mode_t storage_mode;
  // A flat dense array of elements in the type defined by storage_mode.
  // For certain storage modes, such as IREE_VM_STORAGE_MODE_REF, special
  // lifetime management and cleanup logic is required.
  void* storage;
};

IREE_VM_DEFINE_TYPE_ADAPTERS(iree_vm_list, iree_vm_list_t);

static void iree_vm_list_retain_range(iree_vm_list_t* list,
                                      iree_host_size_t offset,
                                      iree_host_size_t length) {
  switch (list->storage_mode) {
    case IREE_VM_LIST_STORAGE_MODE_VALUE:
      // Value types don't need to be retained.
      break;
    case IREE_VM_LIST_STORAGE_MODE_REF: {
      iree_vm_ref_t* ref_storage = (iree_vm_ref_t*)list->storage;
      for (iree_host_size_t i = offset; i < offset + length; ++i) {
        iree_vm_ref_retain_inplace(&ref_storage[i]);
      }
      break;
    }
    case IREE_VM_LIST_STORAGE_MODE_VARIANT: {
      iree_vm_variant_t* variant_storage = (iree_vm_variant_t*)list->storage;
      for (iree_host_size_t i = offset; i < offset + length; ++i) {
        if (iree_vm_type_def_is_ref(&variant_storage[i].type)) {
          iree_vm_ref_retain_inplace(&variant_storage[i].ref);
        }
      }
      break;
    }
  }
}

static void iree_vm_list_reset_range(iree_vm_list_t* list,
                                     iree_host_size_t offset,
                                     iree_host_size_t length) {
  switch (list->storage_mode) {
    case IREE_VM_LIST_STORAGE_MODE_VALUE: {
      void* base_ptr =
          (void*)((uintptr_t)list->storage + offset * list->element_size);
      memset(base_ptr, 0, length * list->element_size);
      break;
    }
    case IREE_VM_LIST_STORAGE_MODE_REF: {
      iree_vm_ref_t* ref_storage = (iree_vm_ref_t*)list->storage;
      for (iree_host_size_t i = offset; i < offset + length; ++i) {
        iree_vm_ref_release(&ref_storage[i]);
      }
      break;
    }
    case IREE_VM_LIST_STORAGE_MODE_VARIANT: {
      iree_vm_variant_t* variant_storage = (iree_vm_variant_t*)list->storage;
      for (iree_host_size_t i = offset; i < offset + length; ++i) {
        if (iree_vm_type_def_is_ref(&variant_storage[i].type)) {
          iree_vm_ref_release(&variant_storage[i].ref);
          memset(&variant_storage[i].type, 0, sizeof(variant_storage[i].type));
        } else {
          memset(&variant_storage[i], 0, sizeof(variant_storage[i]));
        }
      }
      break;
    }
  }
}

IREE_API_EXPORT iree_host_size_t iree_vm_list_storage_size(
    const iree_vm_type_def_t* element_type, iree_host_size_t capacity) {
  iree_host_size_t element_size = sizeof(iree_vm_variant_t);
  if (element_type) {
    if (iree_vm_type_def_is_value(element_type)) {
      element_size = iree_vm_value_type_size(element_type->value_type);
    } else if (iree_vm_type_def_is_ref(element_type)) {
      element_size = sizeof(iree_vm_ref_t);
    } else {
      element_size = sizeof(iree_vm_variant_t);
    }
  }
  return iree_host_align(sizeof(iree_vm_list_t), 8) +
         iree_host_align(capacity * element_size, 8);
}

IREE_API_EXPORT iree_status_t iree_vm_list_initialize(
    iree_byte_span_t storage, const iree_vm_type_def_t* element_type,
    iree_host_size_t capacity, iree_vm_list_t** out_list) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_vm_list_storage_mode_t storage_mode = IREE_VM_LIST_STORAGE_MODE_VARIANT;
  iree_host_size_t element_size = sizeof(iree_vm_variant_t);
  if (element_type) {
    if (iree_vm_type_def_is_value(element_type)) {
      storage_mode = IREE_VM_LIST_STORAGE_MODE_VALUE;
      element_size = iree_vm_value_type_size(element_type->value_type);
    } else if (iree_vm_type_def_is_ref(element_type)) {
      storage_mode = IREE_VM_LIST_STORAGE_MODE_REF;
      element_size = sizeof(iree_vm_ref_t);
    } else {
      storage_mode = IREE_VM_LIST_STORAGE_MODE_VARIANT;
      element_size = sizeof(iree_vm_variant_t);
    }
  }

  iree_host_size_t storage_offset = iree_host_align(sizeof(iree_vm_list_t), 8);
  iree_host_size_t required_storage_size =
      storage_offset + iree_host_align(capacity * element_size, 8);
  if (storage.data_length < required_storage_size) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "storage buffer underflow: provided=%zu < required=%zu",
        storage.data_length, required_storage_size);
  }
  memset(storage.data, 0, required_storage_size);

  iree_vm_list_t* list = (iree_vm_list_t*)storage.data;
  iree_atomic_ref_count_init(&list->ref_object.counter);
  if (element_type) {
    list->element_type = *element_type;
  }
  list->element_size = element_size;
  list->storage_mode = storage_mode;
  list->capacity = capacity;
  list->storage = storage.data + storage_offset;

  *out_list = list;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT void iree_vm_list_deinitialize(iree_vm_list_t* list) {
  IREE_ASSERT_ARGUMENT(list);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_atomic_ref_count_abort_if_uses(&list->ref_object.counter);
  iree_vm_list_reset_range(list, 0, list->count);
  list->count = 0;

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t iree_vm_list_create(
    const iree_vm_type_def_t* element_type, iree_host_size_t initial_capacity,
    iree_allocator_t allocator, iree_vm_list_t** out_list) {
  IREE_ASSERT_ARGUMENT(out_list);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_vm_list_t* list = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*list), (void**)&list));
  memset(list, 0, sizeof(*list));
  iree_atomic_ref_count_init(&list->ref_object.counter);
  list->allocator = allocator;
  if (element_type) {
    list->element_type = *element_type;
  }

  if (iree_vm_type_def_is_value(&list->element_type) && element_type) {
    list->storage_mode = IREE_VM_LIST_STORAGE_MODE_VALUE;
    list->element_size = iree_vm_value_type_size(element_type->value_type);
  } else if (iree_vm_type_def_is_ref(&list->element_type)) {
    list->storage_mode = IREE_VM_LIST_STORAGE_MODE_REF;
    list->element_size = sizeof(iree_vm_ref_t);
  } else {
    list->storage_mode = IREE_VM_LIST_STORAGE_MODE_VARIANT;
    list->element_size = sizeof(iree_vm_variant_t);
  }

  iree_status_t status = iree_vm_list_reserve(list, initial_capacity);

  if (iree_status_is_ok(status)) {
    *out_list = list;
  } else {
    iree_allocator_free(allocator, list);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_vm_list_destroy(void* ptr) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_vm_list_t* list = (iree_vm_list_t*)ptr;
  iree_vm_list_reset_range(list, 0, list->count);
  iree_allocator_free(list->allocator, list->storage);
  iree_allocator_free(list->allocator, list);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t
iree_vm_list_clone(iree_vm_list_t* source, iree_allocator_t host_allocator,
                   iree_vm_list_t** out_target) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_host_size_t count = iree_vm_list_size(source);
  iree_vm_type_def_t element_type = iree_vm_list_element_type(source);
  iree_vm_list_t* target = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_vm_list_create(&element_type, count, host_allocator, &target));
  iree_status_t status = iree_vm_list_resize(target, count);
  if (iree_status_is_ok(status)) {
    // Copy storage directly. Note that we need to retain any refs contained.
    memcpy(target->storage, source->storage,
           target->count * target->element_size);
    iree_vm_list_retain_range(target, 0, count);
  }
  if (iree_status_is_ok(status)) {
    *out_target = target;
  } else {
    iree_vm_list_release(target);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT void iree_vm_list_retain(iree_vm_list_t* list) {
  iree_vm_ref_object_retain(list, &iree_vm_list_descriptor);
}

IREE_API_EXPORT void iree_vm_list_release(iree_vm_list_t* list) {
  iree_vm_ref_object_release(list, &iree_vm_list_descriptor);
}

IREE_API_EXPORT iree_vm_type_def_t
iree_vm_list_element_type(const iree_vm_list_t* list) {
  return list->element_type;
}

IREE_API_EXPORT iree_host_size_t
iree_vm_list_capacity(const iree_vm_list_t* list) {
  return list->capacity;
}

IREE_API_EXPORT iree_status_t
iree_vm_list_reserve(iree_vm_list_t* list, iree_host_size_t minimum_capacity) {
  IREE_ASSERT_ARGUMENT(list);
  if (list->capacity >= minimum_capacity) {
    return iree_ok_status();
  }
  iree_host_size_t old_capacity = list->capacity;
  iree_host_size_t new_capacity = iree_host_align(minimum_capacity, 64);
  IREE_RETURN_IF_ERROR(iree_allocator_realloc(
      list->allocator, new_capacity * list->element_size, &list->storage));
  memset((void*)((uintptr_t)list->storage + old_capacity * list->element_size),
         0, (new_capacity - old_capacity) * list->element_size);
  list->capacity = new_capacity;
  return iree_ok_status();
}

IREE_API_EXPORT iree_host_size_t iree_vm_list_size(const iree_vm_list_t* list) {
  IREE_ASSERT_ARGUMENT(list);
  return list->count;
}

IREE_API_EXPORT iree_status_t iree_vm_list_resize(iree_vm_list_t* list,
                                                  iree_host_size_t new_size) {
  IREE_ASSERT_ARGUMENT(list);
  if (new_size == list->count) {
    return iree_ok_status();
  } else if (new_size < list->count) {
    // Truncating.
    iree_vm_list_reset_range(list, new_size, list->count - new_size);
    list->count = new_size;
  } else if (new_size > list->capacity) {
    // Extending beyond capacity.
    IREE_RETURN_IF_ERROR(iree_vm_list_reserve(
        list, iree_max(list->capacity * 2, iree_host_align(new_size, 64))));
  }
  list->count = new_size;
  return iree_ok_status();
}

IREE_API_EXPORT void iree_vm_list_clear(iree_vm_list_t* list) {
  if (list->count > 0) {
    // Truncating.
    iree_vm_list_reset_range(list, 0, list->count);
  }
  list->count = 0;
}

static void iree_memswap(void* a, void* b, iree_host_size_t size) {
  uint8_t* a_ptr = (uint8_t*)a;
  uint8_t* b_ptr = (uint8_t*)b;
  for (iree_host_size_t i = 0; i < size; ++i) {
    uint8_t t = a_ptr[i];
    a_ptr[i] = b_ptr[i];
    b_ptr[i] = t;
  }
}

IREE_API_EXPORT void iree_vm_list_swap_storage(iree_vm_list_t* list_a,
                                               iree_vm_list_t* list_b) {
  IREE_ASSERT_ARGUMENT(list_a);
  IREE_ASSERT_ARGUMENT(list_b);
  if (list_a == list_b) return;
  iree_memswap(&list_a->allocator, &list_b->allocator,
               sizeof(list_a->allocator));
  iree_memswap(&list_a->capacity, &list_b->capacity, sizeof(list_a->capacity));
  iree_memswap(&list_a->count, &list_b->count, sizeof(list_a->count));
  iree_memswap(&list_a->element_type, &list_b->element_type,
               sizeof(list_a->element_type));
  iree_memswap(&list_a->element_size, &list_b->element_size,
               sizeof(list_a->element_size));
  iree_memswap(&list_a->storage_mode, &list_b->storage_mode,
               sizeof(list_a->storage_mode));
  iree_memswap(&list_a->storage, &list_b->storage, sizeof(list_a->storage));
}

// Returns true if |src_type| can be converted into |dst_type|.
static bool iree_vm_type_def_is_compatible(iree_vm_type_def_t src_type,
                                           iree_vm_type_def_t dst_type) {
  return memcmp(&src_type, &dst_type, sizeof(dst_type)) == 0;
}

// Copies from a |src_list| of any type (value, ref, variant) into a |dst_list|
// in variant storage mode. This cannot fail as variant lists can store any
// type.
static void iree_vm_list_copy_to_variant_list(iree_vm_list_t* src_list,
                                              iree_host_size_t src_i,
                                              iree_vm_list_t* dst_list,
                                              iree_host_size_t dst_i,
                                              iree_host_size_t count) {
  iree_vm_variant_t* dst_storage =
      (iree_vm_variant_t*)dst_list->storage + dst_i;
  switch (src_list->storage_mode) {
    case IREE_VM_LIST_STORAGE_MODE_VALUE: {
      uintptr_t src_storage =
          (uintptr_t)src_list->storage + src_i * src_list->element_size;
      for (iree_host_size_t i = 0; i < count; ++i) {
        if (iree_vm_type_def_is_ref(&dst_storage[i].type)) {
          iree_vm_ref_release(&dst_storage[i].ref);
        }
        dst_storage[i].type = src_list->element_type;
        memcpy(dst_storage[i].value_storage,
               (uint8_t*)src_storage + i * src_list->element_size,
               src_list->element_size);
      }
      break;
    }
    case IREE_VM_LIST_STORAGE_MODE_REF: {
      iree_vm_ref_t* src_storage = (iree_vm_ref_t*)src_list->storage + src_i;
      for (iree_host_size_t i = 0; i < count; ++i) {
        // NOTE: we retain first in case the lists alias and the ref is the
        // same.
        iree_vm_ref_t* ref = &src_storage[i];
        iree_vm_ref_retain_inplace(ref);
        if (iree_vm_type_def_is_ref(&dst_storage[i].type)) {
          iree_vm_ref_release(&dst_storage[i].ref);
        }
        dst_storage->type = iree_vm_type_def_make_ref_type(ref->type);
        dst_storage->ref = *ref;
      }
      break;
    }
    case IREE_VM_LIST_STORAGE_MODE_VARIANT: {
      iree_vm_variant_t* src_storage =
          (iree_vm_variant_t*)src_list->storage + src_i;
      for (iree_host_size_t i = 0; i < count; ++i) {
        // NOTE: we retain first in case the lists alias and the ref is the
        // same.
        if (iree_vm_type_def_is_ref(&src_storage[i].type)) {
          iree_vm_ref_retain_inplace(&src_storage[i].ref);
        }
        if (iree_vm_type_def_is_ref(&dst_storage[i].type)) {
          iree_vm_ref_release(&dst_storage[i].ref);
        }
        memcpy(&dst_storage[i], &src_storage[i], sizeof(dst_storage[i]));
      }
      break;
    }
  }
}

// Copies from a |src_list| in variant storage mode to a |dst_list| of any type
// (value, ref) while checking each element. This first needs to ensure the
// entire source range matches the expected destination type which makes this
// much slower than the other paths that need not check or only check once per
// copy operation instead.
static iree_status_t iree_vm_list_copy_from_variant_list(
    iree_vm_list_t* src_list, iree_host_size_t src_i, iree_vm_list_t* dst_list,
    iree_host_size_t dst_i, iree_host_size_t count) {
  iree_vm_variant_t* src_storage =
      (iree_vm_variant_t*)src_list->storage + src_i;
  switch (dst_list->storage_mode) {
    case IREE_VM_LIST_STORAGE_MODE_VALUE:
    case IREE_VM_LIST_STORAGE_MODE_REF:
      for (iree_host_size_t i = 0; i < count; ++i) {
        if (!iree_vm_type_def_is_compatible(src_storage[i].type,
                                            dst_list->element_type)) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "destination list element type does not "
                                  "match the source element %" PRIhsz,
                                  src_i + i);
        }
      }
      break;
    default:
      // Destination is a variant list and accepts all inputs.
      break;
  }
  switch (dst_list->storage_mode) {
    case IREE_VM_LIST_STORAGE_MODE_VALUE: {
      uintptr_t dst_storage =
          (uintptr_t)dst_list->storage + dst_i * dst_list->element_size;
      for (iree_host_size_t i = 0; i < count; ++i) {
        memcpy((uint8_t*)dst_storage + i * dst_list->element_size,
               src_storage[i].value_storage, dst_list->element_size);
      }
      break;
    }
    case IREE_VM_LIST_STORAGE_MODE_REF: {
      iree_vm_ref_t* dst_storage = (iree_vm_ref_t*)dst_list->storage + dst_i;
      for (iree_host_size_t i = 0; i < count; ++i) {
        iree_vm_ref_retain(&src_storage[i].ref, &dst_storage[i]);
      }
      break;
    }
    default:
    case IREE_VM_LIST_STORAGE_MODE_VARIANT:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "unhandled copy mode");
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_list_copy(iree_vm_list_t* src_list,
                                                iree_host_size_t src_i,
                                                iree_vm_list_t* dst_list,
                                                iree_host_size_t dst_i,
                                                iree_host_size_t count) {
  IREE_ASSERT_ARGUMENT(src_list);
  IREE_ASSERT_ARGUMENT(dst_list);

  // Fast-path no-op check.
  if (count == 0) return iree_ok_status();

  // Verify ranges.
  const iree_host_size_t src_count = iree_vm_list_size(src_list);
  if (src_i + count > src_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "source range [%" PRIhsz ", %" PRIhsz ") of %" PRIhsz
        " elements out of range of source list with size %" PRIhsz,
        src_i, src_i + count, count, src_count);
  }
  const iree_host_size_t dst_count = iree_vm_list_size(dst_list);
  if (dst_i + count > dst_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "destination range [%" PRIhsz ", %" PRIhsz ") of %" PRIhsz
        " elements out of range of destination list with size %" PRIhsz,
        dst_i, dst_i + count, count, dst_count);
  }

  // Prevent overlap when copying within the same list.
  if (src_list == dst_list && src_i + count > dst_i && dst_i + count > src_i) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "overlapping copy of range [%" PRIhsz ", %" PRIhsz
                            ") to [%" PRIhsz ", %" PRIhsz ") of %" PRIhsz
                            " elements not supported",
                            src_i, src_i + count, dst_i, dst_i + count, count);
  }

  // Copies into variant lists is a slow-path as we need to check the type of
  // each element we copy. Note that the source of the copy can be of any type.
  // When copying in the other direction of a variant list to a typed list we
  // need to ensure all copied elements match the expected destination type.
  if (dst_list->storage_mode == IREE_VM_LIST_STORAGE_MODE_VARIANT) {
    iree_vm_list_copy_to_variant_list(src_list, src_i, dst_list, dst_i, count);
    return iree_ok_status();
  } else if (src_list->storage_mode == IREE_VM_LIST_STORAGE_MODE_VARIANT) {
    return iree_vm_list_copy_from_variant_list(src_list, src_i, dst_list, dst_i,
                                               count);
  }

  // If neither source or destination are variant lists we need to match the
  // types exactly.
  if (src_list->storage_mode != dst_list->storage_mode ||
      memcmp(&src_list->element_type, &dst_list->element_type,
             sizeof(src_list->element_type)) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "src/dst element type mismatch");
  }

  if (src_list->storage_mode == IREE_VM_LIST_STORAGE_MODE_VALUE) {
    // Memcpy primitive values fast path.
    memcpy((uint8_t*)dst_list->storage + dst_i * dst_list->element_size,
           (uint8_t*)src_list->storage + src_i * src_list->element_size,
           count * dst_list->element_size);
  } else {
    // Retain ref fast(ish) path - note that iree_vm_ref_retain will release
    // any existing value in the dest list it overwrites.
    iree_vm_ref_t* src_ref_storage = (iree_vm_ref_t*)src_list->storage + src_i;
    iree_vm_ref_t* dst_ref_storage = (iree_vm_ref_t*)dst_list->storage + dst_i;
    for (iree_host_size_t i = 0; i < count; ++i) {
      iree_vm_ref_retain(src_ref_storage + i, dst_ref_storage + i);
    }
  }

  return iree_ok_status();
}

static void iree_vm_list_convert_value_type(
    const iree_vm_value_t* source_value, iree_vm_value_type_t target_value_type,
    iree_vm_value_t* out_value) {
  if (target_value_type == source_value->type) {
    memcpy(out_value, source_value, sizeof(*out_value));
    return;
  }
  out_value->type = target_value_type;
  out_value->i64 = 0;
  switch (source_value->type) {
    default:
      return;
    case IREE_VM_VALUE_TYPE_I8:
      switch (target_value_type) {
        case IREE_VM_VALUE_TYPE_I16:
          out_value->i16 = (int16_t)source_value->i8;
          return;
        case IREE_VM_VALUE_TYPE_I32:
          out_value->i32 = (int32_t)source_value->i8;
          return;
        case IREE_VM_VALUE_TYPE_I64:
          out_value->i64 = (int64_t)source_value->i8;
          return;
        default:
          return;
      }
    case IREE_VM_VALUE_TYPE_I16:
      switch (target_value_type) {
        case IREE_VM_VALUE_TYPE_I8:
          out_value->i8 = (int8_t)source_value->i16;
          return;
        case IREE_VM_VALUE_TYPE_I32:
          out_value->i32 = (int32_t)source_value->i16;
          return;
        case IREE_VM_VALUE_TYPE_I64:
          out_value->i64 = (int64_t)source_value->i16;
          return;
        default:
          return;
      }
    case IREE_VM_VALUE_TYPE_I32:
      switch (target_value_type) {
        case IREE_VM_VALUE_TYPE_I8:
          out_value->i8 = (int8_t)source_value->i32;
          return;
        case IREE_VM_VALUE_TYPE_I16:
          out_value->i16 = (int16_t)source_value->i32;
          return;
        case IREE_VM_VALUE_TYPE_I64:
          out_value->i64 = (int64_t)source_value->i32;
          return;
        default:
          return;
      }
    case IREE_VM_VALUE_TYPE_I64:
      switch (target_value_type) {
        case IREE_VM_VALUE_TYPE_I8:
          out_value->i8 = (int8_t)source_value->i64;
          return;
        case IREE_VM_VALUE_TYPE_I16:
          out_value->i16 = (int16_t)source_value->i64;
          return;
        case IREE_VM_VALUE_TYPE_I32:
          out_value->i32 = (int32_t)source_value->i64;
          return;
        default:
          return;
      }
  }
}

IREE_API_EXPORT iree_status_t
iree_vm_list_get_value(const iree_vm_list_t* list, iree_host_size_t i,
                       iree_vm_value_t* out_value) {
  if (i >= list->count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "index %zu out of bounds (%zu)", i, list->count);
  }
  uintptr_t element_ptr = (uintptr_t)list->storage + i * list->element_size;
  memset(out_value, 0, sizeof(*out_value));
  switch (list->storage_mode) {
    case IREE_VM_LIST_STORAGE_MODE_VALUE: {
      out_value->type = list->element_type.value_type;
      // TODO(benvanik): #ifdef on LITTLE/BIG_ENDIAN and just memcpy.
      switch (list->element_size) {
        case 1:
          out_value->i8 = *(int8_t*)element_ptr;
          break;
        case 2:
          out_value->i16 = *(int16_t*)element_ptr;
          break;
        case 4:
          out_value->i32 = *(int32_t*)element_ptr;
          break;
        case 8:
          out_value->i64 = *(int64_t*)element_ptr;
          break;
      }
      break;
    }
    case IREE_VM_LIST_STORAGE_MODE_VARIANT: {
      iree_vm_variant_t* variant = (iree_vm_variant_t*)element_ptr;
      if (!iree_vm_type_def_is_value(&variant->type)) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "variant at index %zu is not a value type", i);
      }
      out_value->type = variant->type.value_type;
      memcpy(out_value->value_storage, variant->value_storage,
             sizeof(out_value->value_storage));
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION);
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_list_get_value_as(
    const iree_vm_list_t* list, iree_host_size_t i,
    iree_vm_value_type_t value_type, iree_vm_value_t* out_value) {
  if (i >= list->count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "index %zu out of bounds (%zu)", i, list->count);
  }
  uintptr_t element_ptr = (uintptr_t)list->storage + i * list->element_size;
  iree_vm_value_t value;
  value.i64 = 0;
  switch (list->storage_mode) {
    case IREE_VM_LIST_STORAGE_MODE_VALUE: {
      value.type = list->element_type.value_type;
      // TODO(benvanik): #ifdef on LITTLE/BIG_ENDIAN and just memcpy.
      switch (list->element_size) {
        case 1:
          value.i8 = *(int8_t*)element_ptr;
          break;
        case 2:
          value.i16 = *(int16_t*)element_ptr;
          break;
        case 4:
          value.i32 = *(int32_t*)element_ptr;
          break;
        case 8:
          value.i64 = *(int64_t*)element_ptr;
          break;
      }
      break;
    }
    case IREE_VM_LIST_STORAGE_MODE_VARIANT: {
      iree_vm_variant_t* variant = (iree_vm_variant_t*)element_ptr;
      if (!iree_vm_type_def_is_value(&variant->type)) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "variant at index %zu is not a value type", i);
      }
      value.type = variant->type.value_type;
      memcpy(value.value_storage, variant->value_storage,
             sizeof(value.value_storage));
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "list does not store values");
  }
  iree_vm_list_convert_value_type(&value, value_type, out_value);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_list_set_value(
    iree_vm_list_t* list, iree_host_size_t i, const iree_vm_value_t* value) {
  if (i >= list->count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "index %zu out of bounds (%zu)", i, list->count);
  }
  iree_vm_value_type_t target_type;
  switch (list->storage_mode) {
    case IREE_VM_LIST_STORAGE_MODE_VALUE: {
      target_type = list->element_type.value_type;
      break;
    }
    case IREE_VM_LIST_STORAGE_MODE_VARIANT: {
      target_type = value->type;
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "list cannot store values");
  }
  iree_vm_value_t converted_value;
  iree_vm_list_convert_value_type(value, target_type, &converted_value);
  uintptr_t element_ptr = (uintptr_t)list->storage + i * list->element_size;
  switch (list->storage_mode) {
    case IREE_VM_LIST_STORAGE_MODE_VALUE: {
      // TODO(benvanik): #ifdef on LITTLE/BIG_ENDIAN and just memcpy.
      switch (list->element_size) {
        case 1:
          *(int8_t*)element_ptr = converted_value.i8;
          break;
        case 2:
          *(int16_t*)element_ptr = converted_value.i16;
          break;
        case 4:
          *(int32_t*)element_ptr = converted_value.i32;
          break;
        case 8:
          *(int64_t*)element_ptr = converted_value.i64;
          break;
      }
      break;
    }
    case IREE_VM_LIST_STORAGE_MODE_VARIANT: {
      iree_vm_variant_t* variant = (iree_vm_variant_t*)element_ptr;
      if (variant->type.ref_type) {
        iree_vm_ref_release(&variant->ref);
      }
      variant->type.value_type = target_type;
      variant->type.ref_type = IREE_VM_REF_TYPE_NULL;
      memcpy(variant->value_storage, converted_value.value_storage,
             sizeof(variant->value_storage));
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "list cannot store values");
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_vm_list_push_value(iree_vm_list_t* list, const iree_vm_value_t* value) {
  iree_host_size_t i = iree_vm_list_size(list);
  IREE_RETURN_IF_ERROR(iree_vm_list_resize(list, i + 1));
  return iree_vm_list_set_value(list, i, value);
}

IREE_API_EXPORT void* iree_vm_list_get_ref_deref(
    const iree_vm_list_t* list, iree_host_size_t i,
    const iree_vm_ref_type_descriptor_t* type_descriptor) {
  iree_vm_ref_t value = {0};
  iree_status_t status = iree_vm_list_get_ref_assign(list, i, &value);
  if (!iree_status_is_ok(iree_status_consume_code(status))) {
    return NULL;
  }
  status = iree_vm_ref_check(value, type_descriptor->type);
  if (!iree_status_is_ok(iree_status_consume_code(status))) {
    return NULL;
  }
  return value.ptr;
}

// Gets a ref type |list| element at |i| and stores it into |out_value|.
// If |is_retain|=true then the reference count is incremented and otherwise
// the ref type is assigned directly (as with iree_vm_ref_assign).
static iree_status_t iree_vm_list_get_ref_assign_or_retain(
    const iree_vm_list_t* list, iree_host_size_t i, bool is_retain,
    iree_vm_ref_t* out_value) {
  if (i >= list->count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "index %zu out of bounds (%zu)", i, list->count);
  }
  uintptr_t element_ptr = (uintptr_t)list->storage + i * list->element_size;
  switch (list->storage_mode) {
    case IREE_VM_LIST_STORAGE_MODE_REF: {
      iree_vm_ref_t* element_ref = (iree_vm_ref_t*)element_ptr;
      is_retain ? iree_vm_ref_retain(element_ref, out_value)
                : iree_vm_ref_assign(element_ref, out_value);
      break;
    }
    case IREE_VM_LIST_STORAGE_MODE_VARIANT: {
      iree_vm_variant_t* variant = (iree_vm_variant_t*)element_ptr;
      if (!iree_vm_variant_is_empty(*variant) &&
          !iree_vm_type_def_is_ref(&variant->type)) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION);
      }
      is_retain ? iree_vm_ref_retain(&variant->ref, out_value)
                : iree_vm_ref_assign(&variant->ref, out_value);
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "list does not store refs");
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_list_get_ref_assign(
    const iree_vm_list_t* list, iree_host_size_t i, iree_vm_ref_t* out_value) {
  return iree_vm_list_get_ref_assign_or_retain(list, i, /*is_retain=*/false,
                                               out_value);
}

IREE_API_EXPORT iree_status_t iree_vm_list_get_ref_retain(
    const iree_vm_list_t* list, iree_host_size_t i, iree_vm_ref_t* out_value) {
  return iree_vm_list_get_ref_assign_or_retain(list, i, /*is_retain=*/true,
                                               out_value);
}

static iree_status_t iree_vm_list_set_ref(iree_vm_list_t* list,
                                          iree_host_size_t i, bool is_move,
                                          iree_vm_ref_t* value) {
  if (i >= list->count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "index %zu out of bounds (%zu)", i, list->count);
  }
  uintptr_t element_ptr = (uintptr_t)list->storage + i * list->element_size;
  switch (list->storage_mode) {
    case IREE_VM_LIST_STORAGE_MODE_REF: {
      iree_vm_ref_t* element_ref = (iree_vm_ref_t*)element_ptr;
      IREE_RETURN_IF_ERROR(iree_vm_ref_retain_or_move_checked(
          is_move, value, list->element_type.ref_type, element_ref));
      break;
    }
    case IREE_VM_LIST_STORAGE_MODE_VARIANT: {
      iree_vm_variant_t* variant = (iree_vm_variant_t*)element_ptr;
      if (variant->type.value_type) {
        memset(&variant->ref, 0, sizeof(variant->ref));
      }
      variant->type.value_type = IREE_VM_VALUE_TYPE_NONE;
      variant->type.ref_type = value->type;
      iree_vm_ref_retain_or_move(is_move, value, &variant->ref);
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "list cannot store refs");
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_list_set_ref_retain(
    iree_vm_list_t* list, iree_host_size_t i, const iree_vm_ref_t* value) {
  return iree_vm_list_set_ref(list, i, /*is_move=*/false,
                              (iree_vm_ref_t*)value);
}

IREE_API_EXPORT iree_status_t
iree_vm_list_push_ref_retain(iree_vm_list_t* list, const iree_vm_ref_t* value) {
  iree_host_size_t i = iree_vm_list_size(list);
  IREE_RETURN_IF_ERROR(iree_vm_list_resize(list, i + 1));
  return iree_vm_list_set_ref_retain(list, i, value);
}

IREE_API_EXPORT iree_status_t iree_vm_list_set_ref_move(iree_vm_list_t* list,
                                                        iree_host_size_t i,
                                                        iree_vm_ref_t* value) {
  return iree_vm_list_set_ref(list, i, /*is_move=*/true, value);
}

IREE_API_EXPORT iree_status_t iree_vm_list_push_ref_move(iree_vm_list_t* list,
                                                         iree_vm_ref_t* value) {
  iree_host_size_t i = iree_vm_list_size(list);
  IREE_RETURN_IF_ERROR(iree_vm_list_resize(list, i + 1));
  return iree_vm_list_set_ref_move(list, i, value);
}

IREE_API_EXPORT iree_status_t iree_vm_list_pop_front_ref_move(
    iree_vm_list_t* list, iree_vm_ref_t* out_value) {
  iree_host_size_t list_size = iree_vm_list_size(list);
  if (list_size == 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "cannot pop from an empty list");
  }
  IREE_RETURN_IF_ERROR(iree_vm_list_get_ref_assign(list, 0, out_value));
  memmove(list->storage, (uint8_t*)list->storage + list->element_size,
          (list_size - 1) * list->element_size);
  --list->count;
  memset((uint8_t*)list->storage + list->count * list->element_size, 0,
         list->element_size);
  return iree_ok_status();
}

typedef enum {
  IREE_VM_LIST_REF_ASSIGN = 0,
  IREE_VM_LIST_REF_RETAIN,
  IREE_VM_LIST_REF_MOVE,
} iree_vm_list_ref_mode_t;

static void iree_vm_list_ref_op(iree_vm_list_ref_mode_t mode,
                                iree_vm_ref_t* ref, iree_vm_ref_t* out_ref) {
  switch (mode) {
    case IREE_VM_LIST_REF_ASSIGN:
      iree_vm_ref_assign(ref, out_ref);
      break;
    case IREE_VM_LIST_REF_RETAIN:
      iree_vm_ref_retain(ref, out_ref);
      break;
    case IREE_VM_LIST_REF_MOVE:
      iree_vm_ref_move(ref, out_ref);
      break;
  }
}

static iree_status_t iree_vm_list_get_variant(const iree_vm_list_t* list,
                                              iree_host_size_t i,
                                              iree_vm_list_ref_mode_t ref_mode,
                                              iree_vm_variant_t* out_variant) {
  IREE_ASSERT_ARGUMENT(list);
  IREE_ASSERT_ARGUMENT(out_variant);
  if (i >= list->count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "index %zu out of bounds (%zu)", i, list->count);
  }
  iree_vm_variant_reset(out_variant);
  uintptr_t element_ptr = (uintptr_t)list->storage + i * list->element_size;
  switch (list->storage_mode) {
    case IREE_VM_LIST_STORAGE_MODE_VALUE: {
      out_variant->type = list->element_type;
      memcpy(out_variant->value_storage, (void*)element_ptr,
             list->element_size);
      break;
    }
    case IREE_VM_LIST_STORAGE_MODE_REF: {
      iree_vm_ref_t* element_ref = (iree_vm_ref_t*)element_ptr;
      out_variant->type.ref_type = element_ref->type;
      out_variant->type.value_type = IREE_VM_VALUE_TYPE_NONE;
      iree_vm_list_ref_op(ref_mode, element_ref, &out_variant->ref);
      break;
    }
    case IREE_VM_LIST_STORAGE_MODE_VARIANT: {
      iree_vm_variant_t* variant = (iree_vm_variant_t*)element_ptr;
      out_variant->type = variant->type;
      if (iree_vm_type_def_is_ref(&variant->type)) {
        iree_vm_list_ref_op(ref_mode, &variant->ref, &out_variant->ref);
      } else {
        memcpy(out_variant->value_storage, variant->value_storage,
               sizeof(variant->value_storage));
      }
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION);
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_vm_list_get_variant_assign(const iree_vm_list_t* list, iree_host_size_t i,
                                iree_vm_variant_t* out_variant) {
  return iree_vm_list_get_variant(list, i, IREE_VM_LIST_REF_ASSIGN,
                                  out_variant);
}

IREE_API_EXPORT iree_status_t
iree_vm_list_get_variant_retain(const iree_vm_list_t* list, iree_host_size_t i,
                                iree_vm_variant_t* out_variant) {
  return iree_vm_list_get_variant(list, i, IREE_VM_LIST_REF_RETAIN,
                                  out_variant);
}

IREE_API_EXPORT iree_status_t
iree_vm_list_get_variant_move(const iree_vm_list_t* list, iree_host_size_t i,
                              iree_vm_variant_t* out_variant) {
  return iree_vm_list_get_variant(list, i, IREE_VM_LIST_REF_MOVE, out_variant);
}

static iree_status_t iree_vm_list_set_variant(iree_vm_list_t* list,
                                              iree_host_size_t i, bool is_move,
                                              iree_vm_variant_t* variant) {
  if (iree_vm_type_def_is_variant(&variant->type)) {
    iree_vm_value_t value = iree_vm_variant_value(*variant);
    return iree_vm_list_set_value(list, i, &value);
  } else if (iree_vm_type_def_is_value(&variant->type)) {
    iree_vm_value_t value = {
        .type = variant->type.value_type,
    };
    memcpy(value.value_storage, variant->value_storage,
           sizeof(value.value_storage));
    return iree_vm_list_set_value(list, i, &value);
  } else if (iree_vm_type_def_is_ref(&variant->type)) {
    iree_status_t status =
        iree_vm_list_set_ref(list, i, is_move, &variant->ref);
    if (iree_status_is_ok(status) && is_move) {
      variant->type.ref_type = IREE_VM_REF_TYPE_NULL;
    }
    return status;
  } else {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "unhandled variant value type");
  }
}

IREE_API_EXPORT iree_status_t
iree_vm_list_set_variant_retain(iree_vm_list_t* list, iree_host_size_t i,
                                const iree_vm_variant_t* variant) {
  return iree_vm_list_set_variant(list, i, /*is_move=*/false,
                                  (iree_vm_variant_t*)variant);
}

IREE_API_EXPORT iree_status_t iree_vm_list_set_variant_move(
    iree_vm_list_t* list, iree_host_size_t i, iree_vm_variant_t* variant) {
  return iree_vm_list_set_variant(list, i, /*is_move=*/true, variant);
}

static iree_status_t iree_vm_list_push_variant(
    iree_vm_list_t* list, bool is_move, const iree_vm_variant_t* variant) {
  iree_host_size_t i = iree_vm_list_size(list);
  IREE_RETURN_IF_ERROR(iree_vm_list_resize(list, i + 1));
  return iree_vm_list_set_variant(list, i, is_move,
                                  (iree_vm_variant_t*)variant);
}

IREE_API_EXPORT iree_status_t iree_vm_list_push_variant_retain(
    iree_vm_list_t* list, const iree_vm_variant_t* variant) {
  return iree_vm_list_push_variant(list, /*is_move=*/false, variant);
}

IREE_API_EXPORT iree_status_t iree_vm_list_push_variant_move(
    iree_vm_list_t* list, iree_vm_variant_t* variant) {
  return iree_vm_list_push_variant(list, /*is_move=*/true, variant);
}

iree_status_t iree_vm_list_register_types(iree_vm_instance_t* instance) {
  if (iree_vm_list_descriptor.type != IREE_VM_REF_TYPE_NULL) {
    // Already registered.
    return iree_ok_status();
  }

  iree_vm_list_descriptor.destroy = iree_vm_list_destroy;
  iree_vm_list_descriptor.offsetof_counter =
      offsetof(iree_vm_list_t, ref_object.counter);
  iree_vm_list_descriptor.type_name = iree_make_cstring_view("vm.list");
  return iree_vm_ref_register_type(&iree_vm_list_descriptor);
}
