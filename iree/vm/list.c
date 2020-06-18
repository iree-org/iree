// Copyright 2020 Google LLC
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

#include "iree/vm/list.h"

#include "iree/base/alignment.h"

// Size of each iree_vm_value_type_t in bytes.
static const iree_host_size_t kValueTypeSizes[5] = {
    0,  // IREE_VM_VALUE_TYPE_NONE
    1,  // IREE_VM_VALUE_TYPE_I8
    2,  // IREE_VM_VALUE_TYPE_I16
    4,  // IREE_VM_VALUE_TYPE_I32
    8,  // IREE_VM_VALUE_TYPE_I64
};
static_assert(IREE_VM_VALUE_TYPE_COUNT ==
                  (sizeof(kValueTypeSizes) / sizeof(kValueTypeSizes[0])),
              "Enum mismatch");

// Defines how the iree_vm_list_t storage is allocated and what elements are
// interpreted as.
typedef enum iree_vm_list_storage_mode {
  // Each element is a primitive value and stored as a dense array.
  IREE_VM_LIST_STORAGE_MODE_VALUE = 0,
  // Each element is an iree_vm_ref_t of some type.
  IREE_VM_LIST_STORAGE_MODE_REF = 1,
  // Each element is a variant of any type (possibly all different).
  IREE_VM_LIST_STORAGE_MODE_VARIANT = 2,
} iree_vm_list_storage_mode_t;

// A list able to hold either flat primitive elements or ref values.
struct iree_vm_list {
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

static iree_vm_ref_type_descriptor_t iree_vm_list_descriptor = {0};

IREE_VM_DEFINE_TYPE_ADAPTERS(iree_vm_list, iree_vm_list_t);

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_list_create(
    const iree_vm_type_def_t* element_type, iree_host_size_t initial_capacity,
    iree_allocator_t allocator, iree_vm_list_t** out_list) {
  iree_vm_list_t* list = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(iree_vm_list_t), (void**)&list));
  memset(list, 0, sizeof(*list));
  iree_atomic_store(&list->ref_object.counter, 1);
  list->allocator = allocator;
  list->element_type = *element_type;

  if (iree_vm_type_def_is_value(element_type)) {
    list->storage_mode = IREE_VM_LIST_STORAGE_MODE_VALUE;
    list->element_size = kValueTypeSizes[element_type->value_type];
  } else if (iree_vm_type_def_is_ref(element_type)) {
    list->storage_mode = IREE_VM_LIST_STORAGE_MODE_REF;
    list->element_size = sizeof(iree_vm_ref_t);
  } else {
    list->storage_mode = IREE_VM_LIST_STORAGE_MODE_VARIANT;
    list->element_size = sizeof(iree_vm_variant2_t);
  }

  iree_status_t status = iree_vm_list_reserve(list, initial_capacity);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(allocator, list);
    return status;
  }

  *out_list = list;
  return IREE_STATUS_OK;
}

static void iree_vm_list_reset_range(iree_vm_list_t* list,
                                     iree_host_size_t offset,
                                     iree_host_size_t length) {
  switch (list->storage_mode) {
    case IREE_VM_LIST_STORAGE_MODE_VALUE:
      // Nothing special, freeing the storage is all we need.
      break;
    case IREE_VM_LIST_STORAGE_MODE_REF: {
      iree_vm_ref_t* ref_storage = (iree_vm_ref_t*)list->storage;
      for (iree_host_size_t i = offset; i < length; ++i) {
        iree_vm_ref_release(&ref_storage[i]);
      }
      break;
    }
    case IREE_VM_LIST_STORAGE_MODE_VARIANT: {
      iree_vm_variant2_t* variant_storage = (iree_vm_variant2_t*)list->storage;
      for (iree_host_size_t i = offset; i < length; ++i) {
        if (iree_vm_type_def_is_ref(&variant_storage[i].type)) {
          iree_vm_ref_release(&variant_storage[i].ref);
        }
      }
      break;
    }
  }
}

static void iree_vm_list_destroy(void* ptr) {
  iree_vm_list_t* list = (iree_vm_list_t*)ptr;
  iree_vm_list_reset_range(list, 0, list->count);
  iree_allocator_free(list->allocator, list->storage);
  iree_allocator_free(list->allocator, list);
}

IREE_API_EXPORT void IREE_API_CALL iree_vm_list_retain(iree_vm_list_t* list) {
  iree_vm_ref_object_retain(list, &iree_vm_list_descriptor);
}

IREE_API_EXPORT void IREE_API_CALL iree_vm_list_release(iree_vm_list_t* list) {
  iree_vm_ref_object_release(list, &iree_vm_list_descriptor);
}

IREE_API_EXPORT iree_status_t iree_vm_list_element_type(
    const iree_vm_list_t* list, iree_vm_type_def_t* out_element_type) {
  *out_element_type = list->element_type;
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_host_size_t IREE_API_CALL
iree_vm_list_capacity(const iree_vm_list_t* list) {
  return list->capacity;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_list_reserve(iree_vm_list_t* list, iree_host_size_t minimum_capacity) {
  if (list->capacity >= minimum_capacity) {
    return IREE_STATUS_OK;
  }
  iree_host_size_t old_capacity = list->capacity;
  iree_host_size_t new_capacity = iree_align(minimum_capacity, 64);
  IREE_RETURN_IF_ERROR(iree_allocator_realloc(
      list->allocator, new_capacity * list->element_size, &list->storage));
  memset((void*)((uintptr_t)list->storage + old_capacity * list->element_size),
         0, (new_capacity - old_capacity) * list->element_size);
  list->capacity = new_capacity;
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_host_size_t IREE_API_CALL
iree_vm_list_size(const iree_vm_list_t* list) {
  return list->count;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_list_resize(iree_vm_list_t* list, iree_host_size_t new_size) {
  if (new_size == list->count) {
    return IREE_STATUS_OK;
  } else if (new_size < list->count) {
    // Truncating.
    iree_vm_list_reset_range(list, new_size + 1, list->count - new_size);
    list->count = new_size;
  } else if (new_size > list->capacity) {
    // Extending beyond capacity.
    IREE_RETURN_IF_ERROR(iree_vm_list_reserve(list, new_size));
  }
  list->count = new_size;
  return IREE_STATUS_OK;
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

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_list_get_value(const iree_vm_list_t* list, iree_host_size_t i,
                       iree_vm_value_t* out_value) {
  if (i >= list->count) return IREE_STATUS_OUT_OF_RANGE;
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
      iree_vm_variant2_t* variant = (iree_vm_variant2_t*)element_ptr;
      if (!iree_vm_type_def_is_value(&variant->type)) {
        return IREE_STATUS_FAILED_PRECONDITION;
      }
      out_value->type = variant->type.value_type;
      out_value->i64 = variant->i64;
      break;
    }
    default:
      return IREE_STATUS_FAILED_PRECONDITION;
  }
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_list_get_value_as(
    const iree_vm_list_t* list, iree_host_size_t i,
    iree_vm_value_type_t value_type, iree_vm_value_t* out_value) {
  if (i >= list->count) return IREE_STATUS_OUT_OF_RANGE;
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
      iree_vm_variant2_t* variant = (iree_vm_variant2_t*)element_ptr;
      if (!iree_vm_type_def_is_value(&variant->type)) {
        return IREE_STATUS_FAILED_PRECONDITION;
      }
      value.type = variant->type.value_type;
      value.i64 = variant->i64;
      break;
    }
    default:
      return IREE_STATUS_FAILED_PRECONDITION;
  }
  iree_vm_list_convert_value_type(&value, value_type, out_value);
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_list_set_value(
    iree_vm_list_t* list, iree_host_size_t i, const iree_vm_value_t* value) {
  if (i >= list->count) return IREE_STATUS_OUT_OF_RANGE;
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
      return IREE_STATUS_FAILED_PRECONDITION;
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
      iree_vm_variant2_t* variant = (iree_vm_variant2_t*)element_ptr;
      variant->type.value_type = target_type;
      variant->type.ref_type = IREE_VM_REF_TYPE_NULL;
      variant->i64 = converted_value.i64;
      break;
    }
    default:
      return IREE_STATUS_FAILED_PRECONDITION;
  }
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_list_get_ref_retain(
    const iree_vm_list_t* list, iree_host_size_t i, iree_vm_ref_t* out_value) {
  if (i >= list->count) return IREE_STATUS_OUT_OF_RANGE;
  uintptr_t element_ptr = (uintptr_t)list->storage + i * list->element_size;
  switch (list->storage_mode) {
    case IREE_VM_LIST_STORAGE_MODE_REF: {
      iree_vm_ref_t* element_ref = (iree_vm_ref_t*)element_ptr;
      iree_vm_ref_retain(element_ref, out_value);
      break;
    }
    case IREE_VM_LIST_STORAGE_MODE_VARIANT: {
      iree_vm_variant2_t* variant = (iree_vm_variant2_t*)element_ptr;
      if (!iree_vm_type_def_is_ref(&variant->type)) {
        return IREE_STATUS_FAILED_PRECONDITION;
      }
      iree_vm_ref_retain(&variant->ref, out_value);
      break;
    }
    default:
      return IREE_STATUS_FAILED_PRECONDITION;
  }
  return IREE_STATUS_OK;
}

static iree_status_t IREE_API_CALL iree_vm_list_set_ref(iree_vm_list_t* list,
                                                        iree_host_size_t i,
                                                        bool is_move,
                                                        iree_vm_ref_t* value) {
  if (i >= list->count) return IREE_STATUS_OUT_OF_RANGE;
  uintptr_t element_ptr = (uintptr_t)list->storage + i * list->element_size;
  switch (list->storage_mode) {
    case IREE_VM_LIST_STORAGE_MODE_REF: {
      iree_vm_ref_t* element_ref = (iree_vm_ref_t*)element_ptr;
      IREE_RETURN_IF_ERROR(iree_vm_ref_retain_or_move_checked(
          is_move, value, list->element_type.ref_type, element_ref));
      break;
    }
    case IREE_VM_LIST_STORAGE_MODE_VARIANT: {
      iree_vm_variant2_t* variant = (iree_vm_variant2_t*)element_ptr;
      variant->type.value_type = IREE_VM_VALUE_TYPE_NONE;
      variant->type.ref_type = value->type;
      iree_vm_ref_retain_or_move(is_move, value, &variant->ref);
      break;
    }
    default:
      return IREE_STATUS_FAILED_PRECONDITION;
  }
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_list_set_ref_retain(
    iree_vm_list_t* list, iree_host_size_t i, const iree_vm_ref_t* value) {
  return iree_vm_list_set_ref(list, i, /*is_move=*/false,
                              (iree_vm_ref_t*)value);
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_list_set_ref_move(
    iree_vm_list_t* list, iree_host_size_t i, iree_vm_ref_t* value) {
  return iree_vm_list_set_ref(list, i, /*is_move=*/true, value);
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_list_get_variant(const iree_vm_list_t* list, iree_host_size_t i,
                         iree_vm_variant2_t* out_value) {
  return IREE_STATUS_UNIMPLEMENTED;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_list_set_variant(
    iree_vm_list_t* list, iree_host_size_t i, const iree_vm_variant2_t* value) {
  return IREE_STATUS_UNIMPLEMENTED;
}

iree_status_t iree_vm_list_register_types() {
  iree_vm_list_descriptor.destroy = iree_vm_list_destroy;
  iree_vm_list_descriptor.offsetof_counter =
      offsetof(iree_vm_list_t, ref_object.counter);
  iree_vm_list_descriptor.type_name = iree_make_cstring_view("vm.list");
  return iree_vm_ref_register_type(&iree_vm_list_descriptor);
}
