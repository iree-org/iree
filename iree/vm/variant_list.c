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

#include "iree/vm/variant_list.h"

struct iree_vm_variant_list {
  iree_allocator_t allocator;
  iree_host_size_t capacity;
  iree_host_size_t count;
  iree_vm_variant_t values[];
};

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_variant_list_alloc(
    iree_host_size_t capacity, iree_allocator_t allocator,
    iree_vm_variant_list_t** out_list) {
  if (!out_list) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_list = NULL;

  iree_host_size_t alloc_size = iree_vm_variant_list_alloc_size(capacity);
  iree_vm_variant_list_t* list = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, alloc_size, (void**)&list));
  iree_vm_variant_list_init(list, capacity);
  list->allocator = allocator;
  *out_list = list;
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_host_size_t IREE_API_CALL
iree_vm_variant_list_alloc_size(iree_host_size_t capacity) {
  return sizeof(iree_vm_variant_list_t) + sizeof(iree_vm_variant_t) * capacity;
}

IREE_API_EXPORT void IREE_API_CALL iree_vm_variant_list_init(
    iree_vm_variant_list_t* list, iree_host_size_t capacity) {
  memset(&list->allocator, 0, sizeof(list->allocator));
  list->capacity = capacity;
  list->count = 0;
  memset(list->values, 0, sizeof(list->values[0]) * capacity);
}

IREE_API_EXPORT void IREE_API_CALL
iree_vm_variant_list_free(iree_vm_variant_list_t* list) {
  for (iree_host_size_t i = 0; i < list->count; ++i) {
    if (IREE_VM_VARIANT_IS_REF(&list->values[i])) {
      iree_vm_ref_release(&list->values[i].ref);
    }
  }
  iree_allocator_free(list->allocator, list);
}

IREE_API_EXPORT iree_host_size_t IREE_API_CALL
iree_vm_variant_list_capacity(const iree_vm_variant_list_t* list) {
  return list->capacity;
}

IREE_API_EXPORT iree_host_size_t IREE_API_CALL
iree_vm_variant_list_size(const iree_vm_variant_list_t* list) {
  return list->count;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_variant_list_append_value(
    iree_vm_variant_list_t* list, iree_vm_value_t value) {
  if (list->count + 1 > list->capacity) {
    return IREE_STATUS_OUT_OF_RANGE;
  }
  iree_host_size_t i = list->count++;
  list->values[i].value_type = value.type;
  list->values[i].ref_type = IREE_VM_REF_TYPE_NULL;
  list->values[i].i32 = value.i32;
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_variant_list_append_ref_retain(iree_vm_variant_list_t* list,
                                       iree_vm_ref_t* ref) {
  if (list->count + 1 > list->capacity) {
    return IREE_STATUS_OUT_OF_RANGE;
  }
  iree_host_size_t i = list->count++;
  list->values[i].value_type = IREE_VM_VALUE_TYPE_NONE;
  list->values[i].ref_type = ref->type;
  iree_vm_ref_retain(ref, &list->values[i].ref);
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_variant_list_append_ref_move(iree_vm_variant_list_t* list,
                                     iree_vm_ref_t* ref) {
  if (list->count + 1 > list->capacity) {
    return IREE_STATUS_OUT_OF_RANGE;
  }
  iree_host_size_t i = list->count++;
  list->values[i].value_type = IREE_VM_VALUE_TYPE_NONE;
  list->values[i].ref_type = ref->type;
  iree_vm_ref_move(ref, &list->values[i].ref);
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_variant_list_append_null_ref(iree_vm_variant_list_t* list) {
  if (list->count + 1 > list->capacity) {
    return IREE_STATUS_OUT_OF_RANGE;
  }
  iree_host_size_t i = list->count++;
  list->values[i].value_type = IREE_VM_VALUE_TYPE_NONE;
  list->values[i].ref_type = IREE_VM_REF_TYPE_NULL;
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_vm_variant_t* IREE_API_CALL
iree_vm_variant_list_get(iree_vm_variant_list_t* list, iree_host_size_t i) {
  if (i < 0 || i > list->count) return NULL;
  return &list->values[i];
}
