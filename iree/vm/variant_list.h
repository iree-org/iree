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

#ifndef IREE_VM_VARIANT_LIST_H_
#define IREE_VM_VARIANT_LIST_H_

#include <stdint.h>

#include "iree/vm/ref.h"
#include "iree/vm/value.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// A list able to hold both value and ref types, used for marshaling host
// language calls to VM calls.
//
// Lists may either be stack or heap allocated depending on how long they must
// live. Prefer stack allocations when APIs receiving the lists document
// themselves as copying the list values.
typedef struct iree_vm_variant_list iree_vm_variant_list_t;

// An element of an iree_vm_variant_list_t.
typedef struct {
  iree_vm_value_type_t value_type : 8;
  iree_vm_ref_type_t ref_type : 24;
  union {
    int32_t i32;
    iree_vm_ref_t ref;
  };
} iree_vm_variant_t;

#define IREE_VM_VARIANT_IS_VALUE(v) ((v)->value_type != IREE_VM_VALUE_TYPE_NONE)
#define IREE_VM_VARIANT_IS_REF(v) !IREE_VM_VARIANT_IS_VALUE(v)

#ifndef IREE_API_NO_PROTOTYPES

// Allocates a list with the maximum |capacity|.
// The list must be freed with iree_vm_variant_list_free unless ownership is
// transferred to code that will perform the free as documented in its API.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_variant_list_alloc(
    iree_host_size_t capacity, iree_allocator_t allocator,
    iree_vm_variant_list_t** out_list);

// Returns the size, in bytes, required to store a list of the given capacity.
// This can be used to stack-allocate the variant list and then wrap the memory
// for use as a list with iree_vm_variant_list_init.
IREE_API_EXPORT iree_host_size_t IREE_API_CALL
iree_vm_variant_list_alloc_size(iree_host_size_t capacity);

// Initializes the allocated |list| with the given |capacity|. The list
// allocation must be at least the size returned by
// iree_vm_variant_list_alloc_size for the same |capacity|.
// The list must be freed with iree_vm_variant_list_free unless ownership is
// transferred to code that will perform the free as documented in its API.
IREE_API_EXPORT void IREE_API_CALL iree_vm_variant_list_init(
    iree_vm_variant_list_t* list, iree_host_size_t capacity);

// Frees the list using the allocator it was originally allocated from.
IREE_API_EXPORT void IREE_API_CALL
iree_vm_variant_list_free(iree_vm_variant_list_t* list);

// Returns the capacity of the list in elements.
IREE_API_EXPORT iree_host_size_t IREE_API_CALL
iree_vm_variant_list_capacity(const iree_vm_variant_list_t* list);

// Returns the total number of elements added to the list.
IREE_API_EXPORT iree_host_size_t IREE_API_CALL
iree_vm_variant_list_size(const iree_vm_variant_list_t* list);

// Appends a primitive value to the list.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_variant_list_append_value(
    iree_vm_variant_list_t* list, iree_vm_value_t value);

// Appends a ref object to the list by retaining it.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_variant_list_append_ref_retain(iree_vm_variant_list_t* list,
                                       iree_vm_ref_t* ref);

// Appends a ref object to the list by moving it.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_variant_list_append_ref_move(iree_vm_variant_list_t* list,
                                     iree_vm_ref_t* ref);

// Appends a null ref to the list.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_variant_list_append_null_ref(iree_vm_variant_list_t* list);

// Returns a pointer to the variant element at the given index.
IREE_API_EXPORT iree_vm_variant_t* IREE_API_CALL
iree_vm_variant_list_get(iree_vm_variant_list_t* list, iree_host_size_t i);

#endif  // IREE_API_NO_PROTOTYPES

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_VARIANT_LIST_H_
