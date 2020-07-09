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

#ifndef IREE_VM_LIST_H_
#define IREE_VM_LIST_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/vm/ref.h"
#include "iree/vm/type_def.h"
#include "iree/vm/value.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// A growable list that can hold primitive value types or ref objects or a mix.
// This maps to the `!vm.list<...>` type in the VM IR and is designed to enable
// flexible interop between hosting applications using the VM C API to invoke IR
// and custom modules that need to pass arbitrary list-like data across the VM
// ABI. It is not designed for efficiency: if you are performing large amounts
// of work on the list type you should instead be representing that using the
// HAL types so that you can get acceleration.
typedef struct iree_vm_list iree_vm_list_t;

#ifndef IREE_API_NO_PROTOTYPES

// Creates a growable list containing the given |element_type|, which may either
// be a primitive iree_vm_value_type_t value (like i32) or a ref type. When
// storing ref types the list may either store a specific iree_vm_ref_type_t
// and ensure that all elements set match the type or IREE_VM_REF_TYPE_ANY to
// indicate that any ref type is allowed.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_list_create(
    const iree_vm_type_def_t* element_type, iree_host_size_t initial_capacity,
    iree_allocator_t allocator, iree_vm_list_t** out_list);

// Retains the given |list| for the caller.
IREE_API_EXPORT void IREE_API_CALL iree_vm_list_retain(iree_vm_list_t* list);

// Releases the given |list| from the caller.
IREE_API_EXPORT void IREE_API_CALL iree_vm_list_release(iree_vm_list_t* list);

// Returns the element type stored in the list.
IREE_API_EXPORT iree_status_t iree_vm_list_element_type(
    const iree_vm_list_t* list, iree_vm_type_def_t* out_element_type);

// Returns the capacity of the list in elements.
IREE_API_EXPORT iree_host_size_t IREE_API_CALL
iree_vm_list_capacity(const iree_vm_list_t* list);

// Reserves storage for at least minimum_capacity elements. If the list already
// has at least the specified capacity the operation is ignored.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_list_reserve(iree_vm_list_t* list, iree_host_size_t minimum_capacity);

// Returns the current size of the list in elements.
IREE_API_EXPORT iree_host_size_t IREE_API_CALL
iree_vm_list_size(const iree_vm_list_t* list);

// Resizes the list to contain new_size elements. This will either truncate
// the list if the existing size is greater than new_size or extend the list
// with the default list value of 0 if storing primitives, null if refs, or
// empty if variants.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_list_resize(iree_vm_list_t* list, iree_host_size_t new_size);

// Returns the value of the element at the given index.
// Note that the value type may vary from element to element in variant lists
// and callers should check the |out_value| type.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_list_get_value(
    const iree_vm_list_t* list, iree_host_size_t i, iree_vm_value_t* out_value);

// Returns the value of the element at the given index. If the specified
// |value_type| differs from the list storage type the value will be converted
// using the value type semantics (such as sign/zero extend, etc).
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_list_get_value_as(
    const iree_vm_list_t* list, iree_host_size_t i,
    iree_vm_value_type_t value_type, iree_vm_value_t* out_value);

// Sets the value of the element at the given index. If the specified |value|
// type differs from the list storage type the value will be converted using the
// value type semantics (such as sign/zero extend, etc).
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_list_set_value(
    iree_vm_list_t* list, iree_host_size_t i, const iree_vm_value_t* value);

// Returns the ref value of the element at the given index.
// The ref will be retained and must be released by the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_list_get_ref_retain(
    const iree_vm_list_t* list, iree_host_size_t i, iree_vm_ref_t* out_value);

// Sets the ref value of the element at the given index, retaining a reference
// in the list until the element is cleared or the list is disposed.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_list_set_ref_retain(
    iree_vm_list_t* list, iree_host_size_t i, const iree_vm_ref_t* value);

// Sets the ref value of the element at the given index, moving ownership of the
// |value| reference to the list.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_list_set_ref_move(
    iree_vm_list_t* list, iree_host_size_t i, iree_vm_ref_t* value);

// Returns the value of the element at the given index. If the element contains
// a ref then it will be retained and must be released by the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_list_get_variant(const iree_vm_list_t* list, iree_host_size_t i,
                         iree_vm_variant2_t* out_value);

// Sets the value of the element at the given index. If the specified |value|
// type differs from the list storage type the value will be converted using the
// value type semantics (such as sign/zero extend, etc). If the variant is a ref
// then it will be retained.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_list_set_variant(
    iree_vm_list_t* list, iree_host_size_t i, const iree_vm_variant2_t* value);

#endif  // IREE_API_NO_PROTOTYPES

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

IREE_VM_DECLARE_TYPE_ADAPTERS(iree_vm_list, iree_vm_list_t);

#endif  // IREE_VM_LIST_H_
