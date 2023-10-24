// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_LIST_H_
#define IREE_VM_LIST_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/vm/ref.h"
#include "iree/vm/type_def.h"
#include "iree/vm/value.h"
#include "iree/vm/variant.h"

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
//
// This type the same performance characteristics as std::vector; pushes may
// grow the capacity of the list and to ensure minimal wastage it is always
// better to reserve the exact desired element count first.
typedef struct iree_vm_list_t iree_vm_list_t;

// Returns the size in bytes required to store a list with the given element
// type and capacity. This storage size can be used to stack allocate or reserve
// memory that is then used by iree_vm_list_initialize to avoid dynamic
// allocations.
IREE_API_EXPORT iree_host_size_t iree_vm_list_storage_size(
    const iree_vm_type_def_t* element_type, iree_host_size_t capacity);

// Initializes a statically-allocated list in the |storage| memory.
// The storage capacity must be large enough to hold the list internals and
// its contents which may vary across compilers/platforms/etc; use
// iree_vm_list_storage_size to query the required capacity.
//
// Statically-allocated lists have their lifetime controlled by the caller and
// must be deinitialized with iree_vm_list_deinitialize only when there are no
// more users of the list.
IREE_API_EXPORT iree_status_t iree_vm_list_initialize(
    iree_byte_span_t storage, const iree_vm_type_def_t* element_type,
    iree_host_size_t capacity, iree_vm_list_t** out_list);

// Deinitializes a statically-allocated |list| previously initialized with
// iree_vm_list_initialize. Aborts if there are still references remaining.
IREE_API_EXPORT void iree_vm_list_deinitialize(iree_vm_list_t* list);

// Creates a growable list containing the given |element_type|, which may either
// be a primitive iree_vm_value_type_t value (like i32) or a ref type. When
// storing ref types the list may either store a specific iree_vm_ref_type_t
// and ensure that all elements set match the type or IREE_VM_REF_TYPE_ANY to
// indicate that any ref type is allowed.
//
// |element_type| can be set to iree_vm_make_undefined_type_def to indicate that
// the list stores variants (each element can differ in type).
//
// Note: list types *must* be registered, e.g. by creating a VM instance with
// `iree_vm_instance_create()`, prior to using this API.
IREE_API_EXPORT iree_status_t iree_vm_list_create(
    const iree_vm_type_def_t element_type, iree_host_size_t initial_capacity,
    iree_allocator_t allocator, iree_vm_list_t** out_list);

// Shallowly clones |source| into |out_target|.
// The resulting list will be have its capacity set to the |source| size.
IREE_API_EXPORT iree_status_t
iree_vm_list_clone(iree_vm_list_t* source, iree_allocator_t host_allocator,
                   iree_vm_list_t** out_target);

// Retains the given |list| for the caller.
IREE_API_EXPORT void iree_vm_list_retain(iree_vm_list_t* list);

// Releases the given |list| from the caller.
IREE_API_EXPORT void iree_vm_list_release(iree_vm_list_t* list);

// Returns the element type stored in the list.
IREE_API_EXPORT iree_vm_type_def_t
iree_vm_list_element_type(const iree_vm_list_t* list);

// Returns the capacity of the list in elements.
IREE_API_EXPORT iree_host_size_t
iree_vm_list_capacity(const iree_vm_list_t* list);

// Reserves storage for at least minimum_capacity elements. If the list already
// has at least the specified capacity the operation is ignored.
IREE_API_EXPORT iree_status_t
iree_vm_list_reserve(iree_vm_list_t* list, iree_host_size_t minimum_capacity);

// Returns the current size of the list in elements.
IREE_API_EXPORT iree_host_size_t iree_vm_list_size(const iree_vm_list_t* list);

// Resizes the list to contain new_size elements. This will either truncate
// the list if the existing size is greater than new_size or extend the list
// with the default list value of 0 if storing primitives, null if refs, or
// empty if variants.
IREE_API_EXPORT iree_status_t iree_vm_list_resize(iree_vm_list_t* list,
                                                  iree_host_size_t new_size);

// Clears the list contents. Equivalent to resizing to 0.
IREE_API_EXPORT void iree_vm_list_clear(iree_vm_list_t* list);

// Swaps the storage of |list_a| and |list_b|. The list references remain the
// same but the count, capacity, and underlying storage will be swapped. This
// can be used to treat lists as persistent stable references to dynamically
// mutated storage such as when emulating structs or dicts.
//
// WARNING: if a list is initialized in-place with iree_vm_list_initialize this
// will still perform the storage swap but may lead to unexpected issues if the
// lifetime of the storage is shorter than the lifetime of the newly-swapped
// list.
IREE_API_EXPORT void iree_vm_list_swap_storage(iree_vm_list_t* list_a,
                                               iree_vm_list_t* list_b);

// Copies |count| elements from |src_list| starting at |src_i| to |dst_list|
// starting at |dst_i|. The ranges specified must be valid in both lists.
//
// Supported list types:
//   any type -> variant list
//   variant list -> compatible element types only
//   same value type -> same value type
//   same ref type -> same ref type
IREE_API_EXPORT iree_status_t iree_vm_list_copy(iree_vm_list_t* src_list,
                                                iree_host_size_t src_i,
                                                iree_vm_list_t* dst_list,
                                                iree_host_size_t dst_i,
                                                iree_host_size_t count);

// Returns the value of the element at the given index.
// Note that the value type may vary from element to element in variant lists
// and callers should check the |out_value| type.
IREE_API_EXPORT iree_status_t iree_vm_list_get_value(
    const iree_vm_list_t* list, iree_host_size_t i, iree_vm_value_t* out_value);

// Returns the value of the element at the given index. If the specified
// |value_type| differs from the list storage type the value will be converted
// using the value type semantics (such as sign/zero extend, etc).
IREE_API_EXPORT iree_status_t iree_vm_list_get_value_as(
    const iree_vm_list_t* list, iree_host_size_t i,
    iree_vm_value_type_t value_type, iree_vm_value_t* out_value);

// Sets the value of the element at the given index. If the specified |value|
// type differs from the list storage type the value will be converted using the
// value type semantics (such as sign/zero extend, etc).
IREE_API_EXPORT iree_status_t iree_vm_list_set_value(
    iree_vm_list_t* list, iree_host_size_t i, const iree_vm_value_t* value);

// Pushes the value of the element to the end of the list.
// If the specified |value| type differs from the list storage type the value
// will be converted using the value type semantics (such as sign/zero extend,
// etc).
IREE_API_EXPORT iree_status_t
iree_vm_list_push_value(iree_vm_list_t* list, const iree_vm_value_t* value);

// Returns a dereferenced pointer to the given type if the element at the
// given index |i| matches the |type|. Returns NULL on error.
IREE_API_EXPORT void* iree_vm_list_get_ref_deref(const iree_vm_list_t* list,
                                                 iree_host_size_t i,
                                                 iree_vm_ref_type_t type);

// Returns the ref value of the element at the given index.
// The ref will not be retained and must be retained by the caller to extend
// its lifetime.
IREE_API_EXPORT iree_status_t iree_vm_list_get_ref_assign(
    const iree_vm_list_t* list, iree_host_size_t i, iree_vm_ref_t* out_value);

// Returns the ref value of the element at the given index.
// The ref will be retained and must be released by the caller.
IREE_API_EXPORT iree_status_t iree_vm_list_get_ref_retain(
    const iree_vm_list_t* list, iree_host_size_t i, iree_vm_ref_t* out_value);

// Sets the ref value of the element at the given index, retaining a reference
// in the list until the element is cleared or the list is disposed.
IREE_API_EXPORT iree_status_t iree_vm_list_set_ref_retain(
    iree_vm_list_t* list, iree_host_size_t i, const iree_vm_ref_t* value);

// Pushes the ref value of the element to the end of the list, retaining a
// reference in the list until the element is cleared or the list is disposed.
IREE_API_EXPORT iree_status_t
iree_vm_list_push_ref_retain(iree_vm_list_t* list, const iree_vm_ref_t* value);

// Sets the ref value of the element at the given index, moving ownership of the
// |value| reference to the list.
IREE_API_EXPORT iree_status_t iree_vm_list_set_ref_move(iree_vm_list_t* list,
                                                        iree_host_size_t i,
                                                        iree_vm_ref_t* value);

// Pushes the ref value of the element to the end of the list, moving ownership
// of the |value| reference to the list.
IREE_API_EXPORT iree_status_t iree_vm_list_push_ref_move(iree_vm_list_t* list,
                                                         iree_vm_ref_t* value);

// Pops the front ref value from the list and transfers ownership to the caller.
IREE_API_EXPORT iree_status_t
iree_vm_list_pop_front_ref_move(iree_vm_list_t* list, iree_vm_ref_t* out_value);

// Returns the value of the element at the given index. If the element contains
// a ref it will *not* be retained and the caller must retain it to extend its
// lifetime.
IREE_API_EXPORT iree_status_t
iree_vm_list_get_variant_assign(const iree_vm_list_t* list, iree_host_size_t i,
                                iree_vm_variant_t* out_variant);

// Returns the value of the element at the given index.
// If the variant is a ref then it will be retained.
IREE_API_EXPORT iree_status_t
iree_vm_list_get_variant_retain(const iree_vm_list_t* list, iree_host_size_t i,
                                iree_vm_variant_t* out_variant);

// Returns the value of the element at the given index.
// If the variant is a ref then it will be moved.
IREE_API_EXPORT iree_status_t
iree_vm_list_get_variant_move(const iree_vm_list_t* list, iree_host_size_t i,
                              iree_vm_variant_t* out_variant);

// Sets the value of the element at the given index. If the specified |value|
// type differs from the list storage type the value will be converted using the
// value type semantics (such as sign/zero extend, etc). If the variant is a ref
// then it will be retained.
IREE_API_EXPORT iree_status_t iree_vm_list_set_variant_retain(
    iree_vm_list_t* list, iree_host_size_t i, const iree_vm_variant_t* variant);

// Sets the value of the element at the given index. If the specified |value|
// type differs from the list storage type the value will be converted using the
// value type semantics (such as sign/zero extend, etc). If the variant is a ref
// then it will be moved.
IREE_API_EXPORT iree_status_t iree_vm_list_set_variant_move(
    iree_vm_list_t* list, iree_host_size_t i, iree_vm_variant_t* variant);

// Pushes the value of the element to the end of the list. If the specified
// |variant| type differs from the list storage type the value will be converted
// using the value type semantics (such as sign/zero extend, etc). If the
// variant is a ref then it will be retained.
IREE_API_EXPORT iree_status_t iree_vm_list_push_variant_retain(
    iree_vm_list_t* list, const iree_vm_variant_t* variant);

// Pushes the value of the element to the end of the list. If the specified
// |variant| type differs from the list storage type the value will be converted
// using the value type semantics (such as sign/zero extend, etc). If the
// variant is a ref then it will be moved.
IREE_API_EXPORT iree_status_t iree_vm_list_push_variant_move(
    iree_vm_list_t* list, iree_vm_variant_t* variant);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

IREE_VM_DECLARE_TYPE_ADAPTERS(iree_vm_list, iree_vm_list_t);

#endif  // IREE_VM_LIST_H_
