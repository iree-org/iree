// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_SAMPLES_CUSTOM_MODULE_BASIC_MODULE_H_
#define IREE_SAMPLES_CUSTOM_MODULE_BASIC_MODULE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/vm/api.h"

// A non-NUL-terminated string.
typedef struct iree_custom_string_t iree_custom_string_t;
IREE_VM_DECLARE_TYPE_ADAPTERS(iree_custom_string, iree_custom_string_t);

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a new !custom.string object with a copy of the given |value|.
// Applications could use this and any other methods we wanted to expose to
// interop with the loaded VM modules - such as passing in/out the objects.
// We don't need this for the demo but creating the custom object, appending it
// to the invocation input list, and then consuming it in the compiled module
// is straightforward.
iree_status_t iree_custom_string_create(iree_string_view_t value,
                                        iree_allocator_t allocator,
                                        iree_custom_string_t** out_string);

// Registers types provided by the custom module.
// Not required to be called unless trying to create types from the module
// before creating the module (rare).
iree_status_t iree_custom_module_basic_register_types(
    iree_vm_instance_t* instance);

// Creates a native custom module that can be reused in multiple contexts.
// The module itself may hold state that can be shared by all instantiated
// copies but it will require the module to provide synchronization; usually
// it's safer to just treat the module as immutable and keep state within the
// instantiated module states instead.
iree_status_t iree_custom_module_basic_create(iree_vm_instance_t* instance,
                                              iree_allocator_t allocator,
                                              iree_vm_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_SAMPLES_CUSTOM_MODULE_BASIC_MODULE_H_
