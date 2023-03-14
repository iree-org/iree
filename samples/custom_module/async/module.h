// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_SAMPLES_CUSTOM_MODULE_TENSOR_ASYNC_MODULE_H_
#define IREE_SAMPLES_CUSTOM_MODULE_TENSOR_ASYNC_MODULE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a native custom module that can be reused in multiple contexts.
// The module itself may hold state that can be shared by all instantiated
// copies but it will require the module to provide synchronization; usually
// it's safer to just treat the module as immutable and keep state within the
// instantiated module states instead.
iree_status_t iree_custom_module_async_create(iree_vm_instance_t* instance,
                                              iree_hal_device_t* device,
                                              iree_allocator_t host_allocator,
                                              iree_vm_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_SAMPLES_CUSTOM_MODULE_TENSOR_ASYNC_MODULE_H_
