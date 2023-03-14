// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MODULES_HAL_MODULE_H_
#define IREE_MODULES_HAL_MODULE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

enum iree_hal_module_flag_bits_t {
  IREE_HAL_MODULE_FLAG_NONE = 0u,

  // Forces HAL methods to block instead of yielding as a coroutine.
  IREE_HAL_MODULE_FLAG_SYNCHRONOUS = 1u << 0,
};
typedef uint32_t iree_hal_module_flags_t;

// Creates the HAL module initialized to use a specific |device|.
// Each context using this module will share the device and have compatible
// allocations.
IREE_API_EXPORT iree_status_t iree_hal_module_create(
    iree_vm_instance_t* instance, iree_hal_device_t* device,
    iree_hal_module_flags_t flags, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module);

// Returns the device currently in use by the HAL module.
// Returns NULL if no device has been initialized yet.
IREE_API_EXPORT iree_hal_device_t* iree_hal_module_state_device(
    iree_vm_module_state_t* module_state);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MODULES_HAL_MODULE_H_
