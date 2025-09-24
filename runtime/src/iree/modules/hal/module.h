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
#include "iree/modules/hal/debugging.h"
#include "iree/modules/hal/types.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_module_device_policy_t
//===----------------------------------------------------------------------===//

typedef uint64_t iree_hal_module_device_allocator_select_flags_t;
typedef struct iree_hal_device_queue_affinity_pair_t {
  iree_hal_device_t* device;
  iree_hal_queue_affinity_t queue_affinity;
} iree_hal_device_queue_affinity_pair_t;
typedef iree_status_t(
    IREE_API_PTR* iree_hal_module_device_allocator_select_callback_fn_t)(
    void* user_data, iree_host_size_t device_count,
    const iree_hal_device_queue_affinity_pair_t* devices,
    iree_hal_memory_type_t memory_types, iree_hal_buffer_usage_t buffer_usage,
    iree_hal_module_device_allocator_select_flags_t flags,
    iree_host_size_t* out_selection);

typedef struct iree_hal_module_device_allocator_select_callback_t {
  iree_hal_module_device_allocator_select_callback_fn_t fn;
  void* user_data;
} iree_hal_module_device_allocator_select_callback_t;

typedef struct iree_hal_module_device_policy_t {
  iree_hal_module_device_allocator_select_callback_t allocator_select;
} iree_hal_module_device_policy_t;

// Returns a default device policy that chooses defaults deterministically.
// For queries that must select amongst multiple options the first is always
// chosen.
IREE_API_EXPORT iree_hal_module_device_policy_t
iree_hal_module_device_policy_default(void);

//===----------------------------------------------------------------------===//
// iree_hal_module_t
//===----------------------------------------------------------------------===//

enum iree_hal_module_flag_bits_t {
  IREE_HAL_MODULE_FLAG_NONE = 0u,

  // Forces HAL methods to block instead of yielding as a coroutine.
  IREE_HAL_MODULE_FLAG_SYNCHRONOUS = 1u << 0,
};
typedef uint32_t iree_hal_module_flags_t;

// Creates the HAL module initialized to use one or more |devices|.
// Each context using this module will share the devices and have compatible
// allocations.
IREE_API_EXPORT iree_status_t iree_hal_module_create(
    iree_vm_instance_t* instance, iree_hal_module_device_policy_t device_policy,
    iree_host_size_t device_count, iree_hal_device_t** devices,
    iree_hal_module_flags_t flags, iree_hal_module_debug_sink_t debug_sink,
    iree_allocator_t host_allocator, iree_vm_module_t** out_module);

// Returns the total number of available devices registered with the HAL module.
IREE_API_EXPORT iree_host_size_t
iree_hal_module_state_device_count(iree_vm_module_state_t* module_state);

// Returns the device at |index| currently in use by the HAL module.
// Returns NULL if no device has been initialized yet or the index is out of
// bounds.
IREE_API_EXPORT iree_hal_device_t* iree_hal_module_state_device_get(
    iree_vm_module_state_t* module_state, iree_host_size_t index);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MODULES_HAL_MODULE_H_
