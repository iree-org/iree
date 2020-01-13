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

#ifndef IREE_MODULES_HAL_HAL_MODULE_H_
#define IREE_MODULES_HAL_HAL_MODULE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

IREE_VM_DECLARE_TYPE_ADAPTERS(iree_hal_allocator, iree_hal_allocator_t);
IREE_VM_DECLARE_TYPE_ADAPTERS(iree_hal_buffer, iree_hal_buffer_t);
IREE_VM_DECLARE_TYPE_ADAPTERS(iree_hal_command_buffer,
                              iree_hal_command_buffer_t);
IREE_VM_DECLARE_TYPE_ADAPTERS(iree_hal_descriptor_set,
                              iree_hal_descriptor_set_t);
IREE_VM_DECLARE_TYPE_ADAPTERS(iree_hal_descriptor_set_layout,
                              iree_hal_descriptor_set_layout_t);
IREE_VM_DECLARE_TYPE_ADAPTERS(iree_hal_device, iree_hal_device_t);
IREE_VM_DECLARE_TYPE_ADAPTERS(iree_hal_executable, iree_hal_executable_t);

// Registers the custom types used by the HAL module.
// WARNING: not thread-safe; call at startup before using.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_module_register_types();

// Creates the HAL module initialized to use a specific |device|.
// Each context using this module will share the device and have compatible
// allocations.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_module_create(iree_hal_device_t* device, iree_allocator_t allocator,
                       iree_vm_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MODULES_HAL_HAL_MODULE_H_
