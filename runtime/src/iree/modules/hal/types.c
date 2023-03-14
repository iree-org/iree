// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/modules/hal/types.h"

//===----------------------------------------------------------------------===//
// Type wrappers
//===----------------------------------------------------------------------===//

IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_allocator, iree_hal_allocator_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_buffer, iree_hal_buffer_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_buffer_view, iree_hal_buffer_view_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_channel, iree_hal_channel_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_command_buffer,
                             iree_hal_command_buffer_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_descriptor_set_layout,
                             iree_hal_descriptor_set_layout_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_device, iree_hal_device_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_event, iree_hal_event_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_executable, iree_hal_executable_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_pipeline_layout,
                             iree_hal_pipeline_layout_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_fence, iree_hal_fence_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_semaphore, iree_hal_semaphore_t);

//===----------------------------------------------------------------------===//
// Type registration
//===----------------------------------------------------------------------===//

#define IREE_VM_REGISTER_HAL_C_TYPE(type, name, destroy_fn, descriptor)   \
  descriptor.type_name = iree_make_cstring_view(name);                    \
  descriptor.offsetof_counter = offsetof(iree_hal_resource_t, ref_count); \
  descriptor.destroy = (iree_vm_ref_destroy_t)destroy_fn;                 \
  IREE_RETURN_IF_ERROR(iree_vm_ref_register_type(&descriptor));

static iree_status_t iree_hal_module_register_common_types(
    iree_vm_instance_t* instance) {
  static bool has_registered = false;
  if (has_registered) return iree_ok_status();

  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_buffer_t, "hal.buffer",
                              iree_hal_buffer_recycle,
                              iree_hal_buffer_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_buffer_view_t, "hal.buffer_view",
                              iree_hal_buffer_view_destroy,
                              iree_hal_buffer_view_descriptor);

  has_registered = true;
  return iree_ok_status();
}

static iree_status_t iree_hal_module_register_executable_types(
    iree_vm_instance_t* instance) {
  static bool has_registered = false;
  if (has_registered) return iree_ok_status();

  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_executable_t, "hal.executable",
                              iree_hal_executable_destroy,
                              iree_hal_executable_descriptor);

  has_registered = true;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_hal_module_register_inline_types(iree_vm_instance_t* instance) {
  return iree_hal_module_register_common_types(instance);
}

IREE_API_EXPORT iree_status_t
iree_hal_module_register_loader_types(iree_vm_instance_t* instance) {
  IREE_RETURN_IF_ERROR(iree_hal_module_register_common_types(instance));
  IREE_RETURN_IF_ERROR(iree_hal_module_register_executable_types(instance));
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_hal_module_register_all_types(iree_vm_instance_t* instance) {
  static bool has_registered = false;
  if (has_registered) return iree_ok_status();

  IREE_RETURN_IF_ERROR(iree_hal_module_register_common_types(instance));
  IREE_RETURN_IF_ERROR(iree_hal_module_register_executable_types(instance));

  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_allocator_t, "hal.allocator",
                              iree_hal_allocator_destroy,
                              iree_hal_allocator_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_channel_t, "hal.channel",
                              iree_hal_channel_destroy,
                              iree_hal_channel_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_command_buffer_t, "hal.command_buffer",
                              iree_hal_command_buffer_destroy,
                              iree_hal_command_buffer_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_descriptor_set_layout_t,
                              "hal.descriptor_set_layout",
                              iree_hal_descriptor_set_layout_destroy,
                              iree_hal_descriptor_set_layout_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_device_t, "hal.device",
                              iree_hal_device_destroy,
                              iree_hal_device_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_event_t, "hal.event",
                              iree_hal_event_destroy,
                              iree_hal_event_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_fence_t, "hal.fence",
                              iree_hal_fence_destroy,
                              iree_hal_fence_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_pipeline_layout_t, "hal.pipeline_layout",
                              iree_hal_pipeline_layout_destroy,
                              iree_hal_pipeline_layout_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_semaphore_t, "hal.semaphore",
                              iree_hal_semaphore_destroy,
                              iree_hal_semaphore_descriptor);

  has_registered = true;
  return iree_ok_status();
}

//===--------------------------------------------------------------------===//
// Utilities
//===--------------------------------------------------------------------===//

IREE_API_EXPORT iree_hal_buffer_view_t* iree_vm_list_get_buffer_view_assign(
    const iree_vm_list_t* list, iree_host_size_t i) {
  return (iree_hal_buffer_view_t*)iree_vm_list_get_ref_deref(
      list, i, &iree_hal_buffer_view_descriptor);
}

IREE_API_EXPORT iree_hal_buffer_view_t* iree_vm_list_get_buffer_view_retain(
    const iree_vm_list_t* list, iree_host_size_t i) {
  iree_hal_buffer_view_t* value = iree_vm_list_get_buffer_view_assign(list, i);
  iree_hal_buffer_view_retain(value);
  return value;
}

IREE_API_EXPORT iree_status_t iree_vm_list_set_buffer_view_retain(
    iree_vm_list_t* list, iree_host_size_t i, iree_hal_buffer_view_t* value) {
  iree_vm_ref_t value_ref;
  IREE_RETURN_IF_ERROR(iree_vm_ref_wrap_assign(
      value, iree_hal_buffer_view_type_id(), &value_ref));
  return iree_vm_list_set_ref_retain(list, i, &value_ref);
}
