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

#define IREE_VM_REGISTER_HAL_C_TYPE(instance, type, name, destroy_fn,   \
                                    registration)                       \
  static const iree_vm_ref_type_descriptor_t registration##_storage = { \
      .type_name = IREE_SVL(name),                                      \
      .offsetof_counter = offsetof(iree_hal_resource_t, ref_count) /    \
                          IREE_VM_REF_COUNTER_ALIGNMENT,                \
      .destroy = (iree_vm_ref_destroy_t)destroy_fn,                     \
  };                                                                    \
  IREE_RETURN_IF_ERROR(iree_vm_instance_register_type(                  \
      instance, &registration##_storage, &registration));

static iree_status_t iree_hal_module_register_common_types(
    iree_vm_instance_t* instance) {
  IREE_VM_REGISTER_HAL_C_TYPE(instance, iree_hal_buffer_t, "hal.buffer",
                              iree_hal_buffer_recycle,
                              iree_hal_buffer_registration);
  IREE_VM_REGISTER_HAL_C_TYPE(instance, iree_hal_buffer_view_t,
                              "hal.buffer_view", iree_hal_buffer_view_destroy,
                              iree_hal_buffer_view_registration);
  return iree_ok_status();
}

static iree_status_t iree_hal_module_register_executable_types(
    iree_vm_instance_t* instance) {
  IREE_VM_REGISTER_HAL_C_TYPE(instance, iree_hal_executable_t, "hal.executable",
                              iree_hal_executable_destroy,
                              iree_hal_executable_registration);
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
  IREE_RETURN_IF_ERROR(iree_hal_module_register_common_types(instance));
  IREE_RETURN_IF_ERROR(iree_hal_module_register_executable_types(instance));

  IREE_VM_REGISTER_HAL_C_TYPE(instance, iree_hal_allocator_t, "hal.allocator",
                              iree_hal_allocator_destroy,
                              iree_hal_allocator_registration);
  IREE_VM_REGISTER_HAL_C_TYPE(instance, iree_hal_channel_t, "hal.channel",
                              iree_hal_channel_destroy,
                              iree_hal_channel_registration);
  IREE_VM_REGISTER_HAL_C_TYPE(
      instance, iree_hal_command_buffer_t, "hal.command_buffer",
      iree_hal_command_buffer_destroy, iree_hal_command_buffer_registration);
  IREE_VM_REGISTER_HAL_C_TYPE(instance, iree_hal_descriptor_set_layout_t,
                              "hal.descriptor_set_layout",
                              iree_hal_descriptor_set_layout_destroy,
                              iree_hal_descriptor_set_layout_registration);
  IREE_VM_REGISTER_HAL_C_TYPE(instance, iree_hal_device_t, "hal.device",
                              iree_hal_device_destroy,
                              iree_hal_device_registration);
  IREE_VM_REGISTER_HAL_C_TYPE(instance, iree_hal_event_t, "hal.event",
                              iree_hal_event_destroy,
                              iree_hal_event_registration);
  IREE_VM_REGISTER_HAL_C_TYPE(instance, iree_hal_fence_t, "hal.fence",
                              iree_hal_fence_destroy,
                              iree_hal_fence_registration);
  IREE_VM_REGISTER_HAL_C_TYPE(
      instance, iree_hal_pipeline_layout_t, "hal.pipeline_layout",
      iree_hal_pipeline_layout_destroy, iree_hal_pipeline_layout_registration);
  IREE_VM_REGISTER_HAL_C_TYPE(instance, iree_hal_semaphore_t, "hal.semaphore",
                              iree_hal_semaphore_destroy,
                              iree_hal_semaphore_registration);

  return iree_ok_status();
}

#define IREE_VM_RESOLVE_HAL_C_TYPE(instance, type, name, registration)      \
  registration =                                                            \
      iree_vm_instance_lookup_type(instance, iree_make_cstring_view(name)); \
  if (!registration) {                                                      \
    return iree_make_status(IREE_STATUS_INTERNAL,                           \
                            "VM type `" name                                \
                            "` not registered with the instance");          \
  }

static iree_status_t iree_hal_module_resolve_common_types(
    iree_vm_instance_t* instance) {
  IREE_RETURN_IF_ERROR(iree_vm_resolve_builtin_types(instance));

  IREE_VM_RESOLVE_HAL_C_TYPE(instance, iree_hal_buffer_t, "hal.buffer",
                             iree_hal_buffer_registration);
  IREE_VM_RESOLVE_HAL_C_TYPE(instance, iree_hal_buffer_view_t,
                             "hal.buffer_view",
                             iree_hal_buffer_view_registration);

  return iree_ok_status();
}

static iree_status_t iree_hal_module_resolve_executable_types(
    iree_vm_instance_t* instance) {
  IREE_VM_RESOLVE_HAL_C_TYPE(instance, iree_hal_executable_t, "hal.executable",
                             iree_hal_executable_registration);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_hal_module_resolve_inline_types(iree_vm_instance_t* instance) {
  return iree_hal_module_resolve_common_types(instance);
}

IREE_API_EXPORT iree_status_t
iree_hal_module_resolve_loader_types(iree_vm_instance_t* instance) {
  IREE_RETURN_IF_ERROR(iree_hal_module_resolve_common_types(instance));
  IREE_RETURN_IF_ERROR(iree_hal_module_resolve_executable_types(instance));
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_hal_module_resolve_all_types(iree_vm_instance_t* instance) {
  IREE_RETURN_IF_ERROR(iree_hal_module_resolve_common_types(instance));
  IREE_RETURN_IF_ERROR(iree_hal_module_resolve_executable_types(instance));

  IREE_VM_RESOLVE_HAL_C_TYPE(instance, iree_hal_allocator_t, "hal.allocator",
                             iree_hal_allocator_registration);
  IREE_VM_RESOLVE_HAL_C_TYPE(instance, iree_hal_channel_t, "hal.channel",
                             iree_hal_channel_registration);
  IREE_VM_RESOLVE_HAL_C_TYPE(instance, iree_hal_command_buffer_t,
                             "hal.command_buffer",
                             iree_hal_command_buffer_registration);
  IREE_VM_RESOLVE_HAL_C_TYPE(instance, iree_hal_descriptor_set_layout_t,
                             "hal.descriptor_set_layout",
                             iree_hal_descriptor_set_layout_registration);
  IREE_VM_RESOLVE_HAL_C_TYPE(instance, iree_hal_device_t, "hal.device",
                             iree_hal_device_registration);
  IREE_VM_RESOLVE_HAL_C_TYPE(instance, iree_hal_event_t, "hal.event",
                             iree_hal_event_registration);
  IREE_VM_RESOLVE_HAL_C_TYPE(instance, iree_hal_fence_t, "hal.fence",
                             iree_hal_fence_registration);
  IREE_VM_RESOLVE_HAL_C_TYPE(instance, iree_hal_pipeline_layout_t,
                             "hal.pipeline_layout",
                             iree_hal_pipeline_layout_registration);
  IREE_VM_RESOLVE_HAL_C_TYPE(instance, iree_hal_semaphore_t, "hal.semaphore",
                             iree_hal_semaphore_registration);

  return iree_ok_status();
}

//===--------------------------------------------------------------------===//
// Utilities
//===--------------------------------------------------------------------===//

IREE_API_EXPORT iree_hal_buffer_t* iree_vm_list_get_buffer_assign(
    const iree_vm_list_t* list, iree_host_size_t i) {
  return (iree_hal_buffer_t*)iree_vm_list_get_ref_deref(list, i,
                                                        iree_hal_buffer_type());
}

IREE_API_EXPORT iree_hal_buffer_t* iree_vm_list_get_buffer_retain(
    const iree_vm_list_t* list, iree_host_size_t i) {
  iree_hal_buffer_t* value = iree_vm_list_get_buffer_assign(list, i);
  iree_hal_buffer_retain(value);
  return value;
}

IREE_API_EXPORT iree_status_t iree_vm_list_set_buffer_retain(
    iree_vm_list_t* list, iree_host_size_t i, iree_hal_buffer_t* value) {
  iree_vm_ref_t value_ref;
  IREE_RETURN_IF_ERROR(
      iree_vm_ref_wrap_assign(value, iree_hal_buffer_type(), &value_ref));
  return iree_vm_list_set_ref_retain(list, i, &value_ref);
}

IREE_API_EXPORT iree_hal_buffer_view_t* iree_vm_list_get_buffer_view_assign(
    const iree_vm_list_t* list, iree_host_size_t i) {
  return (iree_hal_buffer_view_t*)iree_vm_list_get_ref_deref(
      list, i, iree_hal_buffer_view_type());
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
  IREE_RETURN_IF_ERROR(
      iree_vm_ref_wrap_assign(value, iree_hal_buffer_view_type(), &value_ref));
  return iree_vm_list_set_ref_retain(list, i, &value_ref);
}

IREE_API_EXPORT iree_hal_fence_t* iree_vm_list_get_fence_assign(
    const iree_vm_list_t* list, iree_host_size_t i) {
  return (iree_hal_fence_t*)iree_vm_list_get_ref_deref(list, i,
                                                       iree_hal_fence_type());
}

IREE_API_EXPORT iree_hal_fence_t* iree_vm_list_get_fence_retain(
    const iree_vm_list_t* list, iree_host_size_t i) {
  iree_hal_fence_t* value = iree_vm_list_get_fence_assign(list, i);
  iree_hal_fence_retain(value);
  return value;
}

IREE_API_EXPORT iree_status_t iree_vm_list_set_fence_retain(
    iree_vm_list_t* list, iree_host_size_t i, iree_hal_fence_t* value) {
  iree_vm_ref_t value_ref;
  IREE_RETURN_IF_ERROR(
      iree_vm_ref_wrap_assign(value, iree_hal_fence_type(), &value_ref));
  return iree_vm_list_set_ref_retain(list, i, &value_ref);
}
