// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/modules/resolver.h"

#include "iree/base/tracing.h"

#if defined(IREE_HAVE_VMVX_MODULE)
#include "iree/modules/vmvx/module.h"
#endif  // IREE_HAVE_VMVX_MODULE

#if defined(IREE_HAVE_EXTERNAL_TOOLING_MODULES)
// Defined in the generated registry_external.c file:
extern iree_status_t iree_tooling_register_external_module_types(
    iree_vm_instance_t* instance);
extern iree_status_t iree_tooling_try_resolve_external_module_dependency(
    iree_vm_instance_t* instance, const iree_vm_module_dependency_t* dependency,
    iree_allocator_t host_allocator, iree_vm_module_t** out_module);
#else
static iree_status_t iree_tooling_register_external_module_types(
    iree_vm_instance_t* instance) {
  return iree_ok_status();
}
static iree_status_t iree_tooling_try_resolve_external_module_dependency(
    iree_vm_instance_t* instance, const iree_vm_module_dependency_t* dependency,
    iree_allocator_t host_allocator, iree_vm_module_t** out_module) {
  *out_module = NULL;
  return iree_ok_status();
}
#endif  // IREE_HAVE_EXTERNAL_TOOLING_MODULES

iree_status_t iree_tooling_register_all_module_types(
    iree_vm_instance_t* instance) {
  IREE_RETURN_IF_ERROR(iree_tooling_register_external_module_types(instance));
  return iree_ok_status();
}

iree_status_t iree_tooling_resolve_module_dependency(
    iree_vm_instance_t* instance, const iree_vm_module_dependency_t* dependency,
    iree_allocator_t host_allocator, iree_vm_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(dependency);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, dependency->name.data, dependency->name.size);

  iree_vm_module_t* module = NULL;
  if (iree_string_view_equal(dependency->name, IREE_SV("vmvx"))) {
    // VMVX module used on the host side for the inline HAL.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_vmvx_module_create(instance, host_allocator, &module));
  } else {
    // Try to resolve the module from externally-defined modules.
    // If the module is not found this will succeed but module will be NULL.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_tooling_try_resolve_external_module_dependency(
                instance, dependency, host_allocator, &module));
  }

  IREE_TRACE_ZONE_END(z0);
  if (!module && iree_all_bits_set(dependency->flags,
                                   IREE_VM_MODULE_DEPENDENCY_FLAG_REQUIRED)) {
    // Required but not found; fail.
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "required module '%.*s' not available in the build",
                            (int)dependency->name.size, dependency->name.data);
  }
  *out_module = module;
  return iree_ok_status();
}
