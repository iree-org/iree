// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/runtime/instance.h"

#include <stddef.h>
#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/init.h"
#include "iree/modules/hal/module.h"
#include "iree/vm/api.h"

//===----------------------------------------------------------------------===//
// iree_runtime_instance_options_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_runtime_instance_options_initialize(
    iree_runtime_instance_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
}

IREE_API_EXPORT void iree_runtime_instance_options_use_all_available_drivers(
    iree_runtime_instance_options_t* options) {
  options->driver_registry = iree_hal_driver_registry_default();
  // TODO(benvanik): remove status result from this; it can't (meaningfully)
  // fail and is just extra bookkeeping.
  iree_status_ignore(
      iree_hal_register_all_available_drivers(options->driver_registry));
}

//===----------------------------------------------------------------------===//
// iree_runtime_instance_t
//===----------------------------------------------------------------------===//

struct iree_runtime_instance_t {
  iree_atomic_ref_count_t ref_count;

  // Allocator used to allocate the instance and all of its resources.
  iree_allocator_t host_allocator;

  // An optional driver registry used to enumerate and create HAL devices.
  iree_hal_driver_registry_t* driver_registry;

  // TODO(#5724): we should have a device cache here so that multiple sessions
  // can find the same devices. This may mean a new HAL type like
  // iree_hal_device_pool_t to prevent too much coupling and make weak
  // references easier.

  // VM instance shared across all sessions.
  iree_vm_instance_t* vm_instance;
};

IREE_API_EXPORT iree_status_t iree_runtime_instance_create(
    const iree_runtime_instance_options_t* options,
    iree_allocator_t host_allocator, iree_runtime_instance_t** out_instance) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_instance);
  *out_instance = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate the instance state.
  iree_runtime_instance_t* instance = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*instance),
                                (void**)&instance));
  instance->host_allocator = host_allocator;
  iree_atomic_ref_count_init(&instance->ref_count);

  instance->driver_registry = options->driver_registry;
  // TODO(benvanik): driver registry ref counting.

  iree_status_t status = iree_vm_instance_create(
      IREE_VM_TYPE_CAPACITY_DEFAULT, host_allocator, &instance->vm_instance);
  if (iree_status_is_ok(status)) {
    status = iree_hal_module_register_all_types(instance->vm_instance);
  }

  if (iree_status_is_ok(status)) {
    *out_instance = instance;
  } else {
    iree_runtime_instance_release(instance);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_runtime_instance_destroy(iree_runtime_instance_t* instance) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_vm_instance_release(instance->vm_instance);
  iree_allocator_free(instance->host_allocator, instance);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT void iree_runtime_instance_retain(
    iree_runtime_instance_t* instance) {
  if (instance) {
    iree_atomic_ref_count_inc(&instance->ref_count);
  }
}

IREE_API_EXPORT void iree_runtime_instance_release(
    iree_runtime_instance_t* instance) {
  if (instance && iree_atomic_ref_count_dec(&instance->ref_count) == 1) {
    iree_runtime_instance_destroy(instance);
  }
}

IREE_API_EXPORT iree_allocator_t
iree_runtime_instance_host_allocator(const iree_runtime_instance_t* instance) {
  IREE_ASSERT_ARGUMENT(instance);
  return instance->host_allocator;
}

IREE_API_EXPORT iree_vm_instance_t* iree_runtime_instance_vm_instance(
    const iree_runtime_instance_t* instance) {
  IREE_ASSERT_ARGUMENT(instance);
  return instance->vm_instance;
}

IREE_API_EXPORT iree_hal_driver_registry_t*
iree_runtime_instance_driver_registry(const iree_runtime_instance_t* instance) {
  IREE_ASSERT_ARGUMENT(instance);
  return instance->driver_registry;
}

IREE_API_EXPORT iree_status_t iree_runtime_instance_try_create_default_device(
    iree_runtime_instance_t* instance, iree_string_view_t driver_name,
    iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, driver_name.data, driver_name.size);

  // This is only supported when we have a driver registry we can use to create
  // the drivers.
  iree_hal_driver_registry_t* driver_registry =
      iree_runtime_instance_driver_registry(instance);
  if (!driver_registry) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "instance was created without a driver registry "
                            "and cannot perform enumeration");
  }

  // Create a driver with the given name (if one exists).
  iree_allocator_t host_allocator =
      iree_runtime_instance_host_allocator(instance);
  iree_hal_driver_t* driver = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_driver_registry_try_create(driver_registry, driver_name,
                                              host_allocator, &driver));

  // Create the default device on that driver.
  iree_status_t status =
      iree_hal_driver_create_default_device(driver, host_allocator, out_device);

  iree_hal_driver_release(driver);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
