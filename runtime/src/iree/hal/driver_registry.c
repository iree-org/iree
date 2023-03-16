// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver_registry.h"

#include <stddef.h>
#include <string.h>

#include "iree/base/internal/call_once.h"
#include "iree/base/internal/path.h"
#include "iree/base/internal/synchronization.h"
#include "iree/base/tracing.h"

//===----------------------------------------------------------------------===//
// iree_hal_driver_registry_t
//===----------------------------------------------------------------------===//

// ~N factories is enough for anyone, right?
// But really this is here to prevent the need for dynamically allocated memory.
// Because it's an implementation detail it's easy to grow in the future if we
// want to support additional factories.
//
// An alternative would be to keep factories in an intrusive list - that way
// there is no storage beyond the factory itself. This is less ideal as it would
// force all factory storage to be in writeable memory and limit the ability for
// the same factory to be registered with multiple registries (useful when
// isolating/sandboxing/multi-versioning).
#define IREE_HAL_MAX_DRIVER_FACTORY_COUNT 16

struct iree_hal_driver_registry_t {
  iree_allocator_t host_allocator;
  iree_slim_mutex_t mutex;

  // Factories in registration order. As factories are unregistered the list is
  // shifted to be kept dense.
  iree_host_size_t factory_count;
  const iree_hal_driver_factory_t* factories[IREE_HAL_MAX_DRIVER_FACTORY_COUNT];
};

static iree_hal_driver_registry_t iree_hal_driver_registry_default_;
static iree_once_flag iree_hal_driver_registry_default_flag_ =
    IREE_ONCE_FLAG_INIT;
static void iree_hal_driver_registry_default_initialize(void) {
  memset(&iree_hal_driver_registry_default_, 0,
         sizeof(iree_hal_driver_registry_default_));
  iree_slim_mutex_initialize(&iree_hal_driver_registry_default_.mutex);
}

IREE_API_EXPORT iree_hal_driver_registry_t* iree_hal_driver_registry_default(
    void) {
  iree_call_once(&iree_hal_driver_registry_default_flag_,
                 iree_hal_driver_registry_default_initialize);
  return &iree_hal_driver_registry_default_;
}

IREE_API_EXPORT iree_status_t
iree_hal_driver_registry_allocate(iree_allocator_t host_allocator,
                                  iree_hal_driver_registry_t** out_registry) {
  IREE_ASSERT_ARGUMENT(out_registry);
  *out_registry = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_driver_registry_t* registry = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*registry),
                                (void**)&registry));
  registry->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&registry->mutex);

  *out_registry = registry;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT void iree_hal_driver_registry_free(
    iree_hal_driver_registry_t* registry) {
  if (!registry) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t host_allocator = registry->host_allocator;

  iree_slim_mutex_deinitialize(&registry->mutex);
  iree_allocator_free(host_allocator, registry);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t iree_hal_driver_registry_register_factory(
    iree_hal_driver_registry_t* registry,
    const iree_hal_driver_factory_t* factory) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_slim_mutex_lock(&registry->mutex);

  // Fail if already present; not because having it in there would harm anything
  // but because we can't then balance with unregisters if we were to skip it
  // when present and want to keep the list small and not have callers fill it
  // with tons of duplicate entries.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < registry->factory_count; ++i) {
    if (registry->factories[i] == factory) {
      status = iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                                "factory has already been registered");
      break;
    }
  }

  // Note that we check the capacity limit *after* checking for dupes so that
  // callers will find issues with duplicate registrations easier. Otherwise,
  // they'd just get a RESOURCE_EXHAUSTED and think there were too many unique
  // factories registered already.
  if (iree_status_is_ok(status) &&
      registry->factory_count + 1 >= IREE_ARRAYSIZE(registry->factories)) {
    status = iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "the maximum number of factories (%zu) have been registered",
        IREE_ARRAYSIZE(registry->factories));
  }

  if (iree_status_is_ok(status)) {
    registry->factories[registry->factory_count++] = factory;
  }

  iree_slim_mutex_unlock(&registry->mutex);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_driver_registry_unregister_factory(
    iree_hal_driver_registry_t* registry,
    const iree_hal_driver_factory_t* factory) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_slim_mutex_lock(&registry->mutex);

  iree_status_t status = iree_ok_status();
  iree_host_size_t index = -1;
  for (iree_host_size_t i = 0; i < registry->factory_count; ++i) {
    if (registry->factories[i] != factory) continue;
    index = i;
    break;
  }
  if (index == -1) {
    status =
        iree_make_status(IREE_STATUS_NOT_FOUND,
                         "factory to remove is not registered at this time");
  }

  if (iree_status_is_ok(status)) {
    // Compact list. Note that registration order is preserved.
    memmove((void*)&registry->factories[index], &registry->factories[index + 1],
            registry->factory_count - index - 1);
    registry->factories[--registry->factory_count] = NULL;
  }

  iree_slim_mutex_unlock(&registry->mutex);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Computes the total byte size required to store driver info strings.
static iree_host_size_t iree_hal_driver_info_compute_storage_size(
    const iree_hal_driver_info_t* driver_info) {
  iree_host_size_t storage_size = 0;
  storage_size += driver_info->driver_name.size;
  storage_size += driver_info->full_name.size;
  return storage_size;
}

// Copies |source_driver_info| into |target_driver_info| using |string_storage|
// for the nested strings. Returns the total number of bytes added to
// string_storage.
static iree_host_size_t iree_hal_driver_info_copy(
    const iree_hal_driver_info_t* source_driver_info,
    iree_hal_driver_info_t* target_driver_info, char* string_storage) {
  // Copy everything by default (primitive fields, etc).
  memcpy(target_driver_info, source_driver_info, sizeof(*target_driver_info));

  // Copy in each string field to the string storage and set the ptr.
  iree_host_size_t storage_size = 0;
  storage_size += iree_string_view_append_to_buffer(
      source_driver_info->driver_name, &target_driver_info->driver_name,
      string_storage + storage_size);
  storage_size += iree_string_view_append_to_buffer(
      source_driver_info->full_name, &target_driver_info->full_name,
      string_storage + storage_size);
  return storage_size;
}

IREE_API_EXPORT iree_status_t iree_hal_driver_registry_enumerate(
    iree_hal_driver_registry_t* registry, iree_allocator_t host_allocator,
    iree_host_size_t* out_driver_info_count,
    iree_hal_driver_info_t** out_driver_infos) {
  IREE_ASSERT_ARGUMENT(registry);
  IREE_ASSERT_ARGUMENT(out_driver_info_count);
  IREE_ASSERT_ARGUMENT(out_driver_infos);
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_driver_info_count = 0;
  *out_driver_infos = NULL;

  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&registry->mutex);

  // Enumerate each factory and figure out how much memory we need to fully
  // store all data we need to clone.
  iree_host_size_t total_driver_info_count = 0;
  iree_host_size_t total_storage_size = 0;
  for (iree_host_size_t i = 0; i < registry->factory_count; ++i) {
    const iree_hal_driver_factory_t* factory = registry->factories[i];
    const iree_hal_driver_info_t* driver_infos = NULL;
    iree_host_size_t driver_info_count = 0;
    status =
        factory->enumerate(factory->self, &driver_info_count, &driver_infos);
    if (!iree_status_is_ok(status)) break;
    total_driver_info_count += driver_info_count;
    for (iree_host_size_t j = 0; j < driver_info_count; j++) {
      total_storage_size +=
          iree_hal_driver_info_compute_storage_size(&driver_infos[j]);
    }
  }

  // Allocate the required memory for both the driver infos and the string
  // storage in a single block.
  iree_host_size_t total_driver_infos_size =
      total_driver_info_count * sizeof(iree_hal_driver_info_t);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator,
                                   total_driver_infos_size + total_storage_size,
                                   (void**)out_driver_infos);
  }

  // Write driver info and associated nested resources to the output. We have
  // to enumerate again but enumeration is expected to be immutable for a given
  // registration and we hold the lock so we're safe.
  if (iree_status_is_ok(status)) {
    iree_hal_driver_info_t* driver_info_storage_ptr = *out_driver_infos;
    char* string_storage_ptr =
        (char*)(*out_driver_infos) + total_driver_infos_size;
    for (iree_host_size_t i = 0; i < registry->factory_count; ++i) {
      const iree_hal_driver_factory_t* factory = registry->factories[i];
      const iree_hal_driver_info_t* driver_infos = NULL;
      iree_host_size_t driver_info_count = 0;
      status =
          factory->enumerate(factory->self, &driver_info_count, &driver_infos);
      if (!iree_status_is_ok(status)) break;
      for (iree_host_size_t j = 0; j < driver_info_count; j++) {
        string_storage_ptr += iree_hal_driver_info_copy(
            &driver_infos[j], driver_info_storage_ptr, string_storage_ptr);
        ++driver_info_storage_ptr;
      }
    }
    *out_driver_info_count = total_driver_info_count;
  }

  iree_slim_mutex_unlock(&registry->mutex);

  // Cleanup memory if we failed.
  if (!iree_status_is_ok(status) && *out_driver_infos) {
    iree_allocator_free(host_allocator, *out_driver_infos);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_driver_registry_try_create(
    iree_hal_driver_registry_t* registry, iree_string_view_t driver_name,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(registry);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, driver_name.data, driver_name.size);

  *out_driver = NULL;

  // NOTE: we hold the lock the entire time here so that we can avoid
  // allocations and avoid spurious failures by outside mutation of the
  // registry.
  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&registry->mutex);

  // Enumerate each factory and scan for the requested driver.
  // NOTE: we scan in reverse so that we prefer the first hit in the most
  // recently registered factory.
  const iree_hal_driver_factory_t* hit_factory = NULL;
  for (iree_host_size_t i = 0; i < registry->factory_count; ++i) {
    // Reach inside and grab the internal factory data structures.
    const iree_hal_driver_factory_t* factory =
        registry->factories[registry->factory_count - i - 1];
    const iree_hal_driver_info_t* driver_infos = NULL;
    iree_host_size_t driver_info_count = 0;
    status =
        factory->enumerate(factory->self, &driver_info_count, &driver_infos);
    if (!iree_status_is_ok(status)) break;

    // Scan for the specific driver by name.
    // NOTE: we scan in reverse here too so multiple drivers with the same name
    // from the same factory prefer the later drivers in the list.
    for (iree_host_size_t j = 0; j < driver_info_count; j++) {
      const iree_hal_driver_info_t* driver_info =
          &driver_infos[driver_info_count - j - 1];
      if (iree_string_view_equal(driver_name, driver_info->driver_name)) {
        hit_factory = factory;
        break;
      }
    }
    // Since we are scanning in reverse we stop searching when we find the first
    // hit (aka the most recently added driver).
    if (hit_factory != NULL) break;
  }

  // If we found a driver during the scan try to create it now.
  // This may block the caller (with the lock held!), and may fail if for
  // example a delay-loaded driver cannot be created even if it was enumerated.
  if (hit_factory != NULL) {
    status = hit_factory->try_create(hit_factory->self, driver_name,
                                     host_allocator, out_driver);
  } else {
    status =
        iree_make_status(IREE_STATUS_NOT_FOUND, "no driver '%.*s' registered",
                         (int)driver_name.size, driver_name.data);
  }

  iree_slim_mutex_unlock(&registry->mutex);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_create_device(
    iree_hal_driver_registry_t* registry, iree_string_view_t device_uri,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(registry);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, device_uri.data, device_uri.size);

  // Try to create the driver; the backing driver factory may cache this but
  // it's not safe to assume the returned driver is going to remain live for
  // any longer than this function holds a reference.
  iree_string_view_t driver_name = iree_uri_schema(device_uri);
  iree_hal_driver_t* driver = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_driver_registry_try_create(registry, driver_name, host_allocator,
                                          &driver),
      "creating driver for device '%.*s'", (int)device_uri.size,
      device_uri.data);

  // Have the driver create the device.
  iree_status_t status = iree_hal_driver_create_device_by_uri(
      driver, device_uri, host_allocator, out_device);

  iree_hal_driver_release(driver);

  IREE_TRACE_ZONE_END(z0);
  return status;
}
