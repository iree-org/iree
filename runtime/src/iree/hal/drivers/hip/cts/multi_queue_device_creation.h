// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HIP_REGISTRATION_MULTI_QUEUE_H_
#define IREE_HAL_DRIVERS_HIP_REGISTRATION_MULTI_QUEUE_H_

#include <map>

#include "iree/hal/driver.h"
#include "iree/testing/status_matchers.h"

inline iree_status_t iree_hal_drivers_hip_cts_default_multi_queue_create(
    iree_hal_driver_t* driver, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  *out_device = NULL;
  std::multimap<std::string, iree_host_size_t> grouped_devices;

  iree_host_size_t device_info_count = 0;
  iree_hal_device_info_t* device_infos = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_driver_query_available_devices(
      driver, iree_allocator_system(), &device_info_count, &device_infos));

  for (iree_host_size_t i = 0; i < device_info_count; ++i) {
    const char* nm = device_infos[i].name.data;
    iree_host_size_t size = device_infos[i].name.size;

    std::string name(nm, size);
    grouped_devices.insert(std::make_pair(name, i));
  }

  std::string path;
  iree_host_size_t max_valid_devices = 0;
  for (auto it = grouped_devices.begin(); it != grouped_devices.end();
       /*empty on purpose*/) {
    iree_host_size_t device_count = grouped_devices.count(it->first);
    if (device_count == 1) {
      ++it;
      continue;
    }
    if (device_count <= max_valid_devices) {
      for (iree_host_size_t j = 0; j < device_count; ++j) {
        // No += for multimap iterator.
        ++it;
      }
      continue;
    }
    path = "";
    for (iree_host_size_t i = 0; i < device_count; ++i) {
      if (i > 0) {
        path += ",";
      }
      path += std::to_string(it->second);
      ++it;
    }
    max_valid_devices = device_count;
  }

  iree_allocator_free(iree_allocator_system(), device_infos);

  if (!max_valid_devices) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "No device group found on the system");
  }
  return iree_hal_driver_create_device_by_path(
      driver, IREE_SV("hip"), IREE_SV(path.c_str()), /*param_count=*/0,
      /*params=*/NULL, iree_allocator_system(), out_device);
}

#endif  // IREE_HAL_DRIVERS_HIP_REGISTRATION_MULTI_QUEUE_H_
