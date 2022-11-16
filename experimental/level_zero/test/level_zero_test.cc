// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iree/runtime/api.h>
#include <iree/testing/gtest.h>

#include <memory>
#include <sstream>
#include <string>

template <typename T, typename Deleter>
std::unique_ptr<T, Deleter> make_unique(T* p, Deleter d) {
  return std::unique_ptr<T, Deleter>(p, d);
}

TEST(LevelZeroTest, UUID) {
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);

  static const char* driver_name = "level_zero";

  iree_allocator_t host_allocator = iree_allocator_system();

  // Make instance.
  iree_runtime_instance_t* instance = nullptr;
  ASSERT_EQ(iree_runtime_instance_create(&instance_options,
                                         iree_allocator_system(), &instance),
            iree_ok_status());
  auto instance_deleter = make_unique<iree_runtime_instance_t>(
      instance,
      [](iree_runtime_instance_t* p) { iree_runtime_instance_release(p); });

  // Make Level Zero driver.
  iree_hal_driver_registry_t* driver_registry =
      iree_runtime_instance_driver_registry(instance);
  iree_hal_driver_t* driver = nullptr;
  ASSERT_EQ(iree_hal_driver_registry_try_create(
                driver_registry, iree_make_cstring_view(driver_name),
                host_allocator, &driver),
            iree_ok_status());
  auto driver_deleter = make_unique<iree_hal_driver_t>(
      driver, [](iree_hal_driver_t* p) { iree_hal_driver_release(p); });

  // Get list of available devices.
  iree_hal_device_info_t* device_infos = nullptr;
  iree_host_size_t device_infos_count = 0;
  ASSERT_EQ(iree_hal_driver_query_available_devices(
                driver, host_allocator, &device_infos_count, &device_infos),
            iree_ok_status());
  auto device_infos_deleter = make_unique<iree_hal_device_info_t>(
      device_infos, [host_allocator](iree_hal_device_info_t* p) {
        iree_allocator_free(host_allocator, p);
      });
  ASSERT_GT(device_infos_count, 0);

  // Create a valid device from URI.
  std::stringstream device_uri;
  device_uri << driver_name << "://";
  device_uri << std::string(device_infos[0].path.data,
                            device_infos[0].path.size);
  std::string device_uri_str = device_uri.str();
  iree_hal_device_t* device = nullptr;
  ASSERT_EQ(iree_hal_driver_create_device_by_uri(
                driver, iree_make_cstring_view(device_uri_str.c_str()),
                host_allocator, &device),
            iree_ok_status());
  auto device_deleter = make_unique<iree_hal_device_t>(
      device, [](iree_hal_device_t* p) { iree_hal_device_release(p); });

  // Try create an invalid device from URI.
  std::stringstream invalid_device_uri;
  invalid_device_uri << driver_name
                     << "://4e5a272e-66a7-11ed-9342-4f1f581f812c";
  std::string invalid_device_uri_str = invalid_device_uri.str();
  iree_hal_device_t* invalid_device = nullptr;
  ASSERT_NE(iree_hal_driver_create_device_by_uri(
                driver, iree_make_cstring_view(invalid_device_uri_str.c_str()),
                host_allocator, &invalid_device),
            iree_ok_status());
  auto invalid_device_deleter = make_unique<iree_hal_device_t>(
      invalid_device, [](iree_hal_device_t* p) { iree_hal_device_release(p); });
}
