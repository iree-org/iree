// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/level_zero/dynamic_symbols.h"

#include <iostream>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"

namespace iree {
namespace hal {
namespace level_zero {
namespace {

#define LEVEL_ZERO_CHECK_ERRORS(expr)     \
  {                                       \
    ze_result_t status = expr;            \
    ASSERT_EQ(ZE_RESULT_SUCCESS, status); \
  }

TEST(DynamicSymbolsTest, CreateFromSystemLoader) {
  iree_hal_level_zero_dynamic_symbols_t symbols;
  iree_status_t status = iree_hal_level_zero_dynamic_symbols_initialize(
      iree_allocator_system(), &symbols);
  if (!iree_status_is_ok(status)) {
    std::cerr << "Symbols cannot be loaded, skipping test.";
    GTEST_SKIP();
  }

  LEVEL_ZERO_CHECK_ERRORS(symbols.zeInit(0));
  // Get the driver
  uint32_t driverCount = 0;
  LEVEL_ZERO_CHECK_ERRORS(symbols.zeDriverGet(&driverCount, nullptr));
  ze_driver_handle_t driverHandle;
  if (driverCount > 0) {
    LEVEL_ZERO_CHECK_ERRORS(symbols.zeDriverGet(&driverCount, &driverHandle));
  } else {
    std::cerr << "Cannot find Intel Level Zero driver, skipping test.";
    GTEST_SKIP();
  }

  // Create the context
  ze_context_desc_t contextDescription = {};
  contextDescription.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  ze_context_handle_t context;
  LEVEL_ZERO_CHECK_ERRORS(
      symbols.zeContextCreate(driverHandle, &contextDescription, &context));

  // Get the device
  uint32_t deviceCount = 0;
  LEVEL_ZERO_CHECK_ERRORS(
      symbols.zeDeviceGet(driverHandle, &deviceCount, nullptr));

  ze_device_handle_t device;
  if (deviceCount > 0) {
    LEVEL_ZERO_CHECK_ERRORS(
        symbols.zeDeviceGet(driverHandle, &deviceCount, &device));
  } else {
    std::cerr << "Cannot find Intel Level Zero device, skipping test.";
    GTEST_SKIP();
  }

  // Print basic properties of the device
  ze_device_properties_t deviceProperties = {};
  LEVEL_ZERO_CHECK_ERRORS(
      symbols.zeDeviceGetProperties(device, &deviceProperties));
  std::cout << "Device   : " << deviceProperties.name << "\n"
            << "Type     : "
            << ((deviceProperties.type == ZE_DEVICE_TYPE_GPU) ? "GPU" : "FPGA")
            << "\n"
            << "Vendor ID: " << std::hex << deviceProperties.vendorId
            << std::dec << "\n";

  iree_hal_level_zero_dynamic_symbols_deinitialize(&symbols);
}

}  // namespace
}  // namespace level_zero
}  // namespace hal
}  // namespace iree
