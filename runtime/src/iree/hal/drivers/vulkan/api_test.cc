// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/api.h"

#include <cstring>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "vulkan/vulkan.h"

namespace iree::hal::vulkan {
namespace {

static bool ContainsString(iree_host_size_t count, const char* const* values,
                           const char* value) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (std::strcmp(values[i], value) == 0) return true;
  }
  return false;
}

TEST(ApiTest, RecognizesOptionalDeviceExtensionNames) {
  const char* extension_names[] = {
      VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
      IREE_HAL_VULKAN_KHR_SHADER_BFLOAT16_EXTENSION_NAME,
      VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
      "VK_EXAMPLE_unrecognized",
  };
  iree_hal_vulkan_device_extensions_t extensions = 0;
  IREE_ASSERT_OK(iree_hal_vulkan_device_extensions_from_names(
      IREE_ARRAYSIZE(extension_names), extension_names, &extensions));
  EXPECT_TRUE(iree_all_bits_set(
      extensions, IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_COOPERATIVE_MATRIX));
  EXPECT_TRUE(iree_all_bits_set(
      extensions, IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_SHADER_BFLOAT16));
  EXPECT_TRUE(iree_all_bits_set(
      extensions, IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_PUSH_DESCRIPTOR));
}

TEST(ApiTest, RecognizesAllPublicDeviceExtensionNames) {
  const char* extension_names[] = {
      "VK_EXAMPLE_unrecognized",
      VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
#if defined(VK_USE_PLATFORM_WIN32_KHR)
      VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
#else
      VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
#endif  // defined(VK_USE_PLATFORM_WIN32_KHR)
      VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME,
      VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME,
      VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
      VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
      IREE_HAL_VULKAN_KHR_SHADER_BFLOAT16_EXTENSION_NAME,
  };
  iree_hal_vulkan_device_extensions_t extensions = 0;
  IREE_ASSERT_OK(iree_hal_vulkan_device_extensions_from_names(
      IREE_ARRAYSIZE(extension_names), extension_names, &extensions));
  EXPECT_TRUE(iree_all_bits_set(
      extensions, IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_EXTERNAL_MEMORY));
#if defined(VK_USE_PLATFORM_WIN32_KHR)
  EXPECT_TRUE(iree_all_bits_set(
      extensions, IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_EXTERNAL_MEMORY_WIN32));
#else
  EXPECT_TRUE(iree_all_bits_set(
      extensions, IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_EXTERNAL_MEMORY_FD));
#endif  // defined(VK_USE_PLATFORM_WIN32_KHR)
  EXPECT_TRUE(iree_all_bits_set(
      extensions, IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_EXTERNAL_MEMORY_HOST));
  EXPECT_TRUE(iree_all_bits_set(
      extensions, IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_CALIBRATED_TIMESTAMPS));
  EXPECT_TRUE(iree_all_bits_set(
      extensions, IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_PUSH_DESCRIPTOR));
  EXPECT_TRUE(iree_all_bits_set(
      extensions, IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_COOPERATIVE_MATRIX));
  EXPECT_TRUE(iree_all_bits_set(
      extensions, IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_SHADER_BFLOAT16));
  EXPECT_FALSE(iree_any_bit_set(
      extensions, ~IREE_HAL_VULKAN_DEVICE_EXTENSION_ALL_RECOGNIZED));
}

TEST(ApiTest, DeviceExtensionNameParsingHandlesEmptyAndNullEntries) {
  iree_hal_vulkan_device_extensions_t extensions =
      IREE_HAL_VULKAN_DEVICE_EXTENSION_ALL_RECOGNIZED;
  IREE_ASSERT_OK(iree_hal_vulkan_device_extensions_from_names(
      /*extension_count=*/0, /*extension_names=*/nullptr, &extensions));
  EXPECT_EQ(IREE_HAL_VULKAN_DEVICE_EXTENSION_NONE, extensions);

  const char* extension_names[] = {
      nullptr,
      VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
      nullptr,
  };
  IREE_ASSERT_OK(iree_hal_vulkan_device_extensions_from_names(
      IREE_ARRAYSIZE(extension_names), extension_names, &extensions));
  EXPECT_EQ(IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_PUSH_DESCRIPTOR, extensions);
}

TEST(ApiTest, DeviceExtensionNameParsingRejectsMissingStorage) {
  iree_hal_vulkan_device_extensions_t extensions =
      IREE_HAL_VULKAN_DEVICE_EXTENSION_ALL_RECOGNIZED;
  IREE_EXPECT_STATUS_IS(StatusCode::kInvalidArgument,
                        iree_hal_vulkan_device_extensions_from_names(
                            /*extension_count=*/1,
                            /*extension_names=*/nullptr, &extensions));
  EXPECT_EQ(IREE_HAL_VULKAN_DEVICE_EXTENSION_NONE, extensions);
}

TEST(ApiTest, FeatureBitsReserveRequestFlagPositions) {
  EXPECT_EQ(0u, IREE_HAL_VULKAN_FEATURE_ALL_RECOGNIZED &
                    IREE_HAL_VULKAN_REQUEST_FLAG_ALL_RECOGNIZED);
}

TEST(ApiTest, RequiredExtensibilitySetsAreRequestDriven) {
  iree_host_size_t count = 0;
  IREE_ASSERT_OK(iree_hal_vulkan_query_extensibility_set(
      IREE_HAL_VULKAN_REQUEST_FLAG_NONE,
      IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_LAYERS_REQUIRED,
      /*string_capacity=*/0, &count, /*out_string_values=*/nullptr));
  EXPECT_EQ(0u, count);

  const char* values[1] = {nullptr};
  count = IREE_ARRAYSIZE(values);
  IREE_ASSERT_OK(iree_hal_vulkan_query_extensibility_set(
      IREE_HAL_VULKAN_REQUEST_FLAG_VALIDATION_LAYERS,
      IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_LAYERS_REQUIRED,
      IREE_ARRAYSIZE(values), &count, values));
  ASSERT_EQ(1u, count);
  EXPECT_STREQ("VK_LAYER_KHRONOS_validation", values[0]);

  values[0] = nullptr;
  count = IREE_ARRAYSIZE(values);
  IREE_ASSERT_OK(iree_hal_vulkan_query_extensibility_set(
      IREE_HAL_VULKAN_REQUEST_FLAG_DEBUG_UTILS,
      IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_EXTENSIONS_REQUIRED,
      IREE_ARRAYSIZE(values), &count, values));
  ASSERT_EQ(1u, count);
  EXPECT_STREQ(VK_EXT_DEBUG_UTILS_EXTENSION_NAME, values[0]);
}

TEST(ApiTest, OptionalDeviceExtensionsIncludeSupportedStrategies) {
  iree_host_size_t count = 0;
  IREE_ASSERT_OK(iree_hal_vulkan_query_extensibility_set(
      IREE_HAL_VULKAN_REQUEST_FLAG_NONE,
      IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_OPTIONAL,
      /*string_capacity=*/0, &count, /*out_string_values=*/nullptr));
  std::vector<const char*> values(count);
  IREE_ASSERT_OK(iree_hal_vulkan_query_extensibility_set(
      IREE_HAL_VULKAN_REQUEST_FLAG_NONE,
      IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_OPTIONAL, values.size(),
      &count, values.data()));
  ASSERT_LE(count, values.size());
  EXPECT_TRUE(ContainsString(count, values.data(),
                             VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME));
#if defined(VK_USE_PLATFORM_WIN32_KHR)
  EXPECT_TRUE(ContainsString(count, values.data(),
                             VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME));
#else
  EXPECT_TRUE(ContainsString(count, values.data(),
                             VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME));
#endif  // defined(VK_USE_PLATFORM_WIN32_KHR)
  EXPECT_TRUE(ContainsString(count, values.data(),
                             VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME));
  EXPECT_TRUE(ContainsString(count, values.data(),
                             VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME));
  EXPECT_TRUE(ContainsString(count, values.data(),
                             VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME));
  EXPECT_TRUE(ContainsString(count, values.data(),
                             VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME));
  EXPECT_TRUE(
      ContainsString(count, values.data(),
                     IREE_HAL_VULKAN_KHR_SHADER_BFLOAT16_EXTENSION_NAME));
}

TEST(ApiTest, ExtensibilitySetReportsRequiredCapacity) {
  const char* values[2] = {nullptr, nullptr};
  iree_host_size_t count = 0;
  IREE_EXPECT_STATUS_IS(
      StatusCode::kOutOfRange,
      iree_hal_vulkan_query_extensibility_set(
          IREE_HAL_VULKAN_REQUEST_FLAG_NONE,
          IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_OPTIONAL,
          IREE_ARRAYSIZE(values), &count, values));
  EXPECT_GT(count, IREE_ARRAYSIZE(values));
  EXPECT_NE(nullptr, values[0]);
  EXPECT_NE(nullptr, values[1]);
}

TEST(ApiTest, ExtensibilitySetRejectsInvalidArguments) {
  iree_host_size_t count = 0;
  IREE_EXPECT_STATUS_IS(
      StatusCode::kInvalidArgument,
      iree_hal_vulkan_query_extensibility_set(
          IREE_HAL_VULKAN_REQUEST_FLAG_ALL_RECOGNIZED + 1,
          IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_OPTIONAL,
          /*string_capacity=*/0, &count, /*out_string_values=*/nullptr));

  count = 0;
  IREE_EXPECT_STATUS_IS(
      StatusCode::kInvalidArgument,
      iree_hal_vulkan_query_extensibility_set(
          IREE_HAL_VULKAN_REQUEST_FLAG_NONE,
          IREE_HAL_VULKAN_EXTENSIBILITY_SET_COUNT,
          /*string_capacity=*/0, &count, /*out_string_values=*/nullptr));
}

}  // namespace
}  // namespace iree::hal::vulkan
