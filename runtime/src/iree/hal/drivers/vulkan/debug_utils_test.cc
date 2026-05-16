// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/debug_utils.h"

#include <cstdint>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::vulkan {
namespace {

#if !IREE_HAL_VULKAN_LIBVULKAN_STATIC

struct ObjectNameCapture {
  // Count of fake vkSetDebugUtilsObjectNameEXT calls.
  int call_count = 0;

  // VkDevice passed to the fake entry point.
  VkDevice device = VK_NULL_HANDLE;

  // Object type captured from VkDebugUtilsObjectNameInfoEXT.
  VkObjectType object_type = VK_OBJECT_TYPE_UNKNOWN;

  // Object handle captured from VkDebugUtilsObjectNameInfoEXT.
  uint64_t object_handle = 0;

  // Object name copied during the fake entry-point call.
  std::string name;
};

static ObjectNameCapture* g_object_name_capture = nullptr;

static VKAPI_ATTR VkResult VKAPI_CALL FakeSetDebugUtilsObjectNameEXT(
    VkDevice device, const VkDebugUtilsObjectNameInfoEXT* name_info) {
  ++g_object_name_capture->call_count;
  g_object_name_capture->device = device;
  g_object_name_capture->object_type = name_info->objectType;
  g_object_name_capture->object_handle = name_info->objectHandle;
  g_object_name_capture->name = name_info->pObjectName;
  return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL FakeCmdBeginDebugUtilsLabelEXT(
    VkCommandBuffer command_buffer, const VkDebugUtilsLabelEXT* label_info) {
  (void)command_buffer;
  (void)label_info;
}

static VKAPI_ATTR void VKAPI_CALL
FakeCmdEndDebugUtilsLabelEXT(VkCommandBuffer command_buffer) {
  (void)command_buffer;
}

static VKAPI_ATTR void VKAPI_CALL FakeCmdInsertDebugUtilsLabelEXT(
    VkCommandBuffer command_buffer, const VkDebugUtilsLabelEXT* label_info) {
  (void)command_buffer;
  (void)label_info;
}

static iree_hal_vulkan_device_syms_t MakeDebugUtilsSyms() {
  iree_hal_vulkan_device_syms_t syms = {};
  syms.vkSetDebugUtilsObjectNameEXT = FakeSetDebugUtilsObjectNameEXT;
  syms.vkCmdBeginDebugUtilsLabelEXT = FakeCmdBeginDebugUtilsLabelEXT;
  syms.vkCmdEndDebugUtilsLabelEXT = FakeCmdEndDebugUtilsLabelEXT;
  syms.vkCmdInsertDebugUtilsLabelEXT = FakeCmdInsertDebugUtilsLabelEXT;
  return syms;
}

#endif  // !IREE_HAL_VULKAN_LIBVULKAN_STATIC

TEST(DebugUtilsTest, InitializeWithoutRequestDisablesFamilies) {
  iree_hal_vulkan_device_syms_t syms = {};
  iree_hal_vulkan_debug_utils_t debug_utils;
  IREE_ASSERT_OK(iree_hal_vulkan_debug_utils_initialize(
      IREE_HAL_VULKAN_REQUEST_FLAG_NONE, &syms, &debug_utils));

  EXPECT_EQ(IREE_HAL_VULKAN_DEBUG_UTILS_FLAG_NONE, debug_utils.flags);
}

TEST(DebugUtilsTest, InitializeRejectsUnknownRequestFlags) {
  iree_hal_vulkan_device_syms_t syms = {};
  iree_hal_vulkan_debug_utils_t debug_utils;
  IREE_EXPECT_STATUS_IS(StatusCode::kInvalidArgument,
                        iree_hal_vulkan_debug_utils_initialize(
                            IREE_HAL_VULKAN_REQUEST_FLAG_ALL_RECOGNIZED + 1,
                            &syms, &debug_utils));
}

#if !IREE_HAL_VULKAN_LIBVULKAN_STATIC

TEST(DebugUtilsTest, InitializeWithRequestEnablesFamilies) {
  iree_hal_vulkan_device_syms_t syms = MakeDebugUtilsSyms();

  iree_hal_vulkan_debug_utils_t debug_utils;
  IREE_ASSERT_OK(iree_hal_vulkan_debug_utils_initialize(
      IREE_HAL_VULKAN_REQUEST_FLAG_DEBUG_UTILS, &syms, &debug_utils));

  EXPECT_TRUE(iree_hal_vulkan_debug_utils_has(
      &debug_utils, IREE_HAL_VULKAN_DEBUG_UTILS_FLAG_OBJECT_NAMES));
  EXPECT_TRUE(iree_hal_vulkan_debug_utils_has(
      &debug_utils, IREE_HAL_VULKAN_DEBUG_UTILS_FLAG_COMMAND_LABELS));
}

TEST(DebugUtilsTest, InitializeWithRequestRequiresObjectNames) {
  iree_hal_vulkan_device_syms_t syms = MakeDebugUtilsSyms();
  syms.vkSetDebugUtilsObjectNameEXT = nullptr;

  iree_hal_vulkan_debug_utils_t debug_utils;
  IREE_EXPECT_STATUS_IS(
      StatusCode::kFailedPrecondition,
      iree_hal_vulkan_debug_utils_initialize(
          IREE_HAL_VULKAN_REQUEST_FLAG_DEBUG_UTILS, &syms, &debug_utils));
}

TEST(DebugUtilsTest, InitializeWithRequestRequiresCommandLabels) {
  iree_hal_vulkan_device_syms_t syms = MakeDebugUtilsSyms();
  syms.vkCmdEndDebugUtilsLabelEXT = nullptr;

  iree_hal_vulkan_debug_utils_t debug_utils;
  IREE_EXPECT_STATUS_IS(
      StatusCode::kFailedPrecondition,
      iree_hal_vulkan_debug_utils_initialize(
          IREE_HAL_VULKAN_REQUEST_FLAG_DEBUG_UTILS, &syms, &debug_utils));
}

TEST(DebugUtilsTest, SetObjectNameNoopsWhenObjectNamesDisabled) {
  iree_hal_vulkan_device_syms_t syms = MakeDebugUtilsSyms();
  iree_hal_vulkan_debug_utils_t debug_utils;
  debug_utils.flags = IREE_HAL_VULKAN_DEBUG_UTILS_FLAG_NONE;
  ObjectNameCapture capture;
  g_object_name_capture = &capture;

  IREE_ASSERT_OK(iree_hal_vulkan_debug_utils_set_object_name(
      &debug_utils, &syms,
      reinterpret_cast<VkDevice>(static_cast<uintptr_t>(0x1234)),
      VK_OBJECT_TYPE_BUFFER, 0x5678, IREE_SV("buffer"),
      iree_allocator_system()));

  EXPECT_EQ(0, capture.call_count);
  g_object_name_capture = nullptr;
}

TEST(DebugUtilsTest, SetObjectNameCallsVulkan) {
  iree_hal_vulkan_device_syms_t syms = MakeDebugUtilsSyms();
  iree_hal_vulkan_debug_utils_t debug_utils;
  debug_utils.flags = IREE_HAL_VULKAN_DEBUG_UTILS_FLAG_OBJECT_NAMES;
  ObjectNameCapture capture;
  g_object_name_capture = &capture;
  VkDevice device = reinterpret_cast<VkDevice>(static_cast<uintptr_t>(0x1234));

  IREE_ASSERT_OK(iree_hal_vulkan_debug_utils_set_object_name(
      &debug_utils, &syms, device, VK_OBJECT_TYPE_BUFFER, 0x5678,
      IREE_SV("scratch-buffer"), iree_allocator_system()));

  EXPECT_EQ(1, capture.call_count);
  EXPECT_EQ(device, capture.device);
  EXPECT_EQ(VK_OBJECT_TYPE_BUFFER, capture.object_type);
  EXPECT_EQ(0x5678u, capture.object_handle);
  EXPECT_EQ("scratch-buffer", capture.name);
  g_object_name_capture = nullptr;
}

TEST(DebugUtilsTest, SetQueueNameRejectsMissingRoles) {
  iree_hal_vulkan_device_syms_t syms = MakeDebugUtilsSyms();
  iree_hal_vulkan_debug_utils_t debug_utils;
  debug_utils.flags = IREE_HAL_VULKAN_DEBUG_UTILS_FLAG_OBJECT_NAMES;
  VkDevice device = reinterpret_cast<VkDevice>(static_cast<uintptr_t>(0x1234));
  VkQueue queue = reinterpret_cast<VkQueue>(static_cast<uintptr_t>(0x5678));

  IREE_EXPECT_STATUS_IS(StatusCode::kInvalidArgument,
                        iree_hal_vulkan_debug_utils_set_queue_name(
                            &debug_utils, &syms, device, queue,
                            IREE_HAL_VULKAN_DEBUG_UTILS_QUEUE_ROLE_NONE,
                            /*queue_family_index=*/1, /*queue_index=*/2,
                            IREE_SV("device"), iree_allocator_system()));
}

TEST(DebugUtilsTest, SetQueueNameCallsVulkan) {
  iree_hal_vulkan_device_syms_t syms = MakeDebugUtilsSyms();
  iree_hal_vulkan_debug_utils_t debug_utils;
  debug_utils.flags = IREE_HAL_VULKAN_DEBUG_UTILS_FLAG_OBJECT_NAMES;
  ObjectNameCapture capture;
  g_object_name_capture = &capture;
  VkDevice device = reinterpret_cast<VkDevice>(static_cast<uintptr_t>(0x1234));
  VkQueue queue = reinterpret_cast<VkQueue>(static_cast<uintptr_t>(0x5678));

  IREE_ASSERT_OK(iree_hal_vulkan_debug_utils_set_queue_name(
      &debug_utils, &syms, device, queue,
      IREE_HAL_VULKAN_DEBUG_UTILS_QUEUE_ROLE_COMPUTE |
          IREE_HAL_VULKAN_DEBUG_UTILS_QUEUE_ROLE_TRANSFER,
      /*queue_family_index=*/1, /*queue_index=*/2, IREE_SV("device"),
      iree_allocator_system()));

  EXPECT_EQ(1, capture.call_count);
  EXPECT_EQ(device, capture.device);
  EXPECT_EQ(VK_OBJECT_TYPE_QUEUE, capture.object_type);
  EXPECT_EQ(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(queue)),
            capture.object_handle);
  EXPECT_EQ("device/compute+transfer-queue[1:2]", capture.name);
  g_object_name_capture = nullptr;
}

#endif  // !IREE_HAL_VULKAN_LIBVULKAN_STATIC

}  // namespace
}  // namespace iree::hal::vulkan
