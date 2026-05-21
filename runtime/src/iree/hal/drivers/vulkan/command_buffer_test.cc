// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/command_buffer.h"

#include <cstdint>
#include <memory>
#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::vulkan {
namespace {

#if !IREE_HAL_VULKAN_LIBVULKAN_STATIC

struct NativeReplayCapture {
  // Count of fake vkBeginCommandBuffer calls.
  int begin_command_buffer_count = 0;

  // Count of fake vkEndCommandBuffer calls.
  int end_command_buffer_count = 0;

  // Count of fake vkCmdBeginDebugUtilsLabelEXT calls.
  int begin_label_count = 0;

  // Count of fake vkCmdEndDebugUtilsLabelEXT calls.
  int end_label_count = 0;

  // VkCommandBuffer passed to vkBeginCommandBuffer.
  VkCommandBuffer begin_command_buffer = VK_NULL_HANDLE;

  // VkCommandBuffer passed to vkEndCommandBuffer.
  VkCommandBuffer end_command_buffer = VK_NULL_HANDLE;

  // VkCommandBuffer passed to vkCmdBeginDebugUtilsLabelEXT.
  VkCommandBuffer begin_label_command_buffer = VK_NULL_HANDLE;

  // VkCommandBuffer passed to vkCmdEndDebugUtilsLabelEXT.
  VkCommandBuffer end_label_command_buffer = VK_NULL_HANDLE;

  // Label copied during the fake begin-label entry-point call.
  std::string label;

  // RGBA color copied during the fake begin-label entry-point call.
  float color[4] = {0.0f, 0.0f, 0.0f, 0.0f};
};

static NativeReplayCapture* g_native_replay_capture = nullptr;

static VKAPI_ATTR VkResult VKAPI_CALL
FakeBeginCommandBuffer(VkCommandBuffer command_buffer,
                       const VkCommandBufferBeginInfo* begin_info) {
  ++g_native_replay_capture->begin_command_buffer_count;
  g_native_replay_capture->begin_command_buffer = command_buffer;
  EXPECT_EQ(0u, begin_info->flags);
  return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL
FakeEndCommandBuffer(VkCommandBuffer command_buffer) {
  ++g_native_replay_capture->end_command_buffer_count;
  g_native_replay_capture->end_command_buffer = command_buffer;
  return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL FakeCmdBeginDebugUtilsLabelEXT(
    VkCommandBuffer command_buffer, const VkDebugUtilsLabelEXT* label_info) {
  ++g_native_replay_capture->begin_label_count;
  g_native_replay_capture->begin_label_command_buffer = command_buffer;
  g_native_replay_capture->label = label_info->pLabelName;
  for (int i = 0; i < 4; ++i) {
    g_native_replay_capture->color[i] = label_info->color[i];
  }
}

static VKAPI_ATTR void VKAPI_CALL
FakeCmdEndDebugUtilsLabelEXT(VkCommandBuffer command_buffer) {
  ++g_native_replay_capture->end_label_count;
  g_native_replay_capture->end_label_command_buffer = command_buffer;
}

static VKAPI_ATTR void VKAPI_CALL FakeCmdInsertDebugUtilsLabelEXT(
    VkCommandBuffer command_buffer, const VkDebugUtilsLabelEXT* label_info) {
  (void)command_buffer;
  (void)label_info;
}

static iree_hal_vulkan_device_syms_t MakeNativeReplaySyms() {
  iree_hal_vulkan_device_syms_t syms = {};
  syms.vkBeginCommandBuffer = FakeBeginCommandBuffer;
  syms.vkEndCommandBuffer = FakeEndCommandBuffer;
  syms.vkCmdBeginDebugUtilsLabelEXT = FakeCmdBeginDebugUtilsLabelEXT;
  syms.vkCmdEndDebugUtilsLabelEXT = FakeCmdEndDebugUtilsLabelEXT;
  syms.vkCmdInsertDebugUtilsLabelEXT = FakeCmdInsertDebugUtilsLabelEXT;
  return syms;
}

struct CommandBufferDeleter {
  void operator()(iree_hal_command_buffer_t* command_buffer) const {
    iree_hal_command_buffer_release(command_buffer);
  }
};

using CommandBufferPtr =
    std::unique_ptr<iree_hal_command_buffer_t, CommandBufferDeleter>;

class VulkanCommandBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_hal_allocator_create_heap(
        iree_make_cstring_view("vulkan_command_buffer_test"),
        iree_allocator_system(), iree_allocator_system(), &device_allocator_));
    iree_arena_block_pool_initialize(block_size_, iree_allocator_system(),
                                     &block_pool_);
  }

  void TearDown() override {
    iree_arena_block_pool_deinitialize(&block_pool_);
    iree_hal_allocator_release(device_allocator_);
  }

  CommandBufferPtr CreateCommandBuffer() {
    iree_hal_command_buffer_t* command_buffer = nullptr;
    IREE_EXPECT_OK(iree_hal_vulkan_command_buffer_create(
        device_allocator_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_COMMAND_CATEGORY_ANY, IREE_HAL_QUEUE_AFFINITY_ANY,
        /*binding_capacity=*/0, &block_pool_, iree_allocator_system(),
        &command_buffer));
    return CommandBufferPtr(command_buffer);
  }

 private:
  // Test allocator borrowed by command buffers for validation.
  iree_hal_allocator_t* device_allocator_ = nullptr;

  // Command-buffer block size used by this fixture.
  iree_host_size_t block_size_ = 256;

  // Command-payload and resource-set block pool borrowed by command buffers.
  iree_arena_block_pool_t block_pool_;
};

TEST_F(VulkanCommandBufferTest, ReplaysDebugGroupsAsDebugUtilsLabels) {
  CommandBufferPtr command_buffer = CreateCommandBuffer();
  ASSERT_NE(command_buffer, nullptr);

  iree_hal_label_color_t label_color = iree_hal_label_color_unspecified();
  label_color.r = 0x10;
  label_color.g = 0x20;
  label_color.b = 0x30;
  label_color.a = 0x40;
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer.get()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin_debug_group(
      command_buffer.get(), IREE_SV("dispatch-region"), label_color,
      /*location=*/nullptr));
  IREE_ASSERT_OK(iree_hal_command_buffer_end_debug_group(command_buffer.get()));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer.get()));

  EXPECT_FALSE(iree_hal_vulkan_command_buffer_is_empty(command_buffer.get()));
  EXPECT_TRUE(
      iree_hal_vulkan_command_buffer_has_native_commands(command_buffer.get()));

  NativeReplayCapture capture;
  g_native_replay_capture = &capture;
  iree_hal_vulkan_device_syms_t syms = MakeNativeReplaySyms();
  iree_hal_vulkan_debug_utils_t debug_utils;
  debug_utils.flags = IREE_HAL_VULKAN_DEBUG_UTILS_FLAG_COMMAND_LABELS;
  iree_hal_vulkan_builtins_t builtins = {};
  iree_hal_buffer_binding_table_t binding_table = {};
  VkDevice logical_device =
      reinterpret_cast<VkDevice>(static_cast<uintptr_t>(0x1234));
  VkCommandBuffer native_command_buffer =
      reinterpret_cast<VkCommandBuffer>(static_cast<uintptr_t>(0x5678));

  IREE_ASSERT_OK(iree_hal_vulkan_command_buffer_record_native(
      command_buffer.get(), &syms, logical_device, &debug_utils, &builtins,
      native_command_buffer, /*usage_flags=*/0, VK_NULL_HANDLE, binding_table,
      /*bda_publication=*/nullptr, /*bda_binding_cache=*/nullptr,
      /*profile_marker=*/nullptr, iree_allocator_system()));

  EXPECT_EQ(1, capture.begin_command_buffer_count);
  EXPECT_EQ(1, capture.end_command_buffer_count);
  EXPECT_EQ(1, capture.begin_label_count);
  EXPECT_EQ(1, capture.end_label_count);
  EXPECT_EQ(native_command_buffer, capture.begin_command_buffer);
  EXPECT_EQ(native_command_buffer, capture.end_command_buffer);
  EXPECT_EQ(native_command_buffer, capture.begin_label_command_buffer);
  EXPECT_EQ(native_command_buffer, capture.end_label_command_buffer);
  EXPECT_EQ("dispatch-region", capture.label);
  EXPECT_FLOAT_EQ(16.0f / 255.0f, capture.color[0]);
  EXPECT_FLOAT_EQ(32.0f / 255.0f, capture.color[1]);
  EXPECT_FLOAT_EQ(48.0f / 255.0f, capture.color[2]);
  EXPECT_FLOAT_EQ(64.0f / 255.0f, capture.color[3]);
  g_native_replay_capture = nullptr;
}

#endif  // !IREE_HAL_VULKAN_LIBVULKAN_STATIC

}  // namespace
}  // namespace iree::hal::vulkan
