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

  int pipeline_barrier_count = 0;
  VkMemoryBarrier2 memory_barrier = {};

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

static VKAPI_ATTR void VKAPI_CALL FakeCmdPipelineBarrier2(
    VkCommandBuffer command_buffer, const VkDependencyInfo* dependency_info) {
  (void)command_buffer;
  ++g_native_replay_capture->pipeline_barrier_count;
  ASSERT_LE(dependency_info->memoryBarrierCount, 1u);
  if (dependency_info->memoryBarrierCount == 1) {
    g_native_replay_capture->memory_barrier =
        dependency_info->pMemoryBarriers[0];
  }
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
  syms.vkCmdPipelineBarrier2 = FakeCmdPipelineBarrier2;
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

  CommandBufferPtr CreateCommandBuffer(iree_host_size_t binding_capacity = 0) {
    iree_hal_command_buffer_t* command_buffer = nullptr;
    IREE_EXPECT_OK(iree_hal_vulkan_command_buffer_create(
        device_allocator_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_COMMAND_CATEGORY_ANY, IREE_HAL_QUEUE_AFFINITY_ANY,
        binding_capacity, &block_pool_, iree_allocator_system(),
        &command_buffer));
    return CommandBufferPtr(command_buffer);
  }

  iree_hal_buffer_t* AllocateBuffer(iree_hal_memory_type_t memory_type) {
    iree_hal_buffer_params_t params = {};
    params.type = memory_type;
    params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
    iree_hal_buffer_t* buffer = nullptr;
    IREE_EXPECT_OK(iree_hal_allocator_allocate_buffer(
        device_allocator_, params, /*allocation_size=*/32, &buffer));
    return buffer;
  }

 private:
  // Test allocator borrowed by command buffers for validation.
  iree_hal_allocator_t* device_allocator_ = nullptr;

  // Command-buffer block size used by this fixture.
  iree_host_size_t block_size_ = 256;

  // Command-payload and resource-set block pool borrowed by command buffers.
  iree_arena_block_pool_t block_pool_;
};

static iree_status_t RecordNativeForTest(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    NativeReplayCapture* capture) {
  g_native_replay_capture = capture;
  iree_hal_vulkan_device_syms_t syms = MakeNativeReplaySyms();
  iree_hal_vulkan_debug_utils_t debug_utils = {};
  iree_hal_vulkan_builtins_t builtins = {};
  iree_status_t status = iree_hal_vulkan_command_buffer_record_native(
      command_buffer, &syms,
      reinterpret_cast<VkDevice>(static_cast<uintptr_t>(0x1234)),
      &debug_utils, &builtins,
      reinterpret_cast<VkCommandBuffer>(static_cast<uintptr_t>(0x5678)),
      /*usage_flags=*/0, VK_NULL_HANDLE, binding_table,
      /*bda_publication=*/nullptr, /*bda_binding_cache=*/nullptr,
      /*profile_marker=*/nullptr, iree_allocator_system());
  g_native_replay_capture = nullptr;
  return status;
}

TEST_F(VulkanCommandBufferTest, CoherentDirectMaintenance) {
  iree_hal_buffer_t* buffer = AllocateBuffer(
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE);
  ASSERT_NE(buffer, nullptr);
  CommandBufferPtr command_buffer = CreateCommandBuffer();
  ASSERT_NE(command_buffer, nullptr);
  iree_hal_buffer_ref_t target_ref = iree_hal_make_buffer_ref(buffer, 4, 8);
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer.get()));
  IREE_ASSERT_OK(
      iree_hal_command_buffer_invalidate_buffer(command_buffer.get(), target_ref));
  IREE_ASSERT_OK(
      iree_hal_command_buffer_flush_buffer(command_buffer.get(), target_ref));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer.get()));

  NativeReplayCapture capture;
  IREE_ASSERT_OK(RecordNativeForTest(command_buffer.get(),
                                     iree_hal_buffer_binding_table_empty(),
                                     &capture));
  EXPECT_EQ(1, capture.pipeline_barrier_count);
  EXPECT_EQ(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            capture.memory_barrier.srcStageMask);
  EXPECT_EQ(VK_ACCESS_2_MEMORY_WRITE_BIT, capture.memory_barrier.srcAccessMask);
  EXPECT_EQ(VK_PIPELINE_STAGE_2_HOST_BIT,
            capture.memory_barrier.dstStageMask);
  EXPECT_EQ(VK_ACCESS_2_MEMORY_READ_BIT, capture.memory_barrier.dstAccessMask);
  iree_hal_buffer_release(buffer);
}

TEST_F(VulkanCommandBufferTest, UnsupportedDirectMaintenanceFails) {
  // The heap allocator adds HOST_VISIBLE but does not add HOST_COHERENT.
  iree_hal_buffer_t* buffer =
      AllocateBuffer(IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE);
  ASSERT_NE(buffer, nullptr);
  for (int i = 0; i < 2; ++i) {
    const bool flush = i != 0;
    CommandBufferPtr command_buffer = CreateCommandBuffer();
    ASSERT_NE(command_buffer, nullptr);
    iree_hal_buffer_ref_t target_ref = iree_hal_make_buffer_ref(buffer, 4, 8);
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer.get()));
    if (flush) {
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_UNIMPLEMENTED,
          iree_hal_command_buffer_flush_buffer(command_buffer.get(),
                                               target_ref));
    } else {
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_UNIMPLEMENTED,
          iree_hal_command_buffer_invalidate_buffer(command_buffer.get(),
                                                    target_ref));
    }
  }
  iree_hal_buffer_release(buffer);
}

TEST_F(VulkanCommandBufferTest, IndirectMaintenanceRechecksSubmittedBuffer) {
  iree_hal_buffer_t* coherent_buffer = AllocateBuffer(
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE);
  iree_hal_buffer_t* unsupported_buffer =
      AllocateBuffer(IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE);
  ASSERT_NE(coherent_buffer, nullptr);
  ASSERT_NE(unsupported_buffer, nullptr);
  CommandBufferPtr command_buffer = CreateCommandBuffer(/*binding_capacity=*/1);
  ASSERT_NE(command_buffer, nullptr);
  iree_hal_buffer_ref_t target_ref =
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/0, /*offset=*/2,
                                        /*length=*/8);
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer.get()));
  IREE_ASSERT_OK(
      iree_hal_command_buffer_invalidate_buffer(command_buffer.get(), target_ref));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer.get()));
  EXPECT_TRUE(iree_hal_vulkan_command_buffer_has_indirect_cache_maintenance(
      command_buffer.get()));

  iree_hal_buffer_binding_t binding = {coherent_buffer, 4, 16};
  iree_hal_buffer_binding_table_t binding_table = {1, &binding};
  NativeReplayCapture first_capture;
  IREE_ASSERT_OK(
      RecordNativeForTest(command_buffer.get(), binding_table, &first_capture));
  EXPECT_EQ(0, first_capture.pipeline_barrier_count);

  binding.buffer = unsupported_buffer;
  NativeReplayCapture second_capture;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      RecordNativeForTest(command_buffer.get(), binding_table, &second_capture));
  EXPECT_EQ(0, second_capture.pipeline_barrier_count);

  // The recorded offset and length still constrain the submitted subrange.
  binding.buffer = coherent_buffer;
  binding.length = 8;
  NativeReplayCapture short_binding_capture;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      RecordNativeForTest(command_buffer.get(), binding_table,
                          &short_binding_capture));
  iree_hal_buffer_release(unsupported_buffer);
  iree_hal_buffer_release(coherent_buffer);
}

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
