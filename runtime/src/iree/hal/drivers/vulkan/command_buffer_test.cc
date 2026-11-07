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
#include "iree/hal/drivers/vulkan/buffer.h"
#include "iree/hal/drivers/vulkan/executable.h"
#include "iree/hal/local/transient_buffer.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::vulkan {
namespace {

// These borrowed fake handles are never submitted or destroyed through Vulkan.
static iree_hal_buffer_t* CreateBorrowedBdaBuffer(
    iree_device_size_t length, iree_device_size_t handle_offset = 0) {
  iree_hal_vulkan_device_syms_t syms = {};
  iree_hal_buffer_t* buffer = nullptr;
  IREE_CHECK_OK(iree_hal_vulkan_buffer_create_borrowed(
      &syms, reinterpret_cast<VkDevice>(uintptr_t{1}), /*placement=*/{},
      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE, IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE, length + handle_offset,
      handle_offset, length, /*memory_property_flags=*/0,
      /*non_coherent_atom_size=*/1, VK_NULL_HANDLE, /*mapping_state=*/nullptr,
      reinterpret_cast<VkBuffer>(uintptr_t{2}), /*device_address=*/0x1000,
      /*release_callback=*/{}, iree_allocator_system(), &buffer));
  return buffer;
}

TEST(VulkanBdaBindingTest, CachedSlotPreservesOwnedAllocationTail) {
  iree_hal_buffer_t* reservation = CreateBorrowedBdaBuffer(8);
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_vulkan_buffer_create_from_reservation(
      reservation, 6, iree_allocator_system(), &buffer));
  iree_hal_buffer_release(reservation);
  iree_hal_buffer_binding_t binding = {buffer, 0, 6};
  iree_hal_buffer_binding_table_t table = {1, &binding};
  iree_hal_vulkan_command_buffer_bda_binding_slot_t cached_slot = {};
  iree_hal_vulkan_command_buffer_bda_binding_cache_t cache = {&cached_slot, 1};
  iree_hal_vulkan_command_buffer_bda_binding_slot_t slot = {};
  IREE_ASSERT_OK(iree_hal_vulkan_command_buffer_resolve_bda_binding_table_slot(
      table, 0, /*empty_binding_address=*/0x2000, &cache, &slot));
  EXPECT_EQ(slot.device_address, 0x1000u);
  EXPECT_EQ(slot.length, 6u);
  EXPECT_EQ(slot.owned_length, 8u);

  iree_hal_vulkan_bda_binding_requirement_t requirement = {1, 0, 4};
  iree_hal_vulkan_pipeline_t pipeline = {};
  pipeline.bda.binding_requirements = &requirement;
  pipeline.bda.binding_requirement_count = 1;
  VkDeviceAddress address = 0;
  IREE_ASSERT_OK(iree_hal_vulkan_command_buffer_resolve_bda_binding(
      table, iree_hal_make_indirect_buffer_ref(0, 0, IREE_HAL_WHOLE_BUFFER), 0,
      &pipeline, 0x2000, &cache, &address));
  EXPECT_EQ(address, 0x1000u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_vulkan_command_buffer_resolve_bda_binding(
                            table, iree_hal_make_indirect_buffer_ref(0, 0, 2),
                            0, &pipeline, 0x2000, &cache, &address));
  iree_hal_buffer_release(buffer);
}

TEST(VulkanBdaBindingTest, NarrowedSlotDoesNotOwnNeighbourBytes) {
  iree_hal_buffer_t* buffer = CreateBorrowedBdaBuffer(128);
  iree_hal_buffer_binding_t binding = {buffer, 64, 6};
  iree_hal_buffer_binding_table_t table = {1, &binding};
  iree_hal_vulkan_command_buffer_bda_binding_slot_t slot = {};
  IREE_ASSERT_OK(iree_hal_vulkan_command_buffer_resolve_bda_binding_table_slot(
      table, 0, 0x2000, /*cache=*/nullptr, &slot));
  EXPECT_EQ(slot.device_address, 0x1040u);
  EXPECT_EQ(slot.length, 6u);
  EXPECT_EQ(slot.owned_length, 6u);
  iree_hal_vulkan_bda_binding_requirement_t requirement = {1, 0, 4};
  iree_hal_vulkan_pipeline_t pipeline = {};
  pipeline.bda.binding_requirements = &requirement;
  pipeline.bda.binding_requirement_count = 1;
  VkDeviceAddress address = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_vulkan_command_buffer_resolve_bda_binding(
          table, iree_hal_make_indirect_buffer_ref(0, 0, IREE_HAL_WHOLE_BUFFER),
          0, &pipeline, 0x2000, /*cache=*/nullptr, &address));
  binding.length = 8;
  IREE_ASSERT_OK(iree_hal_vulkan_command_buffer_resolve_bda_binding(
      table, iree_hal_make_indirect_buffer_ref(0, 0, IREE_HAL_WHOLE_BUFFER), 0,
      &pipeline, 0x2000, /*cache=*/nullptr, &address));
  EXPECT_EQ(address, 0x1040u);
  iree_hal_buffer_release(buffer);
}

TEST(VulkanBdaBindingTest, DirectBorrowedViewCannotUseAllocationTail) {
  iree_hal_buffer_t* buffer = CreateBorrowedBdaBuffer(6);
  iree_hal_vulkan_bda_binding_requirement_t requirement = {1, 0, 4};
  iree_hal_vulkan_pipeline_t pipeline = {};
  pipeline.bda.binding_requirements = &requirement;
  pipeline.bda.binding_requirement_count = 1;
  VkDeviceAddress address = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_vulkan_command_buffer_resolve_bda_binding(
          iree_hal_buffer_binding_table_empty(),
          iree_hal_make_buffer_ref(buffer, 0, 6), 0, &pipeline, 0x2000,
          /*cache=*/nullptr, &address));
  // A byte-granular shader can use the exact same view.
  requirement.minimum_access_bytes = 1;
  IREE_ASSERT_OK(iree_hal_vulkan_command_buffer_resolve_bda_binding(
      iree_hal_buffer_binding_table_empty(),
      iree_hal_make_buffer_ref(buffer, 0, 6), 0, &pipeline, 0x2000,
      /*cache=*/nullptr, &address));
  EXPECT_EQ(address, 0x1000u);
  iree_hal_buffer_release(buffer);
}

TEST(VulkanBdaBindingTest, StagedAllocationOwnsItsReservedPadding) {
  iree_hal_buffer_t* backing = CreateBorrowedBdaBuffer(8);
  iree_hal_buffer_t* pending = nullptr;
  iree_hal_buffer_params_t params = {};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
  IREE_ASSERT_OK(iree_hal_local_transient_buffer_create(
      /*placement=*/{}, params, /*allocation_size=*/8, /*byte_length=*/6,
      iree_allocator_system(), &pending));
  iree_hal_local_transient_buffer_stage_backing(pending, backing);
  iree_hal_buffer_release(backing);
  iree_hal_vulkan_bda_binding_requirement_t requirement = {1, 0, 4};
  iree_hal_vulkan_pipeline_t pipeline = {};
  pipeline.bda.binding_requirements = &requirement;
  pipeline.bda.binding_requirement_count = 1;
  VkDeviceAddress address = 0;
  IREE_ASSERT_OK(iree_hal_vulkan_command_buffer_resolve_bda_binding(
      iree_hal_buffer_binding_table_empty(),
      iree_hal_make_buffer_ref(pending, 0, 6), 0, &pipeline, 0x2000,
      /*cache=*/nullptr, &address));
  EXPECT_EQ(address, 0x1000u);
  iree_hal_vulkan_builtins_t builtins = {};
  builtins.max_storage_buffer_range = 128;
  VkDescriptorBufferInfo descriptor = {};
  IREE_ASSERT_OK(iree_hal_vulkan_builtins_resolve_buffer_descriptor(
      &builtins, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4, pending, 0, 6,
      &descriptor));
  EXPECT_EQ(descriptor.range, 8u);
  iree_hal_buffer_release(pending);
}

TEST(VulkanBdaBindingTest, DescriptorRejectsUnalignedBorrowedHandleOffset) {
  iree_hal_buffer_t* buffer = CreateBorrowedBdaBuffer(8, /*handle_offset=*/2);
  iree_hal_vulkan_builtins_t builtins = {};
  builtins.max_storage_buffer_range = 128;
  VkDescriptorBufferInfo descriptor = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_vulkan_builtins_resolve_buffer_descriptor(
                            &builtins, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4,
                            buffer, 0, 8, &descriptor));
  // Byte accesses do not add an alignment requirement at this boundary.
  IREE_ASSERT_OK(iree_hal_vulkan_builtins_resolve_buffer_descriptor(
      &builtins, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, buffer, 0, 8,
      &descriptor));
  EXPECT_EQ(descriptor.offset, 2u);
  EXPECT_EQ(descriptor.range, 8u);
  iree_hal_buffer_release(buffer);
}

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
