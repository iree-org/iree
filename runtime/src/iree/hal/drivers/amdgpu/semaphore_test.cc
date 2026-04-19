// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/semaphore.h"

#include <string.h>

#include <vector>

#include "iree/async/frontier.h"
#include "iree/async/proactor_platform.h"
#include "iree/hal/drivers/amdgpu/host_queue_policy.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/system.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static iree_async_proactor_t* test_proactor() {
  static iree_async_proactor_t* proactor = nullptr;
  if (!proactor) {
    IREE_CHECK_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor));
    atexit([] {
      iree_async_proactor_release(proactor);
      proactor = nullptr;
    });
  }
  return proactor;
}

static iree_async_axis_t test_queue_axis(uint8_t queue_index) {
  return iree_async_axis_make_queue(/*session_epoch=*/1, /*machine_index=*/0,
                                    /*device_index=*/0, queue_index);
}

class FrontierBuilder {
 public:
  iree_async_frontier_t* Build(
      std::initializer_list<iree_async_frontier_entry_t> entries) {
    storage_.resize(sizeof(iree_async_frontier_t) +
                    entries.size() * sizeof(iree_async_frontier_entry_t));
    auto* frontier = reinterpret_cast<iree_async_frontier_t*>(storage_.data());
    iree_async_frontier_initialize(frontier,
                                   static_cast<uint8_t>(entries.size()));
    iree_host_size_t i = 0;
    for (const auto& entry : entries) {
      frontier->entries[i++] = entry;
    }
    return frontier;
  }

 private:
  std::vector<uint8_t> storage_;
};

class SemaphoreTest : public ::testing::Test {
 protected:
  void SetUp() override {
    static uintptr_t fake_device_storage = 0;
    fake_device_ = reinterpret_cast<iree_hal_amdgpu_logical_device_t*>(
        &fake_device_storage);
    IREE_ASSERT_OK(CreateSemaphore(IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore_));
  }

  void TearDown() override { iree_hal_semaphore_release(semaphore_); }

  iree_status_t CreateSemaphore(iree_hal_semaphore_flags_t flags,
                                iree_hal_semaphore_t** out_semaphore) {
    return iree_hal_amdgpu_semaphore_create(
        fake_device_, test_proactor(), IREE_HAL_QUEUE_AFFINITY_ANY,
        /*initial_value=*/0, flags, iree_allocator_system(), out_semaphore);
  }

  iree_hal_amdgpu_logical_device_t* fake_device_ = nullptr;
  iree_hal_semaphore_t* semaphore_ = nullptr;
};

TEST_F(SemaphoreTest, PrivateStreamSemanticsRequireStrictFlags) {
  iree_hal_semaphore_t* private_semaphore = nullptr;
  IREE_ASSERT_OK(CreateSemaphore(IREE_HAL_SEMAPHORE_FLAG_DEVICE_LOCAL |
                                     IREE_HAL_SEMAPHORE_FLAG_SINGLE_PRODUCER,
                                 &private_semaphore));
  EXPECT_TRUE(iree_hal_amdgpu_semaphore_has_private_stream_semantics(
      private_semaphore, fake_device_));
  iree_hal_semaphore_release(private_semaphore);

  iree_hal_semaphore_t* public_local_semaphore = nullptr;
  IREE_ASSERT_OK(CreateSemaphore(IREE_HAL_SEMAPHORE_FLAG_DEFAULT |
                                     IREE_HAL_SEMAPHORE_FLAG_DEVICE_LOCAL |
                                     IREE_HAL_SEMAPHORE_FLAG_SINGLE_PRODUCER,
                                 &public_local_semaphore));
  EXPECT_FALSE(iree_hal_amdgpu_semaphore_has_private_stream_semantics(
      public_local_semaphore, fake_device_));
  iree_hal_semaphore_release(public_local_semaphore);

  iree_hal_semaphore_t* multi_producer_semaphore = nullptr;
  IREE_ASSERT_OK(CreateSemaphore(IREE_HAL_SEMAPHORE_FLAG_DEVICE_LOCAL,
                                 &multi_producer_semaphore));
  EXPECT_FALSE(iree_hal_amdgpu_semaphore_has_private_stream_semantics(
      multi_producer_semaphore, fake_device_));
  iree_hal_semaphore_release(multi_producer_semaphore);
}

TEST_F(SemaphoreTest, QueuePolicyUsesAgentScopeOnlyForSamePhysicalDevice) {
  iree_hal_amdgpu_system_t system;
  memset(&system, 0, sizeof(system));
  system.topology.gpu_agent_queue_count = 2;

  iree_hal_amdgpu_logical_device_t logical_device;
  memset(&logical_device, 0, sizeof(logical_device));
  logical_device.system = &system;
  logical_device.physical_device_count = 2;
  logical_device.queue_affinity_mask = 0xFull;

  iree_hal_amdgpu_host_queue_t queue;
  memset(&queue, 0, sizeof(queue));
  queue.logical_device = (iree_hal_device_t*)&logical_device;
  queue.device_ordinal = 0;

  iree_hal_semaphore_t* same_agent_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_semaphore_create(
      &logical_device, test_proactor(), /*queue_affinity=*/0x3ull,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEVICE_LOCAL,
      iree_allocator_system(), &same_agent_semaphore));
  EXPECT_EQ(iree_hal_amdgpu_host_queue_wait_acquire_scope(&queue,
                                                          same_agent_semaphore),
            IREE_HSA_FENCE_SCOPE_AGENT);
  EXPECT_EQ(iree_hal_amdgpu_host_queue_signal_release_scope(
                &queue, same_agent_semaphore),
            IREE_HSA_FENCE_SCOPE_AGENT);
  iree_hal_semaphore_release(same_agent_semaphore);

  iree_hal_semaphore_t* cross_agent_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_semaphore_create(
      &logical_device, test_proactor(), /*queue_affinity=*/0x4ull,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEVICE_LOCAL,
      iree_allocator_system(), &cross_agent_semaphore));
  EXPECT_EQ(iree_hal_amdgpu_host_queue_wait_acquire_scope(
                &queue, cross_agent_semaphore),
            IREE_HSA_FENCE_SCOPE_SYSTEM);
  EXPECT_EQ(iree_hal_amdgpu_host_queue_signal_release_scope(
                &queue, cross_agent_semaphore),
            IREE_HSA_FENCE_SCOPE_SYSTEM);
  iree_hal_semaphore_release(cross_agent_semaphore);

  iree_hal_semaphore_t* public_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_semaphore_create(
      &logical_device, test_proactor(), /*queue_affinity=*/0x1ull,
      /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEVICE_LOCAL |
          IREE_HAL_SEMAPHORE_FLAG_HOST_INTERRUPT,
      iree_allocator_system(), &public_semaphore));
  EXPECT_EQ(
      iree_hal_amdgpu_host_queue_wait_acquire_scope(&queue, public_semaphore),
      IREE_HSA_FENCE_SCOPE_SYSTEM);
  EXPECT_EQ(
      iree_hal_amdgpu_host_queue_signal_release_scope(&queue, public_semaphore),
      IREE_HSA_FENCE_SCOPE_SYSTEM);
  iree_hal_semaphore_release(public_semaphore);
}

TEST_F(SemaphoreTest, PrivateStreamSignalPublishesExactProducerEpoch) {
  iree_hal_semaphore_t* private_semaphore = nullptr;
  IREE_ASSERT_OK(CreateSemaphore(IREE_HAL_SEMAPHORE_FLAG_DEVICE_LOCAL |
                                     IREE_HAL_SEMAPHORE_FLAG_SINGLE_PRODUCER,
                                 &private_semaphore));

  const iree_async_axis_t producer_axis = test_queue_axis(2);
  iree_hal_amdgpu_semaphore_publish_private_stream_signal(
      private_semaphore, producer_axis, /*producer_epoch=*/7,
      /*producer_value=*/3);

  iree_hal_amdgpu_last_signal_flags_t flags =
      IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_NONE;
  iree_async_axis_t cached_axis = 0;
  uint64_t cached_epoch = 0;
  uint64_t cached_value = 0;
  EXPECT_TRUE(iree_hal_amdgpu_last_signal_load(
      iree_hal_amdgpu_semaphore_last_signal(private_semaphore), &flags,
      &cached_axis, &cached_epoch, &cached_value));
  EXPECT_EQ(cached_axis, producer_axis);
  EXPECT_EQ(cached_epoch, 7u);
  EXPECT_EQ(cached_value, 3u);
  EXPECT_EQ(flags,
            IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_VALID |
                IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_PRODUCER_FRONTIER_EXACT);

  iree_hal_semaphore_release(private_semaphore);
}

TEST_F(SemaphoreTest,
       PublishSignalMarksExactWhenProducerFrontierCoversTransitiveDeps) {
  const iree_async_axis_t producer_axis = test_queue_axis(2);
  const iree_async_axis_t peer_axis = test_queue_axis(1);

  FrontierBuilder frontier_builder;
  iree_async_frontier_t* initial_frontier =
      frontier_builder.Build({iree_async_frontier_entry_t{peer_axis, 4}});
  EXPECT_TRUE(iree_hal_amdgpu_semaphore_publish_signal(
      semaphore_, peer_axis, initial_frontier, /*producer_epoch=*/4,
      /*producer_value=*/1));

  iree_async_frontier_t* transitive_frontier =
      frontier_builder.Build({iree_async_frontier_entry_t{peer_axis, 4},
                              iree_async_frontier_entry_t{producer_axis, 7}});
  EXPECT_TRUE(iree_hal_amdgpu_semaphore_publish_signal(
      semaphore_, producer_axis, transitive_frontier, /*producer_epoch=*/7,
      /*producer_value=*/2));

  iree_hal_amdgpu_last_signal_flags_t flags =
      IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_NONE;
  iree_async_axis_t cached_axis = 0;
  uint64_t cached_epoch = 0;
  uint64_t cached_value = 0;
  EXPECT_TRUE(iree_hal_amdgpu_last_signal_load(
      iree_hal_amdgpu_semaphore_last_signal(semaphore_), &flags, &cached_axis,
      &cached_epoch, &cached_value));
  EXPECT_EQ(cached_axis, producer_axis);
  EXPECT_EQ(cached_epoch, 7u);
  EXPECT_EQ(cached_value, 2u);
  EXPECT_EQ(flags,
            IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_VALID |
                IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_PRODUCER_FRONTIER_EXACT);
}

TEST_F(SemaphoreTest, PublishSignalClearsExactForIndependentFanIn) {
  const iree_async_axis_t first_axis = test_queue_axis(1);
  const iree_async_axis_t second_axis = test_queue_axis(2);

  FrontierBuilder frontier_builder;
  iree_async_frontier_t* first_frontier =
      frontier_builder.Build({iree_async_frontier_entry_t{first_axis, 5}});
  EXPECT_TRUE(iree_hal_amdgpu_semaphore_publish_signal(
      semaphore_, first_axis, first_frontier, /*producer_epoch=*/5,
      /*producer_value=*/1));

  iree_async_frontier_t* second_frontier =
      frontier_builder.Build({iree_async_frontier_entry_t{second_axis, 9}});
  EXPECT_TRUE(iree_hal_amdgpu_semaphore_publish_signal(
      semaphore_, second_axis, second_frontier, /*producer_epoch=*/9,
      /*producer_value=*/2));

  iree_hal_amdgpu_last_signal_flags_t flags =
      IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_NONE;
  iree_async_axis_t cached_axis = 0;
  uint64_t cached_epoch = 0;
  uint64_t cached_value = 0;
  EXPECT_TRUE(iree_hal_amdgpu_last_signal_load(
      iree_hal_amdgpu_semaphore_last_signal(semaphore_), &flags, &cached_axis,
      &cached_epoch, &cached_value));
  EXPECT_EQ(cached_axis, second_axis);
  EXPECT_EQ(cached_epoch, 9u);
  EXPECT_EQ(cached_value, 2u);
  EXPECT_EQ(flags, IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_VALID);
}

}  // namespace
