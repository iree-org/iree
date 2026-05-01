// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/epoch_signal_table.h"

#include <vector>

#include "iree/testing/gtest.h"

namespace {

// Helper to allocate and initialize a table with the given dimensions.
class EpochSignalTable {
 public:
  EpochSignalTable(uint8_t session_epoch, uint8_t machine_index,
                   uint8_t device_count, uint8_t queue_stride)
      : device_count_(device_count), queue_stride_(queue_stride) {
    iree_host_size_t size =
        iree_hal_amdgpu_epoch_signal_table_size(device_count, queue_stride);
    storage_.resize(size);
    table_ = reinterpret_cast<iree_hal_amdgpu_epoch_signal_table_t*>(
        storage_.data());
    iree_hal_amdgpu_epoch_signal_table_initialize(
        table_, session_epoch, machine_index, device_count, queue_stride);
  }

  iree_hal_amdgpu_epoch_signal_table_t* get() { return table_; }
  const iree_hal_amdgpu_epoch_signal_table_t* get() const { return table_; }

 private:
  uint8_t device_count_;
  uint8_t queue_stride_;
  std::vector<uint8_t> storage_;
  iree_hal_amdgpu_epoch_signal_table_t* table_;
};

// Makes a fake hsa_signal_t with the given handle value. No HSA runtime needed.
static hsa_signal_t make_signal(uint64_t handle) {
  hsa_signal_t signal;
  signal.handle = handle;
  return signal;
}

TEST(EpochSignalTable, SizeComputation) {
  // 1 device, 1 queue: header + 1 signal.
  EXPECT_EQ(
      iree_hal_amdgpu_epoch_signal_table_size(1, 1),
      sizeof(iree_hal_amdgpu_epoch_signal_table_t) + 1 * sizeof(hsa_signal_t));
  // 8 devices, 4 queues: header + 32 signals.
  EXPECT_EQ(
      iree_hal_amdgpu_epoch_signal_table_size(8, 4),
      sizeof(iree_hal_amdgpu_epoch_signal_table_t) + 32 * sizeof(hsa_signal_t));
  // 0 devices: header only (degenerate but valid).
  EXPECT_EQ(iree_hal_amdgpu_epoch_signal_table_size(0, 4),
            sizeof(iree_hal_amdgpu_epoch_signal_table_t));
}

TEST(EpochSignalTable, InitializationZerosAllSlots) {
  EpochSignalTable table(/*session_epoch=*/1, /*machine_index=*/0,
                         /*device_count=*/4, /*queue_stride=*/2);
  // Every slot should be unregistered (handle == 0).
  for (uint8_t device = 0; device < 4; ++device) {
    for (uint8_t queue = 0; queue < 2; ++queue) {
      iree_async_axis_t axis = iree_async_axis_make_queue(1, 0, device, queue);
      hsa_signal_t signal;
      EXPECT_FALSE(iree_hal_amdgpu_epoch_signal_table_lookup(table.get(), axis,
                                                             &signal));
    }
  }
}

TEST(EpochSignalTable, RegisterAndLookup) {
  EpochSignalTable table(/*session_epoch=*/3, /*machine_index=*/7,
                         /*device_count=*/2, /*queue_stride=*/2);

  // Register device 0, queue 1.
  iree_hal_amdgpu_epoch_signal_table_register(table.get(), 0, 1,
                                              make_signal(42));

  // Lookup should succeed with the correct signal.
  iree_async_axis_t axis = iree_async_axis_make_queue(3, 7, 0, 1);
  hsa_signal_t signal;
  EXPECT_TRUE(
      iree_hal_amdgpu_epoch_signal_table_lookup(table.get(), axis, &signal));
  EXPECT_EQ(signal.handle, 42u);
}

TEST(EpochSignalTable, SessionMismatch) {
  EpochSignalTable table(/*session_epoch=*/3, /*machine_index=*/7,
                         /*device_count=*/2, /*queue_stride=*/2);
  iree_hal_amdgpu_epoch_signal_table_register(table.get(), 0, 0,
                                              make_signal(100));

  // Same machine, different session.
  iree_async_axis_t axis = iree_async_axis_make_queue(4, 7, 0, 0);
  hsa_signal_t signal;
  EXPECT_FALSE(
      iree_hal_amdgpu_epoch_signal_table_lookup(table.get(), axis, &signal));
}

TEST(EpochSignalTable, MachineMismatch) {
  EpochSignalTable table(/*session_epoch=*/3, /*machine_index=*/7,
                         /*device_count=*/2, /*queue_stride=*/2);
  iree_hal_amdgpu_epoch_signal_table_register(table.get(), 0, 0,
                                              make_signal(100));

  // Same session, different machine.
  iree_async_axis_t axis = iree_async_axis_make_queue(3, 8, 0, 0);
  hsa_signal_t signal;
  EXPECT_FALSE(
      iree_hal_amdgpu_epoch_signal_table_lookup(table.get(), axis, &signal));
}

TEST(EpochSignalTable, NonQueueDomainRejected) {
  EpochSignalTable table(/*session_epoch=*/1, /*machine_index=*/0,
                         /*device_count=*/4, /*queue_stride=*/4);
  iree_hal_amdgpu_epoch_signal_table_register(table.get(), 0, 0,
                                              make_signal(100));

  // COLLECTIVE domain axis with the same ordinal bits.
  iree_async_axis_t collective_axis =
      iree_async_axis_make(1, 0, IREE_ASYNC_CAUSAL_DOMAIN_COLLECTIVE, 0);
  hsa_signal_t signal;
  EXPECT_FALSE(iree_hal_amdgpu_epoch_signal_table_lookup(
      table.get(), collective_axis, &signal));

  // HOST domain.
  iree_async_axis_t host_axis =
      iree_async_axis_make(1, 0, IREE_ASYNC_CAUSAL_DOMAIN_HOST, 0);
  EXPECT_FALSE(iree_hal_amdgpu_epoch_signal_table_lookup(table.get(), host_axis,
                                                         &signal));
}

TEST(EpochSignalTable, DeviceIndexOutOfBounds) {
  EpochSignalTable table(/*session_epoch=*/1, /*machine_index=*/0,
                         /*device_count=*/2, /*queue_stride=*/2);

  // device_index 2 is out of bounds (only 0 and 1 exist).
  iree_async_axis_t axis = iree_async_axis_make_queue(1, 0, 2, 0);
  hsa_signal_t signal;
  EXPECT_FALSE(
      iree_hal_amdgpu_epoch_signal_table_lookup(table.get(), axis, &signal));
}

TEST(EpochSignalTable, QueueIndexOutOfBounds) {
  EpochSignalTable table(/*session_epoch=*/1, /*machine_index=*/0,
                         /*device_count=*/2, /*queue_stride=*/2);

  // queue_index 2 is out of bounds (stride is 2, so only 0 and 1).
  iree_async_axis_t axis = iree_async_axis_make_queue(1, 0, 0, 2);
  hsa_signal_t signal;
  EXPECT_FALSE(
      iree_hal_amdgpu_epoch_signal_table_lookup(table.get(), axis, &signal));
}

TEST(EpochSignalTable, UnregisteredSlotReturnsFalse) {
  EpochSignalTable table(/*session_epoch=*/1, /*machine_index=*/0,
                         /*device_count=*/4, /*queue_stride=*/4);

  // Register only slot (1, 2).
  iree_hal_amdgpu_epoch_signal_table_register(table.get(), 1, 2,
                                              make_signal(99));

  // Adjacent unregistered slot (1, 3) should fail.
  iree_async_axis_t axis = iree_async_axis_make_queue(1, 0, 1, 3);
  hsa_signal_t signal;
  EXPECT_FALSE(
      iree_hal_amdgpu_epoch_signal_table_lookup(table.get(), axis, &signal));

  // Registered slot (1, 2) should succeed.
  axis = iree_async_axis_make_queue(1, 0, 1, 2);
  EXPECT_TRUE(
      iree_hal_amdgpu_epoch_signal_table_lookup(table.get(), axis, &signal));
  EXPECT_EQ(signal.handle, 99u);
}

TEST(EpochSignalTable, Deregister) {
  EpochSignalTable table(/*session_epoch=*/1, /*machine_index=*/0,
                         /*device_count=*/2, /*queue_stride=*/2);
  iree_hal_amdgpu_epoch_signal_table_register(table.get(), 1, 0,
                                              make_signal(77));

  // Verify registered.
  iree_async_axis_t axis = iree_async_axis_make_queue(1, 0, 1, 0);
  hsa_signal_t signal;
  EXPECT_TRUE(
      iree_hal_amdgpu_epoch_signal_table_lookup(table.get(), axis, &signal));
  EXPECT_EQ(signal.handle, 77u);

  // Deregister.
  iree_hal_amdgpu_epoch_signal_table_deregister(table.get(), 1, 0);

  // Should no longer be found.
  EXPECT_FALSE(
      iree_hal_amdgpu_epoch_signal_table_lookup(table.get(), axis, &signal));
}

TEST(EpochSignalTable, MultiSlotIndependence) {
  // 4 devices × 2 queues = 8 slots. Register unique signals in each.
  EpochSignalTable table(/*session_epoch=*/5, /*machine_index=*/2,
                         /*device_count=*/4, /*queue_stride=*/2);

  for (uint8_t device = 0; device < 4; ++device) {
    for (uint8_t queue = 0; queue < 2; ++queue) {
      uint64_t handle = (uint64_t)device * 100 + queue + 1;
      iree_hal_amdgpu_epoch_signal_table_register(table.get(), device, queue,
                                                  make_signal(handle));
    }
  }

  // Verify each slot returns its unique signal.
  for (uint8_t device = 0; device < 4; ++device) {
    for (uint8_t queue = 0; queue < 2; ++queue) {
      iree_async_axis_t axis = iree_async_axis_make_queue(5, 2, device, queue);
      hsa_signal_t signal;
      ASSERT_TRUE(iree_hal_amdgpu_epoch_signal_table_lookup(table.get(), axis,
                                                            &signal));
      uint64_t expected_handle = (uint64_t)device * 100 + queue + 1;
      EXPECT_EQ(signal.handle, expected_handle);
    }
  }
}

TEST(EpochSignalTable, TPCollectiveJoinPattern) {
  // Simulates the 4-GPU TP collective join: Q0 needs to look up epoch signals
  // for Q1, Q2, Q3 to emit barrier-value packets.
  const uint8_t session = 1;
  const uint8_t machine = 0;
  const uint8_t device_count = 4;
  const uint8_t queues_per_device = 1;

  EpochSignalTable table(session, machine, device_count, queues_per_device);

  // Each device has one queue with a unique epoch signal.
  for (uint8_t device = 0; device < device_count; ++device) {
    iree_hal_amdgpu_epoch_signal_table_register(table.get(), device, 0,
                                                make_signal(1000 + device));
  }

  // Q0 (device 0) needs to wait on Q1, Q2, Q3. Look up their signals.
  for (uint8_t peer = 1; peer < device_count; ++peer) {
    iree_async_axis_t peer_axis =
        iree_async_axis_make_queue(session, machine, peer, 0);
    hsa_signal_t peer_signal;
    ASSERT_TRUE(iree_hal_amdgpu_epoch_signal_table_lookup(
        table.get(), peer_axis, &peer_signal));
    EXPECT_EQ(peer_signal.handle, 1000u + peer);
  }

  // Q0's own signal should also be in the table (for other queues looking it
  // up), though Q0 wouldn't barrier on itself.
  iree_async_axis_t self_axis =
      iree_async_axis_make_queue(session, machine, 0, 0);
  hsa_signal_t self_signal;
  ASSERT_TRUE(iree_hal_amdgpu_epoch_signal_table_lookup(table.get(), self_axis,
                                                        &self_signal));
  EXPECT_EQ(self_signal.handle, 1000u);
}

}  // namespace
