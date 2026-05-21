// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/queue_upload_ring.h"

#include <cstdint>

#include "iree/testing/gtest.h"

namespace {

static void InitializeTestRing(uint8_t* storage, uint32_t capacity,
                               iree_hal_amdgpu_queue_upload_ring_t* out_ring) {
  out_ring->base = storage;
  out_ring->device_base = 0x10000000ull;
  out_ring->capacity = capacity;
  out_ring->mask = capacity - 1;
  iree_atomic_store(&out_ring->write_position, 0, iree_memory_order_relaxed);
  iree_atomic_store(&out_ring->read_position, 0, iree_memory_order_relaxed);
}

TEST(QueueUploadRingTest, AllocatesAlignedSpansAndReclaims) {
  alignas(64) uint8_t storage[64] = {};
  iree_hal_amdgpu_queue_upload_ring_t ring = {};
  InitializeTestRing(storage, IREE_ARRAYSIZE(storage), &ring);

  EXPECT_TRUE(iree_hal_amdgpu_queue_upload_ring_can_allocate(
      &ring, /*length=*/7, /*alignment=*/1));
  iree_hal_amdgpu_queue_upload_span_t first =
      iree_hal_amdgpu_queue_upload_ring_allocate(&ring, /*length=*/7,
                                                 /*alignment=*/1);
  EXPECT_EQ(first.host_ptr, &storage[0]);
  EXPECT_EQ(first.device_ptr, 0x10000000ull);
  EXPECT_EQ(first.length, 7u);
  EXPECT_EQ(first.end_position, 7u);

  iree_hal_amdgpu_queue_upload_span_t second =
      iree_hal_amdgpu_queue_upload_ring_allocate(&ring, /*length=*/8,
                                                 /*alignment=*/8);
  EXPECT_EQ(second.host_ptr, &storage[8]);
  EXPECT_EQ(second.device_ptr, 0x10000008ull);
  EXPECT_EQ(second.length, 8u);
  EXPECT_EQ(second.end_position, 16u);

  iree_hal_amdgpu_queue_upload_ring_reclaim(&ring, second.end_position);
  EXPECT_TRUE(iree_hal_amdgpu_queue_upload_ring_can_allocate(
      &ring, /*length=*/16, /*alignment=*/32));

  iree_hal_amdgpu_queue_upload_span_t wrapped =
      iree_hal_amdgpu_queue_upload_ring_allocate(&ring, /*length=*/16,
                                                 /*alignment=*/32);
  EXPECT_EQ(wrapped.host_ptr, &storage[32]);
  EXPECT_EQ(wrapped.device_ptr, 0x10000020ull);
  EXPECT_EQ(wrapped.length, 16u);
  EXPECT_EQ(wrapped.end_position, 48u);
}

TEST(QueueUploadRingTest, SkipsTailWhenAlignedSpanWouldWrap) {
  alignas(64) uint8_t storage[64] = {};
  iree_hal_amdgpu_queue_upload_ring_t ring = {};
  InitializeTestRing(storage, IREE_ARRAYSIZE(storage), &ring);

  iree_hal_amdgpu_queue_upload_span_t first =
      iree_hal_amdgpu_queue_upload_ring_allocate(&ring, /*length=*/48,
                                                 /*alignment=*/1);
  ASSERT_NE(first.host_ptr, nullptr);
  iree_hal_amdgpu_queue_upload_ring_reclaim(&ring, first.end_position);

  iree_hal_amdgpu_queue_upload_span_t wrapped =
      iree_hal_amdgpu_queue_upload_ring_allocate(&ring, /*length=*/24,
                                                 /*alignment=*/32);
  EXPECT_EQ(wrapped.host_ptr, &storage[0]);
  EXPECT_EQ(wrapped.device_ptr, 0x10000000ull);
  EXPECT_EQ(wrapped.length, 24u);
  EXPECT_EQ(wrapped.end_position, 88u);
}

TEST(QueueUploadRingTest, RejectsInvalidRequests) {
  alignas(64) uint8_t storage[32] = {};
  iree_hal_amdgpu_queue_upload_ring_t ring = {};
  InitializeTestRing(storage, IREE_ARRAYSIZE(storage), &ring);

  EXPECT_FALSE(iree_hal_amdgpu_queue_upload_ring_can_allocate(
      &ring, /*length=*/0, /*alignment=*/1));
  EXPECT_FALSE(iree_hal_amdgpu_queue_upload_ring_can_allocate(
      &ring, /*length=*/33, /*alignment=*/1));
  EXPECT_FALSE(iree_hal_amdgpu_queue_upload_ring_can_allocate(
      &ring, /*length=*/8, /*alignment=*/0));
  EXPECT_FALSE(iree_hal_amdgpu_queue_upload_ring_can_allocate(
      &ring, /*length=*/8, /*alignment=*/3));
  EXPECT_FALSE(iree_hal_amdgpu_queue_upload_ring_can_allocate(
      &ring, /*length=*/8, /*alignment=*/64));

  EXPECT_EQ(iree_hal_amdgpu_queue_upload_ring_allocate(&ring, /*length=*/0,
                                                       /*alignment=*/1)
                .host_ptr,
            nullptr);
  EXPECT_EQ(iree_hal_amdgpu_queue_upload_ring_allocate(&ring, /*length=*/8,
                                                       /*alignment=*/3)
                .host_ptr,
            nullptr);
}

TEST(QueueUploadRingTest, PublicationModeNoneSkipsRegisterWrite) {
  volatile uint32_t hdp_mem_flush_control = 0xCAFEu;
  iree_hal_amdgpu_queue_upload_ring_t ring = {};
  ring.publication.mode = IREE_HAL_AMDGPU_KERNARG_RING_PUBLICATION_MODE_NONE;
  ring.publication.hdp_mem_flush_control = &hdp_mem_flush_control;

  iree_hal_amdgpu_queue_upload_ring_publish_host_writes(&ring);

  EXPECT_EQ(hdp_mem_flush_control, 0xCAFEu);
}

TEST(QueueUploadRingTest, PublicationModeHdpFlushWritesRegister) {
  volatile uint32_t hdp_mem_flush_control = 0u;
  iree_hal_amdgpu_queue_upload_ring_t ring = {};
  ring.publication.mode =
      IREE_HAL_AMDGPU_KERNARG_RING_PUBLICATION_MODE_HDP_FLUSH;
  ring.publication.hdp_mem_flush_control = &hdp_mem_flush_control;

  iree_hal_amdgpu_queue_upload_ring_publish_host_writes(&ring);

  EXPECT_EQ(hdp_mem_flush_control, 1u);
}

}  // namespace
