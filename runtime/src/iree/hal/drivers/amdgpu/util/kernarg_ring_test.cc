// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/kernarg_ring.h"

#include <cstdint>

#include "iree/testing/gtest.h"

namespace {

static void InitializeTestRing(iree_hal_amdgpu_kernarg_block_t* storage,
                               uint32_t capacity,
                               iree_hal_amdgpu_kernarg_ring_t* out_ring) {
  out_ring->base = storage;
  out_ring->capacity = capacity;
  out_ring->mask = capacity - 1;
  iree_atomic_store(&out_ring->write_position, 0, iree_memory_order_relaxed);
  iree_atomic_store(&out_ring->read_position, 0, iree_memory_order_relaxed);
}

TEST(KernargRingTest, AllocatesContiguousBlocksAndReclaims) {
  iree_hal_amdgpu_kernarg_block_t storage[8] = {};
  iree_hal_amdgpu_kernarg_ring_t ring = {};
  InitializeTestRing(storage, IREE_ARRAYSIZE(storage), &ring);

  EXPECT_TRUE(iree_hal_amdgpu_kernarg_ring_can_allocate(&ring, 3));

  uint64_t end_position = 0;
  iree_hal_amdgpu_kernarg_block_t* first =
      iree_hal_amdgpu_kernarg_ring_allocate(&ring, 3, &end_position);
  EXPECT_EQ(first, &storage[0]);
  EXPECT_EQ(end_position, 3u);

  iree_hal_amdgpu_kernarg_block_t* second =
      iree_hal_amdgpu_kernarg_ring_allocate(&ring, 4, &end_position);
  EXPECT_EQ(second, &storage[3]);
  EXPECT_EQ(end_position, 7u);

  EXPECT_FALSE(iree_hal_amdgpu_kernarg_ring_can_allocate(&ring, 2));

  iree_hal_amdgpu_kernarg_ring_reclaim(&ring, 3);
  EXPECT_TRUE(iree_hal_amdgpu_kernarg_ring_can_allocate(&ring, 2));

  iree_hal_amdgpu_kernarg_block_t* wrapped =
      iree_hal_amdgpu_kernarg_ring_allocate(&ring, 2, &end_position);
  EXPECT_EQ(wrapped, &storage[0]);
  EXPECT_EQ(end_position, 10u);
}

TEST(KernargRingTest, RejectsInvalidBlockCounts) {
  iree_hal_amdgpu_kernarg_block_t storage[4] = {};
  iree_hal_amdgpu_kernarg_ring_t ring = {};
  InitializeTestRing(storage, IREE_ARRAYSIZE(storage), &ring);

  EXPECT_FALSE(iree_hal_amdgpu_kernarg_ring_can_allocate(&ring, 0));
  EXPECT_FALSE(iree_hal_amdgpu_kernarg_ring_can_allocate(&ring, 5));

  uint64_t end_position = UINT64_MAX;
  EXPECT_EQ(iree_hal_amdgpu_kernarg_ring_allocate(&ring, 0, &end_position),
            nullptr);
  EXPECT_EQ(end_position, 0u);
  end_position = UINT64_MAX;
  EXPECT_EQ(iree_hal_amdgpu_kernarg_ring_allocate(&ring, 5, &end_position),
            nullptr);
  EXPECT_EQ(end_position, 0u);
}

TEST(KernargRingTest, PublicationModeNoneSkipsRegisterWrite) {
  volatile uint32_t hdp_mem_flush_control = 0xCAFEu;
  iree_hal_amdgpu_kernarg_ring_t ring = {};
  ring.publication.mode = IREE_HAL_AMDGPU_KERNARG_RING_PUBLICATION_MODE_NONE;
  ring.publication.hdp_mem_flush_control = &hdp_mem_flush_control;

  iree_hal_amdgpu_kernarg_ring_publish_host_writes(&ring);

  EXPECT_EQ(hdp_mem_flush_control, 0xCAFEu);
}

TEST(KernargRingTest, PublicationModeHdpFlushWritesRegister) {
  volatile uint32_t hdp_mem_flush_control = 0u;
  iree_hal_amdgpu_kernarg_ring_t ring = {};
  ring.publication.mode =
      IREE_HAL_AMDGPU_KERNARG_RING_PUBLICATION_MODE_HDP_FLUSH;
  ring.publication.hdp_mem_flush_control = &hdp_mem_flush_control;

  iree_hal_amdgpu_kernarg_ring_publish_host_writes(&ring);

  EXPECT_EQ(hdp_mem_flush_control, 1u);
}

}  // namespace
