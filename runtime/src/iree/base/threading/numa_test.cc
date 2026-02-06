// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/threading/numa.h"

#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(NumaTest, NodeCountAtLeastOne) { EXPECT_GE(iree_numa_node_count(), 1); }

TEST(NumaTest, OnlineNodesBitmapValid) {
  iree_bitmap_t bitmap = iree_numa_online_nodes();
  // The bitmap should have IREE_NUMA_MAX_NODES bits.
  EXPECT_EQ(bitmap.bit_count, (iree_host_size_t)IREE_NUMA_MAX_NODES);
  // The storage should not be NULL.
  EXPECT_NE(bitmap.words, nullptr);
  // At least one node should be online.
  EXPECT_TRUE(iree_bitmap_any_set(bitmap));
  // Count of online nodes should match iree_numa_node_count().
  EXPECT_EQ(iree_bitmap_count(bitmap), iree_numa_node_count());
}

TEST(NumaTest, OnlineNodesConsistentAcrossCalls) {
  iree_bitmap_t bitmap1 = iree_numa_online_nodes();
  iree_bitmap_t bitmap2 = iree_numa_online_nodes();
  // Should return the same storage (static).
  EXPECT_EQ(bitmap1.words, bitmap2.words);
  EXPECT_EQ(bitmap1.bit_count, bitmap2.bit_count);
}

TEST(NumaTest, CurrentNodeValid) {
  iree_numa_node_id_t node = iree_numa_node_for_current_thread();
  EXPECT_LT(node, (iree_numa_node_id_t)iree_numa_node_count());
  // The current node should be in the online nodes bitmap.
  iree_bitmap_t bitmap = iree_numa_online_nodes();
  EXPECT_TRUE(iree_bitmap_test(bitmap, node));
}

TEST(NumaTest, AllocFreeBasic) {
  iree_numa_alloc_options_t options = iree_numa_alloc_options_default();
  void* ptr = NULL;
  iree_numa_alloc_info_t info;
  IREE_ASSERT_OK(iree_numa_alloc(4096, &options, &ptr, &info));
  ASSERT_NE(ptr, nullptr);
  EXPECT_GE(info.allocated_size, 4096u);

  // Touch memory to verify it's accessible.
  memset(ptr, 0xAB, 4096);

  iree_numa_free(ptr, &info);
}

TEST(NumaTest, AllocFreeZeroNodeId) {
  iree_numa_alloc_options_t options = iree_numa_alloc_options_default();
  options.node_id = 0;  // First NUMA node (always exists).
  void* ptr = NULL;
  iree_numa_alloc_info_t info;
  IREE_ASSERT_OK(iree_numa_alloc(4096, &options, &ptr, &info));
  ASSERT_NE(ptr, nullptr);
  EXPECT_GE(info.allocated_size, 4096u);

  // Touch memory.
  memset(ptr, 0xCD, 4096);

  iree_numa_free(ptr, &info);
}

TEST(NumaTest, AllocFreeCurrentNode) {
  iree_numa_alloc_options_t options = iree_numa_alloc_options_default();
  options.node_id = iree_numa_node_for_current_thread();
  void* ptr = NULL;
  iree_numa_alloc_info_t info;
  IREE_ASSERT_OK(iree_numa_alloc(65536, &options, &ptr, &info));
  ASSERT_NE(ptr, nullptr);
  EXPECT_GE(info.allocated_size, 65536u);

  iree_numa_free(ptr, &info);
}

TEST(NumaTest, AllocFreeLargeSize) {
  iree_numa_alloc_options_t options = iree_numa_alloc_options_default();
  // Allocate 4MB — large enough to exercise huge page paths on systems
  // that have them configured, but small enough to succeed anywhere.
  iree_host_size_t size = 4 * 1024 * 1024;
  void* ptr = NULL;
  iree_numa_alloc_info_t info;
  IREE_ASSERT_OK(iree_numa_alloc(size, &options, &ptr, &info));
  ASSERT_NE(ptr, nullptr);
  EXPECT_GE(info.allocated_size, size);

  // Write first and last page to verify accessibility.
  memset(ptr, 0x11, 4096);
  memset((uint8_t*)ptr + size - 4096, 0x22, 4096);

  iree_numa_free(ptr, &info);
}

TEST(NumaTest, AllocFreeNullIsNoOp) {
  // Freeing NULL should be a no-op.
  iree_numa_alloc_info_t info = {};
  iree_numa_free(NULL, &info);
}

TEST(NumaTest, BindMemoryBasic) {
  // Should succeed or return PERMISSION_DENIED on systems where mbind()
  // requires CAP_SYS_NICE or cgroup permissions (common in CI containers).
  // UNIMPLEMENTED is returned when mbind() is not available (e.g. RISC-V
  // QEMU emulation, minimal kernels).
  void* ptr = NULL;
  iree_numa_alloc_options_t options = iree_numa_alloc_options_default();
  iree_numa_alloc_info_t info;
  IREE_ASSERT_OK(iree_numa_alloc(4096, &options, &ptr, &info));
  ASSERT_NE(ptr, nullptr);

  iree_status_t status = iree_numa_bind_memory(ptr, 4096, 0);
  if (iree_status_code(status) == IREE_STATUS_PERMISSION_DENIED ||
      iree_status_code(status) == IREE_STATUS_UNIMPLEMENTED) {
    iree_status_ignore(status);
  } else {
    IREE_ASSERT_OK(status);
  }

  status = iree_numa_bind_memory(ptr, 4096, IREE_NUMA_NODE_ANY);
  if (iree_status_code(status) == IREE_STATUS_PERMISSION_DENIED ||
      iree_status_code(status) == IREE_STATUS_UNIMPLEMENTED) {
    iree_status_ignore(status);
  } else {
    IREE_ASSERT_OK(status);
  }

  iree_numa_free(ptr, &info);
}

TEST(NumaTest, BindMemoryNullIsNoOp) {
  IREE_EXPECT_OK(iree_numa_bind_memory(NULL, 4096, 0));
}

TEST(NumaTest, DefaultOptionsHaveNoPreference) {
  iree_numa_alloc_options_t options = iree_numa_alloc_options_default();
  EXPECT_EQ(options.node_id, IREE_NUMA_NODE_ANY);
  EXPECT_EQ(options.alignment, 0u);
  EXPECT_EQ(options.huge_page_size, 0u);
  EXPECT_FALSE(options.use_explicit_huge_pages);
  EXPECT_FALSE(options.hint_transparent_huge_pages);
}

TEST(NumaTest, TransparentHugePageHint) {
  iree_numa_alloc_options_t options = iree_numa_alloc_options_default();
  options.hint_transparent_huge_pages = true;
  // Allocate 4MB (2x the typical huge page size).
  iree_host_size_t size = 4 * 1024 * 1024;
  void* ptr = NULL;
  iree_numa_alloc_info_t info;
  IREE_ASSERT_OK(iree_numa_alloc(size, &options, &ptr, &info));
  ASSERT_NE(ptr, nullptr);
  EXPECT_GE(info.allocated_size, size);

  // On Linux, we expect THP method; on other platforms, standard or mmap.
  // Don't assert the specific method — the test validates the allocation
  // succeeds regardless of platform support.

  iree_numa_free(ptr, &info);
}

}  // namespace
