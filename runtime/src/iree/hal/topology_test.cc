// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal {
namespace {

using ::iree::testing::status::IsOk;
using ::iree::testing::status::StatusIs;
using ::testing::Eq;
using ::testing::Ne;

//===----------------------------------------------------------------------===//
// Bitfield overlap tests
//===----------------------------------------------------------------------===//

// Verifies that our bitfields don't overlap and that all fields can be
// independently set without corrupting other fields.
TEST(TopologyEdge, BitfieldOverlap) {
  iree_hal_topology_edge_t edge = 0;

  // Set each field to its maximum value.
  edge = iree_hal_topology_edge_set_wait_mode(
      edge, IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE);
  edge = iree_hal_topology_edge_set_signal_mode(
      edge, IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE);
  edge = iree_hal_topology_edge_set_buffer_read_mode(
      edge, IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE);
  edge = iree_hal_topology_edge_set_buffer_write_mode(
      edge, IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE);
  edge = iree_hal_topology_edge_set_wait_cost(edge, 15);
  edge = iree_hal_topology_edge_set_signal_cost(edge, 15);
  edge = iree_hal_topology_edge_set_copy_cost(edge, 15);
  edge = iree_hal_topology_edge_set_capability_flags(edge, 0x7FF);  // 11 bits
  edge = iree_hal_topology_edge_set_link_class(edge, 7);            // 3 bits

  // Verify all fields retained their values.
  EXPECT_EQ(iree_hal_topology_edge_wait_mode(edge),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE);
  EXPECT_EQ(iree_hal_topology_edge_signal_mode(edge),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE);
  EXPECT_EQ(iree_hal_topology_edge_buffer_read_mode(edge),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE);
  EXPECT_EQ(iree_hal_topology_edge_buffer_write_mode(edge),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE);
  EXPECT_EQ(iree_hal_topology_edge_wait_cost(edge), 15);
  EXPECT_EQ(iree_hal_topology_edge_signal_cost(edge), 15);
  EXPECT_EQ(iree_hal_topology_edge_copy_cost(edge), 15);
  EXPECT_EQ(iree_hal_topology_edge_capability_flags(edge), 0x7FF);
  EXPECT_EQ(iree_hal_topology_edge_link_class(edge), 7);
}

// Verifies that setting each field independently doesn't affect others.
TEST(TopologyEdge, BitfieldIndependence) {
  iree_hal_topology_edge_t edge = 0;

  // Set wait mode and verify only it changes.
  edge = iree_hal_topology_edge_set_wait_mode(
      edge, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  EXPECT_EQ(iree_hal_topology_edge_wait_mode(edge),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  EXPECT_EQ(iree_hal_topology_edge_signal_mode(edge),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);

  // Set signal cost and verify wait mode unchanged (costs are 4-bit, max 15).
  edge = iree_hal_topology_edge_set_signal_cost(edge, 13);
  EXPECT_EQ(iree_hal_topology_edge_signal_cost(edge), 13);
  EXPECT_EQ(iree_hal_topology_edge_wait_mode(edge),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
}

//===----------------------------------------------------------------------===//
// Edge construction tests
//===----------------------------------------------------------------------===//

// Tests creation of a self-edge.
TEST(TopologyEdge, CreateSelf) {
  iree_hal_topology_edge_t edge = iree_hal_topology_edge_make_self();

  // Self-edges should have NATIVE mode for all operations.
  EXPECT_EQ(iree_hal_topology_edge_wait_mode(edge),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  EXPECT_EQ(iree_hal_topology_edge_signal_mode(edge),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  EXPECT_EQ(iree_hal_topology_edge_buffer_read_mode(edge),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  EXPECT_EQ(iree_hal_topology_edge_buffer_write_mode(edge),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);

  // Self-edges should have zero cost.
  EXPECT_EQ(iree_hal_topology_edge_wait_cost(edge), 0);
  EXPECT_EQ(iree_hal_topology_edge_signal_cost(edge), 0);
  EXPECT_EQ(iree_hal_topology_edge_copy_cost(edge), 0);
  EXPECT_EQ(iree_hal_topology_edge_latency_class(edge), 0);
  EXPECT_EQ(iree_hal_topology_edge_numa_distance(edge), 0);

  // Self-edges have all capability flags set.
  iree_hal_topology_capability_t expected_caps =
      IREE_HAL_TOPOLOGY_CAPABILITY_SAME_RUNTIME_DOMAIN |
      IREE_HAL_TOPOLOGY_CAPABILITY_UNIFIED_MEMORY |
      IREE_HAL_TOPOLOGY_CAPABILITY_PEER_COHERENT |
      IREE_HAL_TOPOLOGY_CAPABILITY_HOST_COHERENT |
      IREE_HAL_TOPOLOGY_CAPABILITY_P2P_COPY |
      IREE_HAL_TOPOLOGY_CAPABILITY_CONCURRENT_SAFE |
      IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_DEVICE |
      IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_SYSTEM |
      IREE_HAL_TOPOLOGY_CAPABILITY_TIMELINE_SEMAPHORE;
  EXPECT_EQ(iree_hal_topology_edge_capability_flags(edge), expected_caps);

  // Self-edges use SAME_DIE link class.
  EXPECT_EQ(iree_hal_topology_edge_link_class(edge),
            IREE_HAL_TOPOLOGY_LINK_CLASS_SAME_DIE);
}

// Tests creation of a cross-driver edge.
TEST(TopologyEdge, CreateCrossDriver) {
  iree_hal_topology_edge_t edge = iree_hal_topology_edge_make_cross_driver();

  // Cross-driver edges use IMPORT for semaphores, COPY for buffers.
  EXPECT_EQ(iree_hal_topology_edge_wait_mode(edge),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT);
  EXPECT_EQ(iree_hal_topology_edge_signal_mode(edge),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT);
  EXPECT_EQ(iree_hal_topology_edge_buffer_read_mode(edge),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  EXPECT_EQ(iree_hal_topology_edge_buffer_write_mode(edge),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);

  // Cross-driver has moderate costs.
  EXPECT_EQ(iree_hal_topology_edge_wait_cost(edge), 5);
  EXPECT_EQ(iree_hal_topology_edge_signal_cost(edge), 5);
  EXPECT_EQ(iree_hal_topology_edge_copy_cost(edge), 10);
  EXPECT_EQ(iree_hal_topology_edge_latency_class(edge), 8);
  EXPECT_EQ(iree_hal_topology_edge_numa_distance(edge), 2);

  // No special capabilities for cross-driver.
  EXPECT_EQ(iree_hal_topology_edge_capability_flags(edge),
            IREE_HAL_TOPOLOGY_CAPABILITY_NONE);

  // Cross-driver uses PCIe link class by default.
  EXPECT_EQ(iree_hal_topology_edge_link_class(edge),
            IREE_HAL_TOPOLOGY_LINK_CLASS_PCIE_SAME_ROOT);
}

//===----------------------------------------------------------------------===//
// Resource origin tests
//===----------------------------------------------------------------------===//

// Tests resource origin initialization.
TEST(ResourceOrigin, Initialize) {
  iree_hal_topology_edge_t self_edge = iree_hal_topology_edge_make_self();

  iree_hal_resource_origin_t origin = {
      .self_edge = self_edge,
      .topology_index = 3,
  };

  EXPECT_EQ(origin.self_edge, self_edge);
  EXPECT_EQ(origin.topology_index, 3);

  // Check size is as expected (16 bytes with padding).
  EXPECT_EQ(sizeof(iree_hal_resource_origin_t), 16);
}

// Tests compatibility checking between resources.
TEST(ResourceOrigin, CompatibilityCheck) {
  iree_hal_topology_edge_t edge1 = iree_hal_topology_edge_make_self();
  iree_hal_topology_edge_t edge2 = iree_hal_topology_edge_make_cross_driver();
  edge2 = iree_hal_topology_edge_set_capability_flags(edge2, 0x42);

  iree_hal_resource_origin_t origin1 = {
      .self_edge = edge1,
      .topology_index = 0,
  };
  iree_hal_resource_origin_t origin2 = {
      .self_edge = edge2,
      .topology_index = 1,
  };

  // Self-edges should be different.
  EXPECT_NE(origin1.self_edge, origin2.self_edge);

  // Can check compatibility by comparing capabilities.
  EXPECT_NE(iree_hal_topology_edge_capability_flags(origin1.self_edge),
            iree_hal_topology_edge_capability_flags(origin2.self_edge));
}

//===----------------------------------------------------------------------===//
// Edge formatting tests
//===----------------------------------------------------------------------===//

// Tests edge formatting for debugging.
TEST(TopologyEdge, Formatting) {
  iree_hal_topology_edge_t edge = iree_hal_topology_edge_make_self();

  iree_string_builder_t sb;
  iree_string_builder_initialize(iree_allocator_system(), &sb);
  IREE_ASSERT_OK(iree_hal_topology_edge_format(edge, &sb));
  const char* buffer = iree_string_builder_buffer(&sb);

  // Should contain mode information.
  EXPECT_NE(std::strstr(buffer, "NATIVE"), nullptr);

  // Test cross-driver edge formatting.
  edge = iree_hal_topology_edge_make_cross_driver();
  edge = iree_hal_topology_edge_set_wait_mode(
      edge, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  edge = iree_hal_topology_edge_set_copy_cost(edge, 13);  // 4-bit, max 15

  iree_string_builder_reset(&sb);
  IREE_ASSERT_OK(iree_hal_topology_edge_format(edge, &sb));
  buffer = iree_string_builder_buffer(&sb);

  // Should contain copy mode and cost.
  EXPECT_NE(std::strstr(buffer, "COPY"), nullptr);
  EXPECT_NE(std::strstr(buffer, "copy_cost=13"), nullptr);

  iree_string_builder_deinitialize(&sb);
}

// Tests topology matrix formatting.
TEST(Topology, MatrixFormatting) {
  iree_hal_topology_builder_t builder;
  iree_hal_topology_builder_initialize(&builder, 3);

  // Set cross-device edges (self-edges already initialized).
  for (uint32_t i = 0; i < 3; ++i) {
    for (uint32_t j = 0; j < 3; ++j) {
      if (i != j) {
        iree_hal_topology_edge_t edge =
            iree_hal_topology_edge_make_cross_driver();
        edge = iree_hal_topology_edge_set_link_class(
            edge, IREE_HAL_TOPOLOGY_LINK_CLASS_NVLINK_IF);
        IREE_ASSERT_OK(
            iree_hal_topology_builder_set_edge(&builder, i, j, edge));
      }
    }
  }

  iree_hal_topology_t topology;
  IREE_ASSERT_OK(iree_hal_topology_builder_finalize(&builder, &topology));

  // Dump the matrix for debugging.
  iree_string_builder_t sb;
  iree_string_builder_initialize(iree_allocator_system(), &sb);
  IREE_ASSERT_OK(iree_hal_topology_dump_matrix(&topology, &sb));
  printf("%.*s\n", (int)iree_string_builder_size(&sb),
         iree_string_builder_buffer(&sb));
  iree_string_builder_deinitialize(&sb);
}

}  // namespace
}  // namespace iree::hal
