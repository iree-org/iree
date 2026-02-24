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
// Builder tests
//===----------------------------------------------------------------------===//

// Tests basic builder initialization.
TEST(TopologyBuilder, Initialize) {
  iree_hal_topology_builder_t builder;
  iree_hal_topology_builder_initialize(&builder, 4);

  // Verify device count was set.
  EXPECT_EQ(builder.topology.device_count, 4);

  // Verify self-edges were initialized.
  for (uint32_t i = 0; i < 4; ++i) {
    uint32_t idx = i * 4 + i;
    EXPECT_TRUE(builder.edges_set[idx]);
    EXPECT_EQ(iree_hal_topology_edge_wait_mode(builder.topology.edges[idx]),
              IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  }
}

// Tests setting edges in the builder.
TEST(TopologyBuilder, SetEdges) {
  iree_hal_topology_builder_t builder;
  iree_hal_topology_builder_initialize(&builder, 2);

  // Set self-edges (already initialized, but we can re-set them).
  iree_hal_topology_edge_t self_edge = iree_hal_topology_edge_make_self();
  IREE_ASSERT_OK(iree_hal_topology_builder_set_edge(&builder, 0, 0, self_edge));
  IREE_ASSERT_OK(iree_hal_topology_builder_set_edge(&builder, 1, 1, self_edge));

  // Set cross-device edges.
  iree_hal_topology_edge_t cross_edge =
      iree_hal_topology_edge_make_cross_driver();
  cross_edge = iree_hal_topology_edge_set_wait_mode(
      cross_edge, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  cross_edge = iree_hal_topology_edge_set_signal_mode(
      cross_edge, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  cross_edge = iree_hal_topology_edge_set_link_class(
      cross_edge, IREE_HAL_TOPOLOGY_LINK_CLASS_NVLINK_IF);

  IREE_ASSERT_OK(
      iree_hal_topology_builder_set_edge(&builder, 0, 1, cross_edge));
  IREE_ASSERT_OK(
      iree_hal_topology_builder_set_edge(&builder, 1, 0, cross_edge));

  // Build and get topology.
  iree_hal_topology_t topology;
  IREE_ASSERT_OK(iree_hal_topology_builder_finalize(&builder, &topology));

  // Check the topology.
  EXPECT_EQ(topology.device_count, 2);
  EXPECT_EQ(iree_hal_topology_edge_wait_mode(topology.edges[0 * 2 + 0]),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  EXPECT_EQ(iree_hal_topology_edge_wait_mode(topology.edges[0 * 2 + 1]),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
}

// Tests that validation fails when not all edges are set.
TEST(TopologyBuilder, ValidationFailsIncomplete) {
  iree_hal_topology_builder_t builder;
  iree_hal_topology_builder_initialize(&builder, 2);

  // Clear one of the edges_set flags to simulate missing edge.
  // Self-edges [0,0] and [1,1] are set by initialize.
  // Don't set cross-edges [0,1] and [1,0].
  builder.edges_set[0 * 2 + 1] = false;  // mark [0,1] as not set
  builder.edges_set[1 * 2 + 0] = false;  // mark [1,0] as not set

  iree_hal_topology_t topology;
  iree_status_t status =
      iree_hal_topology_builder_finalize(&builder, &topology);
  EXPECT_THAT(status, StatusIs(iree::StatusCode::kInvalidArgument));
  iree_status_ignore(status);
}

// Tests that validation fails for asymmetric link classes.
TEST(TopologyBuilder, ValidationFailsAsymmetricLinks) {
  iree_hal_topology_builder_t builder;
  iree_hal_topology_builder_initialize(&builder, 2);

  // Self-edges already set by initialize.

  // Set asymmetric cross-device edges (different link classes).
  iree_hal_topology_edge_t edge1 = iree_hal_topology_edge_make_cross_driver();
  edge1 = iree_hal_topology_edge_set_link_class(
      edge1, IREE_HAL_TOPOLOGY_LINK_CLASS_NVLINK_IF);

  iree_hal_topology_edge_t edge2 = iree_hal_topology_edge_make_cross_driver();
  edge2 = iree_hal_topology_edge_set_link_class(
      edge2, IREE_HAL_TOPOLOGY_LINK_CLASS_PCIE_SAME_ROOT);  // Different!

  IREE_ASSERT_OK(iree_hal_topology_builder_set_edge(&builder, 0, 1, edge1));
  IREE_ASSERT_OK(iree_hal_topology_builder_set_edge(&builder, 1, 0, edge2));

  iree_hal_topology_t topology;
  iree_status_t status =
      iree_hal_topology_builder_finalize(&builder, &topology);
  EXPECT_THAT(status, StatusIs(iree::StatusCode::kInvalidArgument));
  iree_status_ignore(status);
}

// Tests NUMA node assignment.
TEST(TopologyBuilder, NumaNodes) {
  iree_hal_topology_builder_t builder;
  iree_hal_topology_builder_initialize(&builder, 4);

  // Assign NUMA nodes.
  IREE_ASSERT_OK(iree_hal_topology_builder_set_numa_node(&builder, 0, 0));
  IREE_ASSERT_OK(iree_hal_topology_builder_set_numa_node(&builder, 1, 0));
  IREE_ASSERT_OK(iree_hal_topology_builder_set_numa_node(&builder, 2, 1));
  IREE_ASSERT_OK(iree_hal_topology_builder_set_numa_node(&builder, 3, 1));

  // Set cross-device edges (self-edges already set by initialize).
  for (uint32_t i = 0; i < 4; ++i) {
    for (uint32_t j = 0; j < 4; ++j) {
      if (i != j) {
        iree_hal_topology_edge_t cross =
            iree_hal_topology_edge_make_cross_driver();
        // Same NUMA node = lower cost (costs are 4-bit, max 15).
        uint8_t cost =
            (builder.topology.numa_nodes[i] == builder.topology.numa_nodes[j])
                ? 5
                : 12;
        cross = iree_hal_topology_edge_set_copy_cost(cross, cost);
        IREE_ASSERT_OK(
            iree_hal_topology_builder_set_edge(&builder, i, j, cross));
      }
    }
  }

  iree_hal_topology_t topology;
  IREE_ASSERT_OK(iree_hal_topology_builder_finalize(&builder, &topology));

  // Check NUMA nodes.
  EXPECT_EQ(topology.numa_nodes[0], 0);
  EXPECT_EQ(topology.numa_nodes[1], 0);
  EXPECT_EQ(topology.numa_nodes[2], 1);
  EXPECT_EQ(topology.numa_nodes[3], 1);

  // Check that intra-NUMA transfers have lower cost.
  EXPECT_EQ(iree_hal_topology_edge_copy_cost(topology.edges[0 * 4 + 1]),
            5);  // same NUMA
  EXPECT_EQ(iree_hal_topology_edge_copy_cost(topology.edges[0 * 4 + 2]),
            12);  // different NUMA
}

}  // namespace
}  // namespace iree::hal
