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
// Builder initialization tests
//===----------------------------------------------------------------------===//

// Tests basic builder initialization.
TEST(TopologyBuilder, Initialize) {
  iree_hal_topology_builder_t builder;
  iree_hal_topology_builder_initialize(&builder, 4);

  // Verify device count was set.
  EXPECT_EQ(builder.topology.device_count, 4);

  // Verify self-edges were initialized with optimal scheduling settings.
  for (uint32_t i = 0; i < 4; ++i) {
    uint32_t idx = i * 4 + i;
    EXPECT_TRUE(builder.edges_set[idx]);
    EXPECT_EQ(iree_hal_topology_edge_wait_mode(builder.topology.edges[idx].lo),
              IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
    EXPECT_EQ(
        iree_hal_topology_edge_signal_mode(builder.topology.edges[idx].lo),
        IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
    EXPECT_EQ(iree_hal_topology_edge_link_class(builder.topology.edges[idx].lo),
              IREE_HAL_TOPOLOGY_LINK_CLASS_SAME_DIE);
  }

  // Verify self-edges have native handle types in the interop word.
  for (uint32_t i = 0; i < 4; ++i) {
    uint32_t idx = i * 4 + i;
    EXPECT_EQ(iree_hal_topology_edge_semaphore_import_types(
                  builder.topology.edges[idx].hi),
              IREE_HAL_TOPOLOGY_HANDLE_TYPE_NATIVE);
    EXPECT_EQ(iree_hal_topology_edge_buffer_import_types(
                  builder.topology.edges[idx].hi),
              IREE_HAL_TOPOLOGY_HANDLE_TYPE_NATIVE);
  }
}

//===----------------------------------------------------------------------===//
// Edge setting tests
//===----------------------------------------------------------------------===//

// Tests setting edges in the builder.
TEST(TopologyBuilder, SetEdges) {
  iree_hal_topology_builder_t builder;
  iree_hal_topology_builder_initialize(&builder, 2);

  // Set self-edges (already initialized, but we can re-set them).
  iree_hal_topology_edge_t self_edge = iree_hal_topology_edge_make_self();
  IREE_ASSERT_OK(iree_hal_topology_builder_set_edge(&builder, 0, 0, self_edge));
  IREE_ASSERT_OK(iree_hal_topology_builder_set_edge(&builder, 1, 1, self_edge));

  // Set cross-device edges with custom scheduling and interop settings.
  iree_hal_topology_edge_t cross_edge =
      iree_hal_topology_edge_make_cross_driver();
  cross_edge.lo = iree_hal_topology_edge_set_wait_mode(
      cross_edge.lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  cross_edge.lo = iree_hal_topology_edge_set_signal_mode(
      cross_edge.lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  cross_edge.lo = iree_hal_topology_edge_set_link_class(
      cross_edge.lo, IREE_HAL_TOPOLOGY_LINK_CLASS_NVLINK_IF);

  // Set interop handle types for the cross-device edge.
  cross_edge.hi = iree_hal_topology_edge_set_semaphore_import_types(
      cross_edge.hi, IREE_HAL_TOPOLOGY_HANDLE_TYPE_OPAQUE_FD);
  cross_edge.hi = iree_hal_topology_edge_set_buffer_import_types(
      cross_edge.hi, IREE_HAL_TOPOLOGY_HANDLE_TYPE_DMA_BUF);

  IREE_ASSERT_OK(
      iree_hal_topology_builder_set_edge(&builder, 0, 1, cross_edge));
  IREE_ASSERT_OK(
      iree_hal_topology_builder_set_edge(&builder, 1, 0, cross_edge));

  // Build and get topology.
  iree_hal_topology_t topology;
  IREE_ASSERT_OK(iree_hal_topology_builder_finalize(&builder, &topology));

  // Check the scheduling word in the topology.
  EXPECT_EQ(topology.device_count, 2);
  EXPECT_EQ(iree_hal_topology_edge_wait_mode(topology.edges[0 * 2 + 0].lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  EXPECT_EQ(iree_hal_topology_edge_wait_mode(topology.edges[0 * 2 + 1].lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);

  // Check the interop word survived the finalize copy.
  EXPECT_EQ(iree_hal_topology_edge_semaphore_import_types(
                topology.edges[0 * 2 + 1].hi),
            IREE_HAL_TOPOLOGY_HANDLE_TYPE_OPAQUE_FD);
  EXPECT_EQ(
      iree_hal_topology_edge_buffer_import_types(topology.edges[0 * 2 + 1].hi),
      IREE_HAL_TOPOLOGY_HANDLE_TYPE_DMA_BUF);
}

// Tests that both words of an edge are preserved through set_edge and finalize.
TEST(TopologyBuilder, FullEdgePreservation) {
  iree_hal_topology_builder_t builder;
  iree_hal_topology_builder_initialize(&builder, 2);

  // Build a fully-populated cross-device edge.
  iree_hal_topology_edge_t edge = iree_hal_topology_edge_make_cross_driver();
  edge.lo = iree_hal_topology_edge_set_wait_mode(
      edge.lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT);
  edge.lo = iree_hal_topology_edge_set_signal_mode(
      edge.lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  edge.lo = iree_hal_topology_edge_set_capability_flags(
      edge.lo, IREE_HAL_TOPOLOGY_CAPABILITY_P2P_COPY |
                   IREE_HAL_TOPOLOGY_CAPABILITY_REMOTE_DMA);
  edge.lo = iree_hal_topology_edge_set_wait_cost(edge.lo, 3);
  edge.lo = iree_hal_topology_edge_set_copy_cost(edge.lo, 7);
  edge.lo = iree_hal_topology_edge_set_link_class(
      edge.lo, IREE_HAL_TOPOLOGY_LINK_CLASS_FABRIC);

  edge.hi = iree_hal_topology_edge_set_semaphore_import_types(
      edge.hi, IREE_HAL_TOPOLOGY_HANDLE_TYPE_SHM);
  edge.hi = iree_hal_topology_edge_set_semaphore_export_types(
      edge.hi, IREE_HAL_TOPOLOGY_HANDLE_TYPE_SHM);
  edge.hi = iree_hal_topology_edge_set_buffer_import_types(
      edge.hi, IREE_HAL_TOPOLOGY_HANDLE_TYPE_RDMA_MR |
                   IREE_HAL_TOPOLOGY_HANDLE_TYPE_SHM);
  edge.hi = iree_hal_topology_edge_set_buffer_export_types(
      edge.hi, IREE_HAL_TOPOLOGY_HANDLE_TYPE_RDMA_MR |
                   IREE_HAL_TOPOLOGY_HANDLE_TYPE_SHM);

  IREE_ASSERT_OK(iree_hal_topology_builder_set_edge(&builder, 0, 1, edge));
  IREE_ASSERT_OK(iree_hal_topology_builder_set_edge(&builder, 1, 0, edge));

  iree_hal_topology_t topology;
  IREE_ASSERT_OK(iree_hal_topology_builder_finalize(&builder, &topology));

  // Verify the entire scheduling word survived.
  iree_hal_topology_edge_t result = topology.edges[0 * 2 + 1];
  EXPECT_EQ(iree_hal_topology_edge_wait_mode(result.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT);
  EXPECT_EQ(iree_hal_topology_edge_signal_mode(result.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  EXPECT_EQ(iree_hal_topology_edge_capability_flags(result.lo),
            IREE_HAL_TOPOLOGY_CAPABILITY_P2P_COPY |
                IREE_HAL_TOPOLOGY_CAPABILITY_REMOTE_DMA);
  EXPECT_EQ(iree_hal_topology_edge_wait_cost(result.lo), 3);
  EXPECT_EQ(iree_hal_topology_edge_copy_cost(result.lo), 7);
  EXPECT_EQ(iree_hal_topology_edge_link_class(result.lo),
            IREE_HAL_TOPOLOGY_LINK_CLASS_FABRIC);

  // Verify the entire interop word survived.
  EXPECT_EQ(iree_hal_topology_edge_semaphore_import_types(result.hi),
            IREE_HAL_TOPOLOGY_HANDLE_TYPE_SHM);
  EXPECT_EQ(iree_hal_topology_edge_semaphore_export_types(result.hi),
            IREE_HAL_TOPOLOGY_HANDLE_TYPE_SHM);
  EXPECT_EQ(iree_hal_topology_edge_buffer_import_types(result.hi),
            IREE_HAL_TOPOLOGY_HANDLE_TYPE_RDMA_MR |
                IREE_HAL_TOPOLOGY_HANDLE_TYPE_SHM);
  EXPECT_EQ(iree_hal_topology_edge_buffer_export_types(result.hi),
            IREE_HAL_TOPOLOGY_HANDLE_TYPE_RDMA_MR |
                IREE_HAL_TOPOLOGY_HANDLE_TYPE_SHM);
}

//===----------------------------------------------------------------------===//
// Validation tests
//===----------------------------------------------------------------------===//

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

  // Set asymmetric cross-device edges (different link classes).
  iree_hal_topology_edge_t edge1 = iree_hal_topology_edge_make_cross_driver();
  edge1.lo = iree_hal_topology_edge_set_link_class(
      edge1.lo, IREE_HAL_TOPOLOGY_LINK_CLASS_NVLINK_IF);

  iree_hal_topology_edge_t edge2 = iree_hal_topology_edge_make_cross_driver();
  edge2.lo = iree_hal_topology_edge_set_link_class(
      edge2.lo, IREE_HAL_TOPOLOGY_LINK_CLASS_PCIE_SAME_ROOT);

  IREE_ASSERT_OK(iree_hal_topology_builder_set_edge(&builder, 0, 1, edge1));
  IREE_ASSERT_OK(iree_hal_topology_builder_set_edge(&builder, 1, 0, edge2));

  iree_hal_topology_t topology;
  iree_status_t status =
      iree_hal_topology_builder_finalize(&builder, &topology);
  EXPECT_THAT(status, StatusIs(iree::StatusCode::kInvalidArgument));
  iree_status_ignore(status);
}

// Tests that self-edges must have NATIVE wait mode.
TEST(TopologyBuilder, ValidationFailsNonNativeSelfEdge) {
  iree_hal_topology_builder_t builder;
  iree_hal_topology_builder_initialize(&builder, 1);

  // Try to set a self-edge with non-NATIVE wait mode.
  iree_hal_topology_edge_t bad_self = iree_hal_topology_edge_make_self();
  bad_self.lo = iree_hal_topology_edge_set_wait_mode(
      bad_self.lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);

  iree_status_t status =
      iree_hal_topology_builder_set_edge(&builder, 0, 0, bad_self);
  EXPECT_THAT(status, StatusIs(iree::StatusCode::kInvalidArgument));
  iree_status_ignore(status);
}

//===----------------------------------------------------------------------===//
// NUMA node tests
//===----------------------------------------------------------------------===//

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
        cross.lo = iree_hal_topology_edge_set_copy_cost(cross.lo, cost);
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
  EXPECT_EQ(iree_hal_topology_edge_copy_cost(topology.edges[0 * 4 + 1].lo),
            5);  // same NUMA
  EXPECT_EQ(iree_hal_topology_edge_copy_cost(topology.edges[0 * 4 + 2].lo),
            12);  // different NUMA
}

//===----------------------------------------------------------------------===//
// Topology query tests
//===----------------------------------------------------------------------===//

// Tests querying edges from a finalized topology.
TEST(TopologyBuilder, QueryEdges) {
  iree_hal_topology_builder_t builder;
  iree_hal_topology_builder_initialize(&builder, 3);

  // Set cross-device edges with varying configurations.
  for (uint32_t i = 0; i < 3; ++i) {
    for (uint32_t j = 0; j < 3; ++j) {
      if (i != j) {
        iree_hal_topology_edge_t edge =
            iree_hal_topology_edge_make_cross_driver();
        // Give each edge a unique copy cost for identification.
        edge.lo =
            iree_hal_topology_edge_set_copy_cost(edge.lo, (uint8_t)(i * 3 + j));
        // Set link class consistently for symmetry validation.
        edge.lo = iree_hal_topology_edge_set_link_class(
            edge.lo, IREE_HAL_TOPOLOGY_LINK_CLASS_PCIE_SAME_ROOT);
        IREE_ASSERT_OK(
            iree_hal_topology_builder_set_edge(&builder, i, j, edge));
      }
    }
  }

  iree_hal_topology_t topology;
  IREE_ASSERT_OK(iree_hal_topology_builder_finalize(&builder, &topology));

  // Query each edge and verify the copy cost encodes the (i,j) pair.
  for (uint32_t i = 0; i < 3; ++i) {
    for (uint32_t j = 0; j < 3; ++j) {
      iree_hal_topology_edge_t queried =
          iree_hal_topology_query_edge(&topology, i, j);
      if (i == j) {
        // Self-edges have zero copy cost.
        EXPECT_EQ(iree_hal_topology_edge_copy_cost(queried.lo), 0);
      } else {
        EXPECT_EQ(iree_hal_topology_edge_copy_cost(queried.lo), i * 3 + j);
      }
    }
  }
}

}  // namespace
}  // namespace iree::hal
