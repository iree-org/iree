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
// Scheduling word bitfield overlap tests
//===----------------------------------------------------------------------===//

// Verifies that scheduling word bitfields don't overlap and that all fields
// can be independently set without corrupting other fields.
TEST(TopologyEdge, SchedulingWordBitfieldOverlap) {
  iree_hal_topology_edge_scheduling_word_t lo = 0;

  // Set each field to its maximum value.
  lo = iree_hal_topology_edge_set_wait_mode(
      lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE);
  lo = iree_hal_topology_edge_set_signal_mode(
      lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE);
  lo = iree_hal_topology_edge_set_buffer_read_mode(
      lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE);
  lo = iree_hal_topology_edge_set_buffer_write_mode(
      lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE);
  lo = iree_hal_topology_edge_set_capability_flags(lo, 0xFFFF);  // 16 bits
  lo = iree_hal_topology_edge_set_wait_cost(lo, 15);
  lo = iree_hal_topology_edge_set_signal_cost(lo, 15);
  lo = iree_hal_topology_edge_set_copy_cost(lo, 15);
  lo = iree_hal_topology_edge_set_latency_class(lo, 15);
  lo = iree_hal_topology_edge_set_numa_distance(lo, 15);
  lo = iree_hal_topology_edge_set_link_class(lo, 7);  // 3 bits

  // Verify all fields retained their values.
  EXPECT_EQ(iree_hal_topology_edge_wait_mode(lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE);
  EXPECT_EQ(iree_hal_topology_edge_signal_mode(lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE);
  EXPECT_EQ(iree_hal_topology_edge_buffer_read_mode(lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE);
  EXPECT_EQ(iree_hal_topology_edge_buffer_write_mode(lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE);
  EXPECT_EQ(iree_hal_topology_edge_capability_flags(lo), 0xFFFF);
  EXPECT_EQ(iree_hal_topology_edge_wait_cost(lo), 15);
  EXPECT_EQ(iree_hal_topology_edge_signal_cost(lo), 15);
  EXPECT_EQ(iree_hal_topology_edge_copy_cost(lo), 15);
  EXPECT_EQ(iree_hal_topology_edge_latency_class(lo), 15);
  EXPECT_EQ(iree_hal_topology_edge_numa_distance(lo), 15);
  EXPECT_EQ(iree_hal_topology_edge_link_class(lo), 7);
}

// Verifies that interop word bitfields don't overlap.
TEST(TopologyEdge, InteropWordBitfieldOverlap) {
  iree_hal_topology_edge_interop_word_t hi = 0;

  // Set each handle type field to its maximum value (8 bits = 0xFF).
  hi = iree_hal_topology_edge_set_semaphore_import_types(hi, 0xFF);
  hi = iree_hal_topology_edge_set_semaphore_export_types(hi, 0xFF);
  hi = iree_hal_topology_edge_set_buffer_import_types(hi, 0xFF);
  hi = iree_hal_topology_edge_set_buffer_export_types(hi, 0xFF);

  // Verify all fields retained their values.
  EXPECT_EQ(iree_hal_topology_edge_semaphore_import_types(hi), 0xFF);
  EXPECT_EQ(iree_hal_topology_edge_semaphore_export_types(hi), 0xFF);
  EXPECT_EQ(iree_hal_topology_edge_buffer_import_types(hi), 0xFF);
  EXPECT_EQ(iree_hal_topology_edge_buffer_export_types(hi), 0xFF);
}

// Verifies that setting each scheduling field independently doesn't affect
// others.
TEST(TopologyEdge, SchedulingWordBitfieldIndependence) {
  iree_hal_topology_edge_scheduling_word_t lo = 0;

  // Set wait mode and verify only it changes.
  lo = iree_hal_topology_edge_set_wait_mode(
      lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  EXPECT_EQ(iree_hal_topology_edge_wait_mode(lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  EXPECT_EQ(iree_hal_topology_edge_signal_mode(lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);

  // Set signal cost and verify wait mode unchanged.
  lo = iree_hal_topology_edge_set_signal_cost(lo, 13);
  EXPECT_EQ(iree_hal_topology_edge_signal_cost(lo), 13);
  EXPECT_EQ(iree_hal_topology_edge_wait_mode(lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
}

// Verifies that setting interop fields independently doesn't affect others.
TEST(TopologyEdge, InteropWordBitfieldIndependence) {
  iree_hal_topology_edge_interop_word_t hi = 0;

  // Set semaphore import types.
  hi = iree_hal_topology_edge_set_semaphore_import_types(
      hi, IREE_HAL_TOPOLOGY_HANDLE_TYPE_OPAQUE_FD |
              IREE_HAL_TOPOLOGY_HANDLE_TYPE_RDMA_MR);
  EXPECT_EQ(iree_hal_topology_edge_semaphore_import_types(hi),
            IREE_HAL_TOPOLOGY_HANDLE_TYPE_OPAQUE_FD |
                IREE_HAL_TOPOLOGY_HANDLE_TYPE_RDMA_MR);
  EXPECT_EQ(iree_hal_topology_edge_semaphore_export_types(hi), 0);
  EXPECT_EQ(iree_hal_topology_edge_buffer_import_types(hi), 0);

  // Set buffer export types and verify semaphore import unchanged.
  hi = iree_hal_topology_edge_set_buffer_export_types(
      hi, IREE_HAL_TOPOLOGY_HANDLE_TYPE_DMA_BUF);
  EXPECT_EQ(iree_hal_topology_edge_buffer_export_types(hi),
            IREE_HAL_TOPOLOGY_HANDLE_TYPE_DMA_BUF);
  EXPECT_EQ(iree_hal_topology_edge_semaphore_import_types(hi),
            IREE_HAL_TOPOLOGY_HANDLE_TYPE_OPAQUE_FD |
                IREE_HAL_TOPOLOGY_HANDLE_TYPE_RDMA_MR);
}

//===----------------------------------------------------------------------===//
// Edge construction tests
//===----------------------------------------------------------------------===//

// Tests creation of a self-edge.
TEST(TopologyEdge, CreateSelf) {
  iree_hal_topology_edge_t edge = iree_hal_topology_edge_make_self();

  // Self-edges should have NATIVE mode for all operations.
  EXPECT_EQ(iree_hal_topology_edge_wait_mode(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  EXPECT_EQ(iree_hal_topology_edge_signal_mode(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  EXPECT_EQ(iree_hal_topology_edge_buffer_read_mode(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  EXPECT_EQ(iree_hal_topology_edge_buffer_write_mode(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);

  // Self-edges should have zero cost.
  EXPECT_EQ(iree_hal_topology_edge_wait_cost(edge.lo), 0);
  EXPECT_EQ(iree_hal_topology_edge_signal_cost(edge.lo), 0);
  EXPECT_EQ(iree_hal_topology_edge_copy_cost(edge.lo), 0);
  EXPECT_EQ(iree_hal_topology_edge_latency_class(edge.lo), 0);
  EXPECT_EQ(iree_hal_topology_edge_numa_distance(edge.lo), 0);

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
  EXPECT_EQ(iree_hal_topology_edge_capability_flags(edge.lo), expected_caps);

  // Self-edges use SAME_DIE link class.
  EXPECT_EQ(iree_hal_topology_edge_link_class(edge.lo),
            IREE_HAL_TOPOLOGY_LINK_CLASS_SAME_DIE);

  // Self-edges should have NATIVE handle types.
  EXPECT_EQ(iree_hal_topology_edge_semaphore_import_types(edge.hi),
            IREE_HAL_TOPOLOGY_HANDLE_TYPE_NATIVE);
  EXPECT_EQ(iree_hal_topology_edge_semaphore_export_types(edge.hi),
            IREE_HAL_TOPOLOGY_HANDLE_TYPE_NATIVE);
  EXPECT_EQ(iree_hal_topology_edge_buffer_import_types(edge.hi),
            IREE_HAL_TOPOLOGY_HANDLE_TYPE_NATIVE);
  EXPECT_EQ(iree_hal_topology_edge_buffer_export_types(edge.hi),
            IREE_HAL_TOPOLOGY_HANDLE_TYPE_NATIVE);
}

// Tests creation of a cross-driver edge.
TEST(TopologyEdge, CreateCrossDriver) {
  iree_hal_topology_edge_t edge = iree_hal_topology_edge_make_cross_driver();

  // Cross-driver edges use IMPORT for semaphores, COPY for buffers.
  EXPECT_EQ(iree_hal_topology_edge_wait_mode(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT);
  EXPECT_EQ(iree_hal_topology_edge_signal_mode(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT);
  EXPECT_EQ(iree_hal_topology_edge_buffer_read_mode(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  EXPECT_EQ(iree_hal_topology_edge_buffer_write_mode(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);

  // Cross-driver has moderate costs.
  EXPECT_EQ(iree_hal_topology_edge_wait_cost(edge.lo), 5);
  EXPECT_EQ(iree_hal_topology_edge_signal_cost(edge.lo), 5);
  EXPECT_EQ(iree_hal_topology_edge_copy_cost(edge.lo), 10);
  EXPECT_EQ(iree_hal_topology_edge_latency_class(edge.lo), 8);
  EXPECT_EQ(iree_hal_topology_edge_numa_distance(edge.lo), 2);

  // No special capabilities for cross-driver.
  EXPECT_EQ(iree_hal_topology_edge_capability_flags(edge.lo),
            IREE_HAL_TOPOLOGY_CAPABILITY_NONE);

  // Cross-driver uses PCIe link class by default.
  EXPECT_EQ(iree_hal_topology_edge_link_class(edge.lo),
            IREE_HAL_TOPOLOGY_LINK_CLASS_PCIE_SAME_ROOT);

  // No handle types set by default for cross-driver.
  EXPECT_EQ(iree_hal_topology_edge_semaphore_import_types(edge.hi), 0);
  EXPECT_EQ(iree_hal_topology_edge_buffer_import_types(edge.hi), 0);
}

//===----------------------------------------------------------------------===//
// Aliased device detection tests
//===----------------------------------------------------------------------===//

// Tests that from_capabilities detects aliased same-driver devices and returns
// a self-edge when driver_device_handle matches.
TEST(TopologyEdge, AliasedDeviceDetection) {
  iree_hal_device_capabilities_t caps = {0};
  caps.flags = IREE_HAL_DEVICE_CAPABILITY_TIMELINE_SEMAPHORES;
  caps.driver_device_handle = 0x12345678;

  // Same driver, same handle → should return self-edge.
  iree_hal_topology_edge_t edge = iree_hal_topology_edge_from_capabilities(
      &caps, &caps, IREE_SV("hip"), IREE_SV("hip"));

  EXPECT_EQ(iree_hal_topology_edge_wait_mode(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  EXPECT_EQ(iree_hal_topology_edge_signal_mode(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  EXPECT_EQ(iree_hal_topology_edge_buffer_read_mode(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  EXPECT_EQ(iree_hal_topology_edge_copy_cost(edge.lo), 0);
  EXPECT_EQ(iree_hal_topology_edge_link_class(edge.lo),
            IREE_HAL_TOPOLOGY_LINK_CLASS_SAME_DIE);
}

// Tests that different handles on the same driver are NOT treated as aliased.
TEST(TopologyEdge, DifferentHandlesNotAliased) {
  iree_hal_device_capabilities_t caps_a = {0};
  caps_a.driver_device_handle = 0x11111111;

  iree_hal_device_capabilities_t caps_b = {0};
  caps_b.driver_device_handle = 0x22222222;

  iree_hal_topology_edge_t edge = iree_hal_topology_edge_from_capabilities(
      &caps_a, &caps_b, IREE_SV("hip"), IREE_SV("hip"));

  // Different handles → should NOT be a self-edge.
  // Same driver without P2P or UUID match gives cross-driver-like costs.
  EXPECT_NE(iree_hal_topology_edge_copy_cost(edge.lo), 0);
}

// Tests that zero handle (sentinel for "not set") skips aliasing detection.
TEST(TopologyEdge, ZeroHandleNoAliasing) {
  iree_hal_device_capabilities_t caps = {0};
  // driver_device_handle is 0 (default from zero-init).

  iree_hal_topology_edge_t edge = iree_hal_topology_edge_from_capabilities(
      &caps, &caps, IREE_SV("hip"), IREE_SV("hip"));

  // Zero handle → no aliasing detection.
  // Same driver but no UUID or P2P → non-zero copy cost.
  EXPECT_NE(iree_hal_topology_edge_copy_cost(edge.lo), 0);
}

//===----------------------------------------------------------------------===//
// Resource origin tests
//===----------------------------------------------------------------------===//

// Tests resource origin initialization.
TEST(ResourceOrigin, Initialize) {
  iree_hal_topology_edge_t edge = iree_hal_topology_edge_make_self();

  iree_hal_resource_origin_t origin = {
      /*.self_edge=*/edge.lo,
      /*.topology_index=*/3,
  };

  EXPECT_EQ(origin.self_edge, edge.lo);
  EXPECT_EQ(origin.topology_index, 3);

  // Check size is as expected (16 bytes with padding).
  EXPECT_EQ(sizeof(iree_hal_resource_origin_t), 16);
}

// Tests compatibility checking between resources.
TEST(ResourceOrigin, CompatibilityCheck) {
  iree_hal_topology_edge_t edge1 = iree_hal_topology_edge_make_self();

  iree_hal_topology_edge_scheduling_word_t lo2 = 0;
  lo2 = iree_hal_topology_edge_set_wait_mode(
      lo2, IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT);
  lo2 = iree_hal_topology_edge_set_capability_flags(lo2, 0x42);

  iree_hal_resource_origin_t origin1 = {
      /*.self_edge=*/edge1.lo,
      /*.topology_index=*/0,
  };
  iree_hal_resource_origin_t origin2 = {
      /*.self_edge=*/lo2,
      /*.topology_index=*/1,
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
  edge.lo = iree_hal_topology_edge_set_wait_mode(
      edge.lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  edge.lo = iree_hal_topology_edge_set_copy_cost(edge.lo, 13);

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
        edge.lo = iree_hal_topology_edge_set_link_class(
            edge.lo, IREE_HAL_TOPOLOGY_LINK_CLASS_NVLINK_IF);
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

//===----------------------------------------------------------------------===//
// New handle type tests
//===----------------------------------------------------------------------===//

// Tests that new handle types (RDMA_MR, SHM, etc.) can be set and retrieved.
TEST(TopologyEdge, NewHandleTypes) {
  iree_hal_topology_edge_interop_word_t hi = 0;

  // Set all 8 handle type bits.
  iree_hal_topology_handle_type_t all_types =
      IREE_HAL_TOPOLOGY_HANDLE_TYPE_NATIVE |
      IREE_HAL_TOPOLOGY_HANDLE_TYPE_OPAQUE_FD |
      IREE_HAL_TOPOLOGY_HANDLE_TYPE_OPAQUE_WIN32 |
      IREE_HAL_TOPOLOGY_HANDLE_TYPE_DMA_BUF |
      IREE_HAL_TOPOLOGY_HANDLE_TYPE_RDMA_MR |
      IREE_HAL_TOPOLOGY_HANDLE_TYPE_SHM |
      IREE_HAL_TOPOLOGY_HANDLE_TYPE_METAL_IOSURFACE |
      IREE_HAL_TOPOLOGY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER;

  hi = iree_hal_topology_edge_set_buffer_import_types(hi, all_types);
  EXPECT_EQ(iree_hal_topology_edge_buffer_import_types(hi), all_types);
  EXPECT_EQ(iree_hal_topology_edge_buffer_import_types(hi), 0xFF);
}

// Tests RDMA-specific handle types in a realistic configuration.
TEST(TopologyEdge, RdmaHandleTypes) {
  iree_hal_topology_edge_interop_word_t hi = 0;

  // An RDMA-capable edge would support MR for buffers and SHM for semaphores.
  hi = iree_hal_topology_edge_set_buffer_import_types(
      hi, IREE_HAL_TOPOLOGY_HANDLE_TYPE_RDMA_MR |
              IREE_HAL_TOPOLOGY_HANDLE_TYPE_SHM);
  hi = iree_hal_topology_edge_set_buffer_export_types(
      hi, IREE_HAL_TOPOLOGY_HANDLE_TYPE_RDMA_MR |
              IREE_HAL_TOPOLOGY_HANDLE_TYPE_SHM);
  hi = iree_hal_topology_edge_set_semaphore_import_types(
      hi, IREE_HAL_TOPOLOGY_HANDLE_TYPE_SHM);
  hi = iree_hal_topology_edge_set_semaphore_export_types(
      hi, IREE_HAL_TOPOLOGY_HANDLE_TYPE_SHM);

  // Verify RDMA_MR is set for buffers but not semaphores.
  EXPECT_TRUE(iree_hal_topology_edge_buffer_import_types(hi) &
              IREE_HAL_TOPOLOGY_HANDLE_TYPE_RDMA_MR);
  EXPECT_FALSE(iree_hal_topology_edge_semaphore_import_types(hi) &
               IREE_HAL_TOPOLOGY_HANDLE_TYPE_RDMA_MR);

  // Verify SHM is set for both.
  EXPECT_TRUE(iree_hal_topology_edge_buffer_import_types(hi) &
              IREE_HAL_TOPOLOGY_HANDLE_TYPE_SHM);
  EXPECT_TRUE(iree_hal_topology_edge_semaphore_import_types(hi) &
              IREE_HAL_TOPOLOGY_HANDLE_TYPE_SHM);
}

//===----------------------------------------------------------------------===//
// Topology info cost query tests
//===----------------------------------------------------------------------===//

// Tests iree_hal_device_topology_query_edge returns the correct edge
// when both devices share the same topology.
TEST(TopologyInfo, QueryEdgeSameTopology) {
  // Build a 2-device topology.
  iree_hal_topology_builder_t builder;
  iree_hal_topology_builder_initialize(&builder, 2);

  iree_hal_topology_edge_t cross = iree_hal_topology_edge_make_cross_driver();
  cross.lo = iree_hal_topology_edge_set_copy_cost(cross.lo, 7);
  cross.lo = iree_hal_topology_edge_set_link_class(
      cross.lo, IREE_HAL_TOPOLOGY_LINK_CLASS_NVLINK_IF);
  cross.hi = iree_hal_topology_edge_set_buffer_import_types(
      cross.hi, IREE_HAL_TOPOLOGY_HANDLE_TYPE_DMA_BUF);

  IREE_ASSERT_OK(iree_hal_topology_builder_set_edge(&builder, 0, 1, cross));
  IREE_ASSERT_OK(iree_hal_topology_builder_set_edge(&builder, 1, 0, cross));

  iree_hal_topology_t topology;
  IREE_ASSERT_OK(iree_hal_topology_builder_finalize(&builder, &topology));

  // Simulate two devices pointing at the same topology.
  iree_hal_device_topology_info_t info0 = {0};
  info0.self_edge = topology.edges[0].lo;
  info0.topology_index = 0;
  info0.topology = &topology;

  iree_hal_device_topology_info_t info1 = {0};
  info1.self_edge = topology.edges[3].lo;
  info1.topology_index = 1;
  info1.topology = &topology;

  // Query the edge from device 0 to device 1.
  iree_hal_topology_edge_t queried =
      iree_hal_device_topology_query_edge(&info0, &info1);
  EXPECT_EQ(iree_hal_topology_edge_copy_cost(queried.lo), 7);
  EXPECT_EQ(iree_hal_topology_edge_link_class(queried.lo),
            IREE_HAL_TOPOLOGY_LINK_CLASS_NVLINK_IF);
  EXPECT_EQ(iree_hal_topology_edge_buffer_import_types(queried.hi),
            IREE_HAL_TOPOLOGY_HANDLE_TYPE_DMA_BUF);

  // Self-query returns the self-edge.
  iree_hal_topology_edge_t self =
      iree_hal_device_topology_query_edge(&info0, &info0);
  EXPECT_EQ(iree_hal_topology_edge_wait_mode(self.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  EXPECT_EQ(iree_hal_topology_edge_copy_cost(self.lo), 0);
}

// Tests that iree_hal_device_topology_query_edge returns empty when devices
// are in different topologies or not in any topology.
TEST(TopologyInfo, QueryEdgeDifferentTopologies) {
  iree_hal_topology_t topology_a = {/*.device_count=*/1};
  topology_a.edges[0] = iree_hal_topology_edge_make_self();

  iree_hal_topology_t topology_b = {/*.device_count=*/1};
  topology_b.edges[0] = iree_hal_topology_edge_make_self();

  iree_hal_device_topology_info_t info_a = {0};
  info_a.topology_index = 0;
  info_a.topology = &topology_a;

  iree_hal_device_topology_info_t info_b = {0};
  info_b.topology_index = 0;
  info_b.topology = &topology_b;

  // Different topologies: should return empty edge.
  iree_hal_topology_edge_t edge =
      iree_hal_device_topology_query_edge(&info_a, &info_b);
  EXPECT_TRUE(iree_hal_topology_edge_is_empty(edge));
}

// Tests that iree_hal_device_topology_query_edge returns empty when the
// topology pointer is NULL (standalone device).
TEST(TopologyInfo, QueryEdgeStandaloneDevice) {
  iree_hal_device_topology_info_t info_standalone = {0};
  info_standalone.topology = NULL;

  iree_hal_topology_t topology = {/*.device_count=*/1};
  topology.edges[0] = iree_hal_topology_edge_make_self();
  iree_hal_device_topology_info_t info_grouped = {0};
  info_grouped.topology = &topology;

  // NULL topology: should return empty edge.
  iree_hal_topology_edge_t edge =
      iree_hal_device_topology_query_edge(&info_standalone, &info_grouped);
  EXPECT_TRUE(iree_hal_topology_edge_is_empty(edge));

  // Both NULL: should return empty edge.
  iree_hal_device_topology_info_t info_standalone2 = {0};
  edge =
      iree_hal_device_topology_query_edge(&info_standalone, &info_standalone2);
  EXPECT_TRUE(iree_hal_topology_edge_is_empty(edge));
}

}  // namespace
}  // namespace iree::hal
