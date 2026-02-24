// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/testing/mock_device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal {
namespace {

using ::iree::testing::status::StatusIs;

// Helper: creates a mock device with the given identifier and default
// (zeroed) capabilities.
static iree_hal_device_t* CreateMockDevice(const char* identifier) {
  iree_hal_mock_device_options_t options;
  iree_hal_mock_device_options_initialize(&options);
  options.identifier = iree_make_cstring_view(identifier);
  iree_hal_device_t* device = NULL;
  IREE_CHECK_OK(
      iree_hal_mock_device_create(&options, iree_allocator_system(), &device));
  return device;
}

// Helper: creates a mock device with configurable capabilities.
static iree_hal_device_t* CreateMockDeviceWithCapabilities(
    const char* identifier,
    const iree_hal_device_capabilities_t* capabilities) {
  iree_hal_mock_device_options_t options;
  iree_hal_mock_device_options_initialize(&options);
  options.identifier = iree_make_cstring_view(identifier);
  options.capabilities = *capabilities;
  iree_hal_device_t* device = NULL;
  IREE_CHECK_OK(
      iree_hal_mock_device_create(&options, iree_allocator_system(), &device));
  return device;
}

//===----------------------------------------------------------------------===//
// Builder validation tests
//===----------------------------------------------------------------------===//

// An empty builder (no devices added) must fail to finalize.
TEST(DeviceGroupBuilder, EmptyBuilderFails) {
  iree_hal_device_group_builder_t builder;
  iree_hal_device_group_builder_initialize(&builder);

  iree_hal_device_group_t* group = NULL;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_device_group_builder_finalize(
                            &builder, iree_allocator_system(), &group));
  EXPECT_EQ(group, nullptr);

  iree_hal_device_group_builder_deinitialize(&builder);
}

// Adding more than IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT devices must fail.
TEST(DeviceGroupBuilder, ExceedsMaxCapacity) {
  iree_hal_device_group_builder_t builder;
  iree_hal_device_group_builder_initialize(&builder);

  iree_hal_device_t* device = CreateMockDevice("mock");
  for (iree_host_size_t i = 0; i < IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT; ++i) {
    IREE_ASSERT_OK(iree_hal_device_group_builder_add_device(&builder, device));
  }

  // The next add must fail.
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_device_group_builder_add_device(&builder, device));

  iree_hal_device_group_builder_deinitialize(&builder);
  iree_hal_device_release(device);
}

// After finalize the builder is zeroed and must not be reused.
TEST(DeviceGroupBuilder, FinalizeInvalidatesBuilder) {
  iree_hal_device_group_builder_t builder;
  iree_hal_device_group_builder_initialize(&builder);

  iree_hal_device_t* device = CreateMockDevice("mock");
  IREE_ASSERT_OK(iree_hal_device_group_builder_add_device(&builder, device));

  iree_hal_device_group_t* group = NULL;
  IREE_ASSERT_OK(iree_hal_device_group_builder_finalize(
      &builder, iree_allocator_system(), &group));

  // Builder should be zeroed.
  EXPECT_EQ(builder.count, 0u);
  for (iree_host_size_t i = 0; i < IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT; ++i) {
    EXPECT_EQ(builder.devices[i], nullptr);
  }

  iree_hal_device_group_release(group);
  iree_hal_device_release(device);
}

// A failed finalize (empty builder) also zeroes the builder.
TEST(DeviceGroupBuilder, FailedFinalizeInvalidatesBuilder) {
  iree_hal_device_group_builder_t builder;
  iree_hal_device_group_builder_initialize(&builder);

  iree_hal_device_group_t* group = NULL;
  iree_status_t status = iree_hal_device_group_builder_finalize(
      &builder, iree_allocator_system(), &group);
  EXPECT_THAT(status, StatusIs(iree::StatusCode::kInvalidArgument));
  iree_status_ignore(status);

  EXPECT_EQ(builder.count, 0u);
}

//===----------------------------------------------------------------------===//
// Single-device group tests
//===----------------------------------------------------------------------===//

// A group with one device should have a self-edge topology.
TEST(DeviceGroup, SingleDevice) {
  iree_hal_device_t* device = CreateMockDevice("mock");

  iree_hal_device_group_builder_t builder;
  iree_hal_device_group_builder_initialize(&builder);
  IREE_ASSERT_OK(iree_hal_device_group_builder_add_device(&builder, device));

  iree_hal_device_group_t* group = NULL;
  IREE_ASSERT_OK(iree_hal_device_group_builder_finalize(
      &builder, iree_allocator_system(), &group));

  // Verify accessors.
  EXPECT_EQ(iree_hal_device_group_device_count(group), 1u);
  EXPECT_EQ(iree_hal_device_group_device_at(group, 0), device);
  EXPECT_EQ(iree_hal_device_group_device_at(group, 1), nullptr);

  // Topology should have 1 device with a valid self-edge.
  const iree_hal_topology_t* topology = iree_hal_device_group_topology(group);
  EXPECT_EQ(topology->device_count, 1u);
  iree_hal_topology_edge_t self_edge =
      iree_hal_topology_query_edge(topology, 0, 0);
  EXPECT_EQ(iree_hal_topology_edge_wait_mode(self_edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  EXPECT_EQ(iree_hal_topology_edge_signal_mode(self_edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);

  // Device should have topology info assigned.
  const iree_hal_device_topology_info_t* info =
      iree_hal_device_topology_info(device);
  EXPECT_EQ(info->topology_index, 0u);
  EXPECT_EQ(info->topology, topology);

  // Single-device group: all bitmaps should be 0 (no peers).
  EXPECT_EQ(info->can_wait_from, 0u);
  EXPECT_EQ(info->can_signal_to, 0u);
  EXPECT_EQ(info->can_import_from, 0u);
  EXPECT_EQ(info->can_p2p_with, 0u);

  iree_hal_device_group_release(group);
  iree_hal_device_release(device);
}

//===----------------------------------------------------------------------===//
// Multi-device group tests
//===----------------------------------------------------------------------===//

// Two devices with the same driver identifier.
TEST(DeviceGroup, TwoDevicesSameDriver) {
  iree_hal_device_t* device_a = CreateMockDevice("mock");
  iree_hal_device_t* device_b = CreateMockDevice("mock");

  iree_hal_device_group_builder_t builder;
  iree_hal_device_group_builder_initialize(&builder);
  IREE_ASSERT_OK(iree_hal_device_group_builder_add_device(&builder, device_a));
  IREE_ASSERT_OK(iree_hal_device_group_builder_add_device(&builder, device_b));

  iree_hal_device_group_t* group = NULL;
  IREE_ASSERT_OK(iree_hal_device_group_builder_finalize(
      &builder, iree_allocator_system(), &group));

  EXPECT_EQ(iree_hal_device_group_device_count(group), 2u);
  EXPECT_EQ(iree_hal_device_group_device_at(group, 0), device_a);
  EXPECT_EQ(iree_hal_device_group_device_at(group, 1), device_b);

  const iree_hal_topology_t* topology = iree_hal_device_group_topology(group);
  EXPECT_EQ(topology->device_count, 2u);

  // Cross-device edge should exist (computed from zeroed capabilities).
  iree_hal_topology_edge_t edge_ab =
      iree_hal_topology_query_edge(topology, 0, 1);
  EXPECT_FALSE(iree_hal_topology_edge_is_empty(edge_ab));

  // Both devices should have topology info with correct indices.
  const iree_hal_device_topology_info_t* info_a =
      iree_hal_device_topology_info(device_a);
  const iree_hal_device_topology_info_t* info_b =
      iree_hal_device_topology_info(device_b);
  EXPECT_EQ(info_a->topology_index, 0u);
  EXPECT_EQ(info_b->topology_index, 1u);
  EXPECT_EQ(info_a->topology, topology);
  EXPECT_EQ(info_b->topology, topology);

  // Cross-device query through topology_info should return the same edge.
  iree_hal_topology_edge_t queried =
      iree_hal_device_topology_query_edge(info_a, info_b);
  EXPECT_EQ(queried.lo, edge_ab.lo);
  EXPECT_EQ(queried.hi, edge_ab.hi);

  iree_hal_device_group_release(group);
  iree_hal_device_release(device_a);
  iree_hal_device_release(device_b);
}

// Two devices with different driver identifiers (cross-driver pair).
TEST(DeviceGroup, TwoDevicesDifferentDrivers) {
  iree_hal_device_t* device_a = CreateMockDevice("driver_a");
  iree_hal_device_t* device_b = CreateMockDevice("driver_b");

  iree_hal_device_group_builder_t builder;
  iree_hal_device_group_builder_initialize(&builder);
  IREE_ASSERT_OK(iree_hal_device_group_builder_add_device(&builder, device_a));
  IREE_ASSERT_OK(iree_hal_device_group_builder_add_device(&builder, device_b));

  iree_hal_device_group_t* group = NULL;
  IREE_ASSERT_OK(iree_hal_device_group_builder_finalize(
      &builder, iree_allocator_system(), &group));

  const iree_hal_topology_t* topology = iree_hal_device_group_topology(group);

  // Cross-driver edge from edge_from_capabilities with zeroed capabilities
  // (no import/export handle types) produces COPY mode for everything — there
  // are no external handles to import, so the only option is host-staged copy.
  iree_hal_topology_edge_t edge_ab =
      iree_hal_topology_query_edge(topology, 0, 1);
  EXPECT_EQ(iree_hal_topology_edge_wait_mode(edge_ab.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  EXPECT_EQ(iree_hal_topology_edge_buffer_read_mode(edge_ab.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);

  iree_hal_device_group_release(group);
  iree_hal_device_release(device_a);
  iree_hal_device_release(device_b);
}

// Three devices: verifies bitmap computation with more than two devices.
TEST(DeviceGroup, ThreeDevicesBitmaps) {
  iree_hal_device_t* device_a = CreateMockDevice("mock");
  iree_hal_device_t* device_b = CreateMockDevice("mock");
  iree_hal_device_t* device_c = CreateMockDevice("mock");

  iree_hal_device_group_builder_t builder;
  iree_hal_device_group_builder_initialize(&builder);
  IREE_ASSERT_OK(iree_hal_device_group_builder_add_device(&builder, device_a));
  IREE_ASSERT_OK(iree_hal_device_group_builder_add_device(&builder, device_b));
  IREE_ASSERT_OK(iree_hal_device_group_builder_add_device(&builder, device_c));

  iree_hal_device_group_t* group = NULL;
  IREE_ASSERT_OK(iree_hal_device_group_builder_finalize(
      &builder, iree_allocator_system(), &group));

  EXPECT_EQ(iree_hal_device_group_device_count(group), 3u);

  const iree_hal_device_topology_info_t* info_a =
      iree_hal_device_topology_info(device_a);
  const iree_hal_device_topology_info_t* info_b =
      iree_hal_device_topology_info(device_b);
  const iree_hal_device_topology_info_t* info_c =
      iree_hal_device_topology_info(device_c);

  // All three devices should have topology indices 0, 1, 2.
  EXPECT_EQ(info_a->topology_index, 0u);
  EXPECT_EQ(info_b->topology_index, 1u);
  EXPECT_EQ(info_c->topology_index, 2u);

  // With zeroed capabilities, edge_from_capabilities produces edges with
  // IMPORT wait mode (non-NONE) and IMPORT signal mode, so can_wait_from
  // and can_signal_to should have all peer bits set.
  //
  // Device 0 peers are 1 and 2: bitmap = (1<<1) | (1<<2) = 0x6.
  // Device 1 peers are 0 and 2: bitmap = (1<<0) | (1<<2) = 0x5.
  // Device 2 peers are 0 and 1: bitmap = (1<<0) | (1<<1) = 0x3.
  EXPECT_EQ(info_a->can_wait_from, 0x6u);
  EXPECT_EQ(info_b->can_wait_from, 0x5u);
  EXPECT_EQ(info_c->can_wait_from, 0x3u);

  EXPECT_EQ(info_a->can_signal_to, 0x6u);
  EXPECT_EQ(info_b->can_signal_to, 0x5u);
  EXPECT_EQ(info_c->can_signal_to, 0x3u);

  iree_hal_device_group_release(group);
  iree_hal_device_release(device_a);
  iree_hal_device_release(device_b);
  iree_hal_device_release(device_c);
}

//===----------------------------------------------------------------------===//
// Configurable capabilities tests
//===----------------------------------------------------------------------===//

// Devices with specific NUMA node assignments should produce a topology with
// those NUMA nodes set.
TEST(DeviceGroup, NumaNodeAssignment) {
  iree_hal_device_capabilities_t caps_a;
  memset(&caps_a, 0, sizeof(caps_a));
  caps_a.numa_node = 0;

  iree_hal_device_capabilities_t caps_b;
  memset(&caps_b, 0, sizeof(caps_b));
  caps_b.numa_node = 1;

  iree_hal_device_t* device_a =
      CreateMockDeviceWithCapabilities("mock", &caps_a);
  iree_hal_device_t* device_b =
      CreateMockDeviceWithCapabilities("mock", &caps_b);

  iree_hal_device_group_builder_t builder;
  iree_hal_device_group_builder_initialize(&builder);
  IREE_ASSERT_OK(iree_hal_device_group_builder_add_device(&builder, device_a));
  IREE_ASSERT_OK(iree_hal_device_group_builder_add_device(&builder, device_b));

  iree_hal_device_group_t* group = NULL;
  IREE_ASSERT_OK(iree_hal_device_group_builder_finalize(
      &builder, iree_allocator_system(), &group));

  const iree_hal_topology_t* topology = iree_hal_device_group_topology(group);
  EXPECT_EQ(topology->numa_nodes[0], 0u);
  EXPECT_EQ(topology->numa_nodes[1], 1u);

  iree_hal_device_group_release(group);
  iree_hal_device_release(device_a);
  iree_hal_device_release(device_b);
}

//===----------------------------------------------------------------------===//
// Lifetime and reference counting tests
//===----------------------------------------------------------------------===//

// The group retains devices, so the device stays alive after the caller
// releases it.
TEST(DeviceGroup, GroupRetainsDevices) {
  iree_hal_device_t* device = CreateMockDevice("mock");

  iree_hal_device_group_builder_t builder;
  iree_hal_device_group_builder_initialize(&builder);
  IREE_ASSERT_OK(iree_hal_device_group_builder_add_device(&builder, device));

  iree_hal_device_group_t* group = NULL;
  IREE_ASSERT_OK(iree_hal_device_group_builder_finalize(
      &builder, iree_allocator_system(), &group));

  // Release our reference — the group still holds one.
  iree_hal_device_release(device);

  // Device should still be accessible through the group.
  EXPECT_NE(iree_hal_device_group_device_at(group, 0), nullptr);

  // Releasing the group releases the last device reference.
  iree_hal_device_group_release(group);
}

// Retain/release on the group itself.
TEST(DeviceGroup, RetainRelease) {
  iree_hal_device_t* device = CreateMockDevice("mock");

  iree_hal_device_group_builder_t builder;
  iree_hal_device_group_builder_initialize(&builder);
  IREE_ASSERT_OK(iree_hal_device_group_builder_add_device(&builder, device));

  iree_hal_device_group_t* group = NULL;
  IREE_ASSERT_OK(iree_hal_device_group_builder_finalize(
      &builder, iree_allocator_system(), &group));

  // Extra retain + release cycle.
  iree_hal_device_group_retain(group);
  iree_hal_device_group_release(group);

  // Group is still alive, topology pointer should be valid.
  const iree_hal_topology_t* topology = iree_hal_device_group_topology(group);
  EXPECT_EQ(topology->device_count, 1u);

  iree_hal_device_group_release(group);
  iree_hal_device_release(device);
}

// Null is safe for retain/release.
TEST(DeviceGroup, NullRetainRelease) {
  iree_hal_device_group_retain(NULL);
  iree_hal_device_group_release(NULL);
}

//===----------------------------------------------------------------------===//
// Topology pointer stability tests
//===----------------------------------------------------------------------===//

// The topology pointer returned by the group must remain stable and match
// what was assigned to devices.
TEST(DeviceGroup, TopologyPointerStability) {
  iree_hal_device_t* device_a = CreateMockDevice("mock");
  iree_hal_device_t* device_b = CreateMockDevice("mock");

  iree_hal_device_group_builder_t builder;
  iree_hal_device_group_builder_initialize(&builder);
  IREE_ASSERT_OK(iree_hal_device_group_builder_add_device(&builder, device_a));
  IREE_ASSERT_OK(iree_hal_device_group_builder_add_device(&builder, device_b));

  iree_hal_device_group_t* group = NULL;
  IREE_ASSERT_OK(iree_hal_device_group_builder_finalize(
      &builder, iree_allocator_system(), &group));

  const iree_hal_topology_t* group_topology =
      iree_hal_device_group_topology(group);
  const iree_hal_device_topology_info_t* info_a =
      iree_hal_device_topology_info(device_a);
  const iree_hal_device_topology_info_t* info_b =
      iree_hal_device_topology_info(device_b);

  // Both devices should point at the same topology that the group owns.
  EXPECT_EQ(info_a->topology, group_topology);
  EXPECT_EQ(info_b->topology, group_topology);

  iree_hal_device_group_release(group);
  iree_hal_device_release(device_a);
  iree_hal_device_release(device_b);
}

}  // namespace
}  // namespace iree::hal
