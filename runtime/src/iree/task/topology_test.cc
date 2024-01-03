// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/task/topology.h"

#include <cstddef>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using namespace iree::testing::status;

TEST(TopologyTest, Lifetime) {
  iree_task_topology_t topology;
  iree_task_topology_initialize(&topology);
  EXPECT_GT(iree_task_topology_group_capacity(&topology), 0);
  EXPECT_EQ(0, iree_task_topology_group_count(&topology));
  iree_task_topology_deinitialize(&topology);
}

TEST(TopologyTest, Empty) {
  iree_task_topology_t topology;
  iree_task_topology_initialize(&topology);

  EXPECT_EQ(0, iree_task_topology_group_count(&topology));
  EXPECT_EQ(NULL, iree_task_topology_get_group(&topology, 0));
  EXPECT_EQ(NULL, iree_task_topology_get_group(&topology, 100));

  iree_task_topology_deinitialize(&topology);
}

TEST(TopologyTest, Parsing) {
  // TODO(benvanik): implement parsing.
}

TEST(TopologyTest, Formatting) {
  // TODO(benvanik): implement formatting.
}

TEST(TopologyTest, Construction) {
  iree_task_topology_t topology;
  iree_task_topology_initialize(&topology);

  EXPECT_EQ(0, iree_task_topology_group_count(&topology));

  for (iree_host_size_t i = 0; i < 8; ++i) {
    iree_task_topology_group_t group;
    iree_task_topology_group_initialize(i, &group);
    IREE_EXPECT_OK(iree_task_topology_push_group(&topology, &group));
    EXPECT_EQ(i + 1, iree_task_topology_group_count(&topology));
  }
  EXPECT_EQ(8, iree_task_topology_group_count(&topology));

  for (iree_host_size_t i = 0; i < 8; ++i) {
    const iree_task_topology_group_t* group =
        iree_task_topology_get_group(&topology, i);
    EXPECT_EQ(i, group->group_index);
  }

  iree_task_topology_deinitialize(&topology);
}

TEST(TopologyTest, MaxCapacity) {
  iree_task_topology_t topology;
  iree_task_topology_initialize(&topology);

  EXPECT_EQ(0, iree_task_topology_group_count(&topology));

  // Fill up to capacity.
  for (iree_host_size_t i = 0; i < iree_task_topology_group_capacity(&topology);
       ++i) {
    iree_task_topology_group_t group;
    iree_task_topology_group_initialize(i, &group);
    IREE_EXPECT_OK(iree_task_topology_push_group(&topology, &group));
    EXPECT_EQ(i + 1, iree_task_topology_group_count(&topology));
  }
  EXPECT_EQ(iree_task_topology_group_capacity(&topology),
            iree_task_topology_group_count(&topology));

  // Try adding one more - it should it fail because we are at capacity.
  iree_task_topology_group_t extra_group;
  iree_task_topology_group_initialize(UINT8_MAX, &extra_group);
  iree_status_t status = iree_task_topology_push_group(&topology, &extra_group);
  EXPECT_TRUE(iree_status_is_resource_exhausted(status));
  iree_status_ignore(status);

  // Confirm that the only groups we have are the valid ones we added above.
  for (iree_host_size_t i = 0; i < 8; ++i) {
    const iree_task_topology_group_t* group =
        iree_task_topology_get_group(&topology, i);
    EXPECT_EQ(i, group->group_index);
  }

  iree_task_topology_deinitialize(&topology);
}

TEST(TopologyTest, FromGroupCount) {
  static constexpr iree_host_size_t kGroupCount = 4;
  iree_task_topology_t topology;
  iree_task_topology_initialize(&topology);

  iree_task_topology_initialize_from_group_count(kGroupCount, &topology);
  EXPECT_LE(iree_task_topology_group_count(&topology),
            iree_task_topology_group_capacity(&topology));
  EXPECT_EQ(iree_task_topology_group_count(&topology), kGroupCount);
  for (iree_host_size_t i = 0; i < kGroupCount; ++i) {
    const iree_task_topology_group_t* group =
        iree_task_topology_get_group(&topology, i);
    EXPECT_EQ(i, group->group_index);
  }

  iree_task_topology_deinitialize(&topology);
}

// Verifies only that the |topology| is usable.
// If we actually checked the contents here then we'd just be validating that
// cpuinfo was working and the tests would become machine-dependent.
static void EnsureTopologyValid(iree_host_size_t max_group_count,
                                iree_task_topology_t* topology) {
  EXPECT_LE(iree_task_topology_group_count(topology),
            iree_task_topology_group_capacity(topology));
  EXPECT_LE(iree_task_topology_group_count(topology), max_group_count);
  EXPECT_GE(iree_task_topology_group_count(topology), 1);
  for (iree_host_size_t i = 0; i < iree_task_topology_group_count(topology);
       ++i) {
    const iree_task_topology_group_t* group =
        iree_task_topology_get_group(topology, i);
    EXPECT_EQ(i, group->group_index);
  }
}

TEST(TopologyTest, FromPhysicalCores) {
  static constexpr iree_host_size_t kMaxGroupCount = 4;
  iree_task_topology_t topology;
  iree_task_topology_initialize(&topology);
  IREE_ASSERT_OK(iree_task_topology_initialize_from_physical_cores(
      IREE_TASK_TOPOLOGY_NODE_ID_ANY, IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_ANY,
      kMaxGroupCount, &topology));
  EnsureTopologyValid(kMaxGroupCount, &topology);
  iree_task_topology_deinitialize(&topology);
}

}  // namespace
