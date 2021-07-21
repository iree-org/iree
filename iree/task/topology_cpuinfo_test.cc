// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/task/topology_cpuinfo.h"

#include <cstddef>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using namespace iree::testing::status;

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
  iree_task_topology_initialize_from_physical_cores(kMaxGroupCount, &topology);
  EnsureTopologyValid(kMaxGroupCount, &topology);
  iree_task_topology_deinitialize(&topology);
}

}  // namespace
