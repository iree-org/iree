// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/sysfs.h"
#include "iree/task/topology.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

#include <cstdlib>
#include <string>

#include <unistd.h>

namespace {

#if defined(IREE_PLATFORM_LINUX) && !defined(IREE_PLATFORM_WASM)

using namespace iree::testing::status;

// Resolves the hybrid sparse-cluster fixture directory, extracting the checked
// in tar.gz when needed. Override with IREE_SYSFS_TEST_ROOT for local debugging.
static bool FixturePresentReadable(const std::string& dir) {
  const std::string present = dir + "/cpu/present";
  return access(present.c_str(), R_OK) == 0;
}

static bool TryExtractFixture(const std::string& base, std::string* out_dir) {
  const std::string dir = base + "/x86_hybrid_sparse_clusters";
  if (FixturePresentReadable(dir)) {
    *out_dir = dir;
    return true;
  }
  const std::string tar = base + "/x86_hybrid_sparse_clusters.tar.gz";
  if (access(tar.c_str(), R_OK) != 0) {
    return false;
  }
  const std::string cmd = "tar xzf \"" + tar + "\" -C \"" + base + "\"";
  if (std::system(cmd.c_str()) != 0) {
    return false;
  }
  if (!FixturePresentReadable(dir)) {
    return false;
  }
  *out_dir = dir;
  return true;
}

static std::string ResolveHybridFixtureRoot() {
  if (const char* env = std::getenv("IREE_SYSFS_TEST_ROOT")) {
    return std::string(env);
  }

  std::string dir;
#if defined(IREE_TASK_SYSFS_TESTDATA_DIR)
  if (TryExtractFixture(IREE_TASK_SYSFS_TESTDATA_DIR, &dir)) {
    return dir;
  }
#endif

  if (const char* srcdir = std::getenv("TEST_SRCDIR")) {
    const char* suffixes[] = {
        "/runtime/src/iree/task/testdata/sysfs",
        "/_main/runtime/src/iree/task/testdata/sysfs",
        "/iree/runtime/src/iree/task/testdata/sysfs",
    };
    for (const char* suffix : suffixes) {
      if (TryExtractFixture(std::string(srcdir) + suffix, &dir)) {
        return dir;
      }
    }
  }

  if (TryExtractFixture("runtime/src/iree/task/testdata/sysfs", &dir)) {
    return dir;
  }

  ADD_FAILURE() << "unable to locate x86_hybrid_sparse_clusters fixture "
                   "(set IREE_SYSFS_TEST_ROOT or extract testdata tar.gz)";
  return "";
}

class TopologySysfsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    fixture_root_ = ResolveHybridFixtureRoot();
    ASSERT_FALSE(fixture_root_.empty());
    iree_sysfs_set_root_path_for_testing(fixture_root_.c_str());
  }

  void TearDown() override { iree_sysfs_set_root_path_for_testing(NULL); }

  std::string fixture_root_;
};

// T1 / A1: sparse cluster_ids must not drive node_count (single NUMA node0).
TEST_F(TopologySysfsTest, SparseClustersYieldSingleNumaNode) {
  EXPECT_EQ(1u, iree_task_topology_query_node_count());
  EXPECT_EQ(0u, iree_task_topology_query_current_node());
}

// T2 / A2: node 0 covers all logical CPUs, not a single sparse cluster.
TEST_F(TopologySysfsTest, PhysicalCoresOnNode0SpanAllCpus) {
  static constexpr iree_host_size_t kExpectedCores = 8;
  iree_task_topology_t topology;
  iree_task_topology_initialize(&topology);
  IREE_ASSERT_OK(iree_task_topology_initialize_from_physical_cores(
      /*node_id=*/0, IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_ANY,
      IREE_TASK_TOPOLOGY_DISTRIBUTION_SCATTER, kExpectedCores, &topology));
  EXPECT_EQ(kExpectedCores, iree_task_topology_group_count(&topology));

  // Affinity group still carries cluster_id (sparse), not dense node id.
  bool saw_sparse_cluster = false;
  for (iree_host_size_t i = 0; i < topology.group_count; ++i) {
    const uint32_t group = topology.groups[i].ideal_thread_affinity.group;
    if (group == 8 || group == 16 || group == 24 || group == 32) {
      saw_sparse_cluster = true;
    }
  }
  EXPECT_TRUE(saw_sparse_cluster);

  iree_task_topology_deinitialize(&topology);
}

// T3 / A3: NODE_ID_ANY must not under-count due to sparse cluster ids.
TEST_F(TopologySysfsTest, AllNodesDoesNotUndercountSparseClusters) {
  static constexpr iree_host_size_t kExpectedCores = 8;
  iree_task_topology_t topology;
  iree_task_topology_initialize(&topology);
  IREE_ASSERT_OK(iree_task_topology_initialize_from_physical_cores(
      IREE_TASK_TOPOLOGY_NODE_ID_ANY, IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_ANY,
      IREE_TASK_TOPOLOGY_DISTRIBUTION_SCATTER, kExpectedCores, &topology));
  EXPECT_EQ(kExpectedCores, iree_task_topology_group_count(&topology));
  iree_task_topology_deinitialize(&topology);
}

#else

TEST(TopologySysfsTest, PlatformDisabled) {}

#endif  // IREE_PLATFORM_LINUX && !IREE_PLATFORM_WASM

}  // namespace
