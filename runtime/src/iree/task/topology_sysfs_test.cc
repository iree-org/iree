// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// TDD seams for #24761 (confirmed):
//   S1 — query_node_count / query_current_node
//   S2 — initialize_from_physical_cores → group_count + processor set
//   S3 — initialize_from_flags (public flag-driven path; default nodes=current)
//
// Oracle literals: issue #24761 paste / ORACLE_TOPOLOGY.md (independent of
// how fixtures were authored). Plain-text trees under testdata/sysfs/; A0
// Pixel remains the upstream .tar.gz from #22455.

#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "iree/base/internal/sysfs.h"
#include "iree/task/api.h"
#include "iree/task/topology.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

#if defined(IREE_PLATFORM_LINUX) && !defined(IREE_PLATFORM_WASM)

using namespace iree::testing::status;

//===----------------------------------------------------------------------===//
// Fixture resolution (plain trees preferred; Pixel A0 may still be tar.gz)
//===----------------------------------------------------------------------===//

static bool FixturePresentReadable(const std::string& dir) {
  const std::string present = dir + "/cpu/present";
  return access(present.c_str(), R_OK) == 0;
}

static bool TryExtractFixture(const std::string& base, const char* name,
                              std::string* out_dir) {
  const std::string dir = std::string(base) + "/" + name;
  if (FixturePresentReadable(dir)) {
    *out_dir = dir;
    return true;
  }
  const std::string tar = std::string(base) + "/" + name + ".tar.gz";
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

static std::string ResolveFixtureRoot(const char* fixture_name) {
  const std::string env_key =
      std::string("IREE_SYSFS_TEST_ROOT_") + fixture_name;
  if (const char* env = std::getenv(env_key.c_str())) {
    return std::string(env);
  }
  if (const char* env = std::getenv("IREE_SYSFS_TEST_ROOT")) {
    if (FixturePresentReadable(env)) {
      const std::string as_is(env);
      if (as_is.size() >= std::strlen(fixture_name) &&
          as_is.compare(as_is.size() - std::strlen(fixture_name),
                        std::strlen(fixture_name), fixture_name) == 0) {
        return as_is;
      }
    }
  }

  std::string dir;
#if defined(IREE_TASK_SYSFS_TESTDATA_DIR)
  if (TryExtractFixture(IREE_TASK_SYSFS_TESTDATA_DIR, fixture_name, &dir)) {
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
      if (TryExtractFixture(std::string(srcdir) + suffix, fixture_name, &dir)) {
        return dir;
      }
    }
  }

  if (TryExtractFixture("runtime/src/iree/task/testdata/sysfs", fixture_name,
                        &dir)) {
    return dir;
  }

  ADD_FAILURE() << "unable to locate fixture '" << fixture_name
                << "' (set IREE_SYSFS_TEST_ROOT or check testdata/sysfs)";
  return "";
}

class TopologySysfsFixtureTest : public ::testing::Test {
 protected:
  void TearDown() override { iree_sysfs_set_root_path_for_testing(NULL); }

  void UseFixture(const char* fixture_name) {
    fixture_root_ = ResolveFixtureRoot(fixture_name);
    ASSERT_FALSE(fixture_root_.empty());
    iree_sysfs_set_root_path_for_testing(fixture_root_.c_str());
  }

  std::string fixture_root_;
};

static iree_host_size_t InitPhysicalCoreCount(
    iree_task_topology_node_id_t node_id, iree_host_size_t max_cores,
    iree_task_topology_t* out_topology) {
  iree_task_topology_initialize(out_topology);
  IREE_EXPECT_OK(iree_task_topology_initialize_from_physical_cores(
      node_id, IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_ANY,
      IREE_TASK_TOPOLOGY_DISTRIBUTION_SCATTER, max_cores, out_topology));
  return iree_task_topology_group_count(out_topology);
}

static std::set<uint32_t> CollectAffinityGroups(
    const iree_task_topology_t& topology) {
  std::set<uint32_t> groups;
  for (iree_host_size_t i = 0; i < topology.group_count; ++i) {
    groups.insert(topology.groups[i].ideal_thread_affinity.group);
  }
  return groups;
}

static std::set<uint32_t> CollectProcessors(
    const iree_task_topology_t& topology) {
  std::set<uint32_t> cpus;
  for (iree_host_size_t i = 0; i < topology.group_count; ++i) {
    cpus.insert(topology.groups[i].processor_index);
  }
  return cpus;
}

static bool AllGroupsIdAssigned(const iree_task_topology_t& topology) {
  for (iree_host_size_t i = 0; i < topology.group_count; ++i) {
    if (!topology.groups[i].ideal_thread_affinity.id_assigned) return false;
  }
  return true;
}

static bool SetIsSubset(const std::set<uint32_t>& subset,
                        const std::set<uint32_t>& universe) {
  for (uint32_t v : subset) {
    if (!universe.count(v)) return false;
  }
  return true;
}

// Oracle unique cluster_id set — issue #24761 paste / ORACLE_TOPOLOGY.md §2.
// Independent literals (not recomputed from fixture-authoring scripts).
static const std::set<uint32_t> kIssue24761ClusterIds = {0,  8,  16, 24, 32,
                                                         40, 48, 56, 64, 72};

// Physical cores on A1 harness (24 logical CPUs, SMT pairs → 16 cores).
static constexpr iree_host_size_t kIssue24761PhysicalCoreCount = 16;

//===----------------------------------------------------------------------===//
// S1 — node identity queries (must not treat cluster_id as node)
//===----------------------------------------------------------------------===//

TEST_F(TopologySysfsFixtureTest, S1_A1_NodeCountIsNumaNotClusterCount) {
  UseFixture("x86_hybrid_sparse_clusters");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  const iree_host_size_t node_count = iree_task_topology_query_node_count();
  // Oracle: single NUMA on reporter-class hybrid → exactly one task node.
  EXPECT_EQ(1u, node_count);
  // Anti-regression: must not equal unique sparse cluster count (10).
  EXPECT_NE(node_count, kIssue24761ClusterIds.size());
}

TEST_F(TopologySysfsFixtureTest, S1_A1_CurrentNodeIsDenseZero) {
  UseFixture("x86_hybrid_sparse_clusters");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  // With a single NUMA node0, current maps to dense ordinal 0 — not a sparse
  // cluster_id such as 8/16/64.
  EXPECT_EQ(0u, iree_task_topology_query_current_node());
}

TEST_F(TopologySysfsFixtureTest, S1_P2_SparseKernelNuma_DenseCount) {
  UseFixture("prop_sparse_kernel_numa");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  // online=0,2 → two dense nodes (not three, not raw id 2 as a third slot).
  EXPECT_EQ(2u, iree_task_topology_query_node_count());
}

TEST_F(TopologySysfsFixtureTest, S1_A0_PixelPackageFallback_OneNode) {
  UseFixture("arm64_pixel6_tensor");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  EXPECT_EQ(1u, iree_task_topology_query_node_count());
}

//===----------------------------------------------------------------------===//
// S2 — initialize_from_physical_cores (worker pool size + membership)
//===----------------------------------------------------------------------===//

TEST_F(TopologySysfsFixtureTest, S2_A1_CurrentNodeDoesNotCollapseToOneCluster) {
  UseFixture("x86_hybrid_sparse_clusters");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  iree_task_topology_t topology;
  // Bug symptom: cluster-as-node left ~1–2 workers. Fixed: full package/NUMA.
  const iree_host_size_t groups =
      InitPhysicalCoreCount(/*node_id=*/0, 24, &topology);
  EXPECT_EQ(kIssue24761PhysicalCoreCount, groups);
  EXPECT_GT(groups, 2u);  // hard anti-collapse vs single-cluster pin
  EXPECT_TRUE(AllGroupsIdAssigned(topology));
  iree_task_topology_deinitialize(&topology);
}

TEST_F(TopologySysfsFixtureTest, S2_A1_AffinityRetainsIssueClusterIds) {
  UseFixture("x86_hybrid_sparse_clusters");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  iree_task_topology_t topology;
  ASSERT_EQ(kIssue24761PhysicalCoreCount,
            InitPhysicalCoreCount(/*node_id=*/0, 24, &topology));
  const auto affinity = CollectAffinityGroups(topology);
  for (uint32_t cluster_id : kIssue24761ClusterIds) {
    EXPECT_TRUE(affinity.count(cluster_id))
        << "missing issue cluster_id=" << cluster_id;
  }
  for (uint32_t g : affinity) {
    EXPECT_TRUE(kIssue24761ClusterIds.count(g))
        << "unexpected affinity.group=" << g;
  }
  // Mask-safety: ids ≥64 appear as affinity.group metadata, not as node bits.
  EXPECT_TRUE(affinity.count(64));
  EXPECT_TRUE(affinity.count(72));
  iree_task_topology_deinitialize(&topology);
}

TEST_F(TopologySysfsFixtureTest, S2_P1_DualNuma_Membership) {
  UseFixture("prop_dual_numa");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  EXPECT_EQ(2u, iree_task_topology_query_node_count());

  const std::set<uint32_t> node0_cpus = {0, 1, 2, 3};
  const std::set<uint32_t> node1_cpus = {4, 5, 6, 7};

  iree_task_topology_t topology;
  EXPECT_EQ(4u, InitPhysicalCoreCount(/*node_id=*/0, 8, &topology));
  EXPECT_TRUE(SetIsSubset(CollectProcessors(topology), node0_cpus));
  iree_task_topology_deinitialize(&topology);

  EXPECT_EQ(4u, InitPhysicalCoreCount(/*node_id=*/1, 8, &topology));
  EXPECT_TRUE(SetIsSubset(CollectProcessors(topology), node1_cpus));
  iree_task_topology_deinitialize(&topology);
}

TEST_F(TopologySysfsFixtureTest, S2_P2_RawKernelNodeIdSelectsNothing) {
  UseFixture("prop_sparse_kernel_numa");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  iree_task_topology_t topology;
  // Dense 0/1 are valid; misusing kernel id 2 as dense id → empty.
  EXPECT_EQ(0u, InitPhysicalCoreCount(/*node_id=*/2, 8, &topology));
  iree_task_topology_deinitialize(&topology);
}

TEST_F(TopologySysfsFixtureTest, S2_P3_NumaWinsOverPackage) {
  UseFixture("prop_numa_over_package");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  EXPECT_EQ(2u, iree_task_topology_query_node_count());

  iree_task_topology_t topology;
  EXPECT_EQ(4u, InitPhysicalCoreCount(/*node_id=*/0, 8, &topology));
  EXPECT_TRUE(SetIsSubset(CollectProcessors(topology), {0, 1, 2, 3}));
  iree_task_topology_deinitialize(&topology);
  EXPECT_EQ(4u, InitPhysicalCoreCount(/*node_id=*/1, 8, &topology));
  EXPECT_TRUE(SetIsSubset(CollectProcessors(topology), {4, 5, 6, 7}));
  iree_task_topology_deinitialize(&topology);
}

TEST_F(TopologySysfsFixtureTest, S2_P4_PackageMulti_WhenNoNode) {
  UseFixture("prop_package_multi");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  EXPECT_EQ(2u, iree_task_topology_query_node_count());

  iree_task_topology_t topology;
  EXPECT_EQ(4u, InitPhysicalCoreCount(/*node_id=*/0, 8, &topology));
  EXPECT_TRUE(SetIsSubset(CollectProcessors(topology), {0, 1, 2, 3}));
  iree_task_topology_deinitialize(&topology);
  EXPECT_EQ(4u, InitPhysicalCoreCount(/*node_id=*/1, 8, &topology));
  EXPECT_TRUE(SetIsSubset(CollectProcessors(topology), {4, 5, 6, 7}));
  iree_task_topology_deinitialize(&topology);
}

TEST_F(TopologySysfsFixtureTest, S2_P5_EmptyCpulist_MapsNone) {
  UseFixture("prop_empty_cpulist");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  EXPECT_EQ(2u, iree_task_topology_query_node_count());

  iree_task_topology_t topology;
  EXPECT_EQ(0u, InitPhysicalCoreCount(/*node_id=*/0, 8, &topology));
  iree_task_topology_deinitialize(&topology);
  EXPECT_EQ(8u, InitPhysicalCoreCount(/*node_id=*/1, 8, &topology));
  iree_task_topology_deinitialize(&topology);
}

TEST_F(TopologySysfsFixtureTest, S2_P6_UncoveredCpu_DegradeKeep) {
  UseFixture("prop_uncovered_cpu");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  iree_task_topology_t topology;
  EXPECT_EQ(5u, InitPhysicalCoreCount(/*node_id=*/0, 8, &topology));
  {
    const auto cpus = CollectProcessors(topology);
    EXPECT_TRUE(cpus.count(6));
    EXPECT_TRUE(cpus.count(7));
  }
  iree_task_topology_deinitialize(&topology);
}

TEST_F(TopologySysfsFixtureTest, S2_P7_NumaWithoutPackage) {
  UseFixture("prop_numa_no_package");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  EXPECT_EQ(2u, iree_task_topology_query_node_count());

  iree_task_topology_t topology;
  EXPECT_EQ(4u, InitPhysicalCoreCount(/*node_id=*/0, 8, &topology));
  EXPECT_TRUE(SetIsSubset(CollectProcessors(topology), {0, 1, 2, 3}));
  iree_task_topology_deinitialize(&topology);
}

TEST_F(TopologySysfsFixtureTest, S2_A2a_NoNuma_PackageFallback) {
  UseFixture("x86_no_numa_single_package");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  EXPECT_EQ(1u, iree_task_topology_query_node_count());

  iree_task_topology_t topology;
  EXPECT_EQ(kIssue24761PhysicalCoreCount,
            InitPhysicalCoreCount(/*node_id=*/0, 24, &topology));
  EXPECT_TRUE(CollectAffinityGroups(topology).count(72));
  iree_task_topology_deinitialize(&topology);
}

TEST_F(TopologySysfsFixtureTest, S2_A2b_MissingCluster_AffinityPackage) {
  UseFixture("x86_missing_cluster_id");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  iree_task_topology_t topology;
  EXPECT_EQ(kIssue24761PhysicalCoreCount,
            InitPhysicalCoreCount(/*node_id=*/0, 24, &topology));
  for (iree_host_size_t i = 0; i < topology.group_count; ++i) {
    EXPECT_EQ(0u, topology.groups[i].ideal_thread_affinity.group);
  }
  iree_task_topology_deinitialize(&topology);
}

TEST_F(TopologySysfsFixtureTest, S2_A0_Pixel_NotThreeClusterNodes) {
  UseFixture("arm64_pixel6_tensor");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  iree_task_topology_t topology;
  EXPECT_EQ(
      8u, InitPhysicalCoreCount(IREE_TASK_TOPOLOGY_NODE_ID_ANY, 8, &topology));
  const auto affinity = CollectAffinityGroups(topology);
  EXPECT_TRUE(affinity.count(0));
  EXPECT_TRUE(affinity.count(1));
  EXPECT_TRUE(affinity.count(2));
  EXPECT_NE(iree_task_topology_query_node_count(), affinity.size());
  iree_task_topology_deinitialize(&topology);
}

//===----------------------------------------------------------------------===//
// S3 — public flags path (initialize_from_flags; default mode=physical_cores)
//===----------------------------------------------------------------------===//

TEST_F(TopologySysfsFixtureTest, S3_A1_FlagsPathMatchesPhysicalCoresOnCurrent) {
  UseFixture("x86_hybrid_sparse_clusters");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  // Mirrors api.c default: nodes=current → one topology for current node.
  const iree_task_topology_node_id_t current =
      iree_task_topology_query_current_node();
  EXPECT_EQ(0u, current);

  iree_task_topology_t topology;
  iree_task_topology_initialize(&topology);
  IREE_ASSERT_OK(iree_task_topology_initialize_from_flags(current, &topology));
  // Same anti-collapse gate as S2: must not be a single-cluster worker pool.
  EXPECT_EQ(kIssue24761PhysicalCoreCount, topology.group_count);
  EXPECT_GT(topology.group_count, 2u);
  iree_task_topology_deinitialize(&topology);
}

TEST_F(TopologySysfsFixtureTest, S3_P1_FlagsPathRespectsDenseNodeOne) {
  UseFixture("prop_dual_numa");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  iree_task_topology_t topology;
  iree_task_topology_initialize(&topology);
  IREE_ASSERT_OK(
      iree_task_topology_initialize_from_flags(/*node_id=*/1, &topology));
  EXPECT_EQ(4u, topology.group_count);
  EXPECT_TRUE(SetIsSubset(CollectProcessors(topology), {4, 5, 6, 7}));
  iree_task_topology_deinitialize(&topology);
}

#else

TEST(TopologySysfsTest, PlatformDisabled) {}

#endif  // IREE_PLATFORM_LINUX && !IREE_PLATFORM_WASM

}  // namespace
