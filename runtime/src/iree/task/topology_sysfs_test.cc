// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "iree/base/internal/sysfs.h"
#include "iree/task/topology.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

#if defined(IREE_PLATFORM_LINUX) && !defined(IREE_PLATFORM_WASM)

using namespace iree::testing::status;

//===----------------------------------------------------------------------===//
// Fixture resolution (tar.gz under testdata/sysfs)
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
  // Shared override: absolute path to an already-extracted fixture directory.
  if (const char* env = std::getenv("IREE_SYSFS_TEST_ROOT")) {
    if (FixturePresentReadable(env)) {
      // If the env points at the named fixture or its parent sysfs dir.
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
                << "' (set IREE_SYSFS_TEST_ROOT or extract testdata tar.gz)";
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

static bool AffinityGroupsInclude(const iree_task_topology_t& topology,
                                  const std::vector<uint32_t>& expected) {
  for (uint32_t want : expected) {
    bool found = false;
    for (iree_host_size_t i = 0; i < topology.group_count; ++i) {
      if (topology.groups[i].ideal_thread_affinity.group == want) {
        found = true;
        break;
      }
    }
    if (!found) return false;
  }
  return true;
}

static bool AllGroupsIdAssigned(const iree_task_topology_t& topology) {
  for (iree_host_size_t i = 0; i < topology.group_count; ++i) {
    if (!topology.groups[i].ideal_thread_affinity.id_assigned) return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// G1 — reporter hybrid sparse clusters (single NUMA)
//===----------------------------------------------------------------------===//

TEST_F(TopologySysfsFixtureTest, G1_HybridSparseClusters_SingleNuma) {
  UseFixture("x86_hybrid_sparse_clusters");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  EXPECT_EQ(1u, iree_task_topology_query_node_count());
  EXPECT_EQ(0u, iree_task_topology_query_current_node());

  iree_task_topology_t topology;
  EXPECT_EQ(8u, InitPhysicalCoreCount(/*node_id=*/0, 8, &topology));
  EXPECT_TRUE(AllGroupsIdAssigned(topology));
  EXPECT_TRUE(AffinityGroupsInclude(topology, {8, 16, 24, 32}));
  iree_task_topology_deinitialize(&topology);

  EXPECT_EQ(
      8u, InitPhysicalCoreCount(IREE_TASK_TOPOLOGY_NODE_ID_ANY, 8, &topology));
  iree_task_topology_deinitialize(&topology);
}

//===----------------------------------------------------------------------===//
// G2 — hybrid + SMT
//===----------------------------------------------------------------------===//

TEST_F(TopologySysfsFixtureTest, G2_HybridSmtSparseClusters) {
  UseFixture("x86_hybrid_smt_sparse_clusters");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  EXPECT_EQ(1u, iree_task_topology_query_node_count());

  iree_task_topology_t topology;
  // 8 logical CPUs / 2 threads per core → 4 physical cores.
  EXPECT_EQ(4u, InitPhysicalCoreCount(/*node_id=*/0, 8, &topology));
  EXPECT_TRUE(AffinityGroupsInclude(topology, {0, 8, 16, 24}));
  iree_task_topology_deinitialize(&topology);
}

//===----------------------------------------------------------------------===//
// G3 — dual socket × dual NUMA
//===----------------------------------------------------------------------===//

TEST_F(TopologySysfsFixtureTest, G3_DualNumaSparseClusters) {
  UseFixture("x86_dual_numa_sparse_clusters");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  EXPECT_EQ(2u, iree_task_topology_query_node_count());

  iree_task_topology_t topology;
  EXPECT_EQ(4u, InitPhysicalCoreCount(/*node_id=*/0, 8, &topology));
  iree_task_topology_deinitialize(&topology);
  EXPECT_EQ(4u, InitPhysicalCoreCount(/*node_id=*/1, 8, &topology));
  iree_task_topology_deinitialize(&topology);
  EXPECT_EQ(
      8u, InitPhysicalCoreCount(IREE_TASK_TOPOLOGY_NODE_ID_ANY, 8, &topology));
  // Must not collapse to a single sparse cluster.
  EXPECT_GT(topology.group_count, 1u);
  iree_task_topology_deinitialize(&topology);
}

//===----------------------------------------------------------------------===//
// G4 — single socket multi-NUMA (SNC); package must not override NUMA
//===----------------------------------------------------------------------===//

TEST_F(TopologySysfsFixtureTest, G4_SingleSocketMultiNuma) {
  UseFixture("x86_single_socket_multi_numa");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  // NUMA wins over the single package_id=0 on all CPUs.
  EXPECT_EQ(2u, iree_task_topology_query_node_count());

  iree_task_topology_t topology;
  EXPECT_EQ(4u, InitPhysicalCoreCount(/*node_id=*/0, 8, &topology));
  iree_task_topology_deinitialize(&topology);
  EXPECT_EQ(4u, InitPhysicalCoreCount(/*node_id=*/1, 8, &topology));
  iree_task_topology_deinitialize(&topology);
}

//===----------------------------------------------------------------------===//
// G5 + G15 — sparse kernel NUMA ids; dense ordinals; OOR misuse
//===----------------------------------------------------------------------===//

TEST_F(TopologySysfsFixtureTest, G5_SparseKernelNumaDenseRemap) {
  UseFixture("x86_sparse_kernel_numa");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  // online=0,2 → dense count 2 (not kernel id 2 as a third node).
  EXPECT_EQ(2u, iree_task_topology_query_node_count());

  iree_task_topology_t topology;
  EXPECT_EQ(4u, InitPhysicalCoreCount(/*node_id=*/0, 8, &topology));
  iree_task_topology_deinitialize(&topology);
  // Dense 1 == kernel node 2's CPUs.
  EXPECT_EQ(4u, InitPhysicalCoreCount(/*node_id=*/1, 8, &topology));
  iree_task_topology_deinitialize(&topology);
  // OOR / kernel-id misuse: treating 2 as dense id yields no groups.
  EXPECT_EQ(0u, InitPhysicalCoreCount(/*node_id=*/2, 8, &topology));
  iree_task_topology_deinitialize(&topology);
}

//===----------------------------------------------------------------------===//
// G6 — package fallback (no node/)
//===----------------------------------------------------------------------===//

TEST_F(TopologySysfsFixtureTest, G6_MultiPackageNoNuma) {
  UseFixture("x86_multi_package_no_numa");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  EXPECT_EQ(2u, iree_task_topology_query_node_count());

  iree_task_topology_t topology;
  EXPECT_EQ(4u, InitPhysicalCoreCount(/*node_id=*/0, 8, &topology));
  iree_task_topology_deinitialize(&topology);
  EXPECT_EQ(4u, InitPhysicalCoreCount(/*node_id=*/1, 8, &topology));
  iree_task_topology_deinitialize(&topology);
}

//===----------------------------------------------------------------------===//
// G7 — no NUMA, single package, sparse clusters
//===----------------------------------------------------------------------===//

TEST_F(TopologySysfsFixtureTest, G7_NoNumaSinglePackage) {
  UseFixture("x86_no_numa_single_package");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  EXPECT_EQ(1u, iree_task_topology_query_node_count());

  iree_task_topology_t topology;
  EXPECT_EQ(8u, InitPhysicalCoreCount(/*node_id=*/0, 8, &topology));
  EXPECT_TRUE(AffinityGroupsInclude(topology, {0, 8, 16, 24, 32}));
  iree_task_topology_deinitialize(&topology);
}

//===----------------------------------------------------------------------===//
// G8 — ARM Pixel-like (existing fixture)
//===----------------------------------------------------------------------===//

TEST_F(TopologySysfsFixtureTest, G8_Arm64PixelPackageFallback) {
  UseFixture("arm64_pixel6_tensor");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  // No node/; single package → one dense node (not 3 clusters).
  EXPECT_EQ(1u, iree_task_topology_query_node_count());

  iree_task_topology_t topology;
  EXPECT_EQ(
      8u, InitPhysicalCoreCount(IREE_TASK_TOPOLOGY_NODE_ID_ANY, 8, &topology));
  EXPECT_TRUE(AffinityGroupsInclude(topology, {0, 1, 2}));
  iree_task_topology_deinitialize(&topology);
}

//===----------------------------------------------------------------------===//
// G9 + G16 — missing cluster_id → affinity falls back to package
//===----------------------------------------------------------------------===//

TEST_F(TopologySysfsFixtureTest, G9_MissingClusterIdPackageAffinity) {
  UseFixture("x86_missing_cluster_id");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  EXPECT_EQ(2u, iree_task_topology_query_node_count());

  iree_task_topology_t topology;
  EXPECT_EQ(4u, InitPhysicalCoreCount(/*node_id=*/0, 8, &topology));
  EXPECT_TRUE(AllGroupsIdAssigned(topology));
  // cluster_id absent → physical_package_id used for affinity.group.
  EXPECT_TRUE(AffinityGroupsInclude(topology, {0}));
  for (iree_host_size_t i = 0; i < topology.group_count; ++i) {
    EXPECT_EQ(0u, topology.groups[i].ideal_thread_affinity.group);
  }
  iree_task_topology_deinitialize(&topology);

  EXPECT_EQ(4u, InitPhysicalCoreCount(/*node_id=*/1, 8, &topology));
  for (iree_host_size_t i = 0; i < topology.group_count; ++i) {
    EXPECT_EQ(1u, topology.groups[i].ideal_thread_affinity.group);
  }
  iree_task_topology_deinitialize(&topology);
}

//===----------------------------------------------------------------------===//
// G10 — NUMA without physical_package_id
//===----------------------------------------------------------------------===//

TEST_F(TopologySysfsFixtureTest, G10_NumaMissingPackage) {
  UseFixture("x86_numa_missing_package");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  EXPECT_EQ(2u, iree_task_topology_query_node_count());

  iree_task_topology_t topology;
  EXPECT_EQ(4u, InitPhysicalCoreCount(/*node_id=*/0, 8, &topology));
  iree_task_topology_deinitialize(&topology);
  EXPECT_EQ(4u, InitPhysicalCoreCount(/*node_id=*/1, 8, &topology));
  iree_task_topology_deinitialize(&topology);
}

//===----------------------------------------------------------------------===//
// G11 — bare minimal (no NUMA, no package, no cluster)
//===----------------------------------------------------------------------===//

TEST_F(TopologySysfsFixtureTest, G11_BareMinimalDegenerate) {
  UseFixture("x86_bare_minimal");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  EXPECT_EQ(1u, iree_task_topology_query_node_count());

  iree_task_topology_t topology;
  EXPECT_EQ(4u, InitPhysicalCoreCount(/*node_id=*/0, 4, &topology));
  EXPECT_TRUE(AllGroupsIdAssigned(topology));
  iree_task_topology_deinitialize(&topology);
}

//===----------------------------------------------------------------------===//
// G12 — partial cpu dirs
//===----------------------------------------------------------------------===//

TEST_F(TopologySysfsFixtureTest, G12_PartialCpus) {
  UseFixture("x86_partial_cpus");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  EXPECT_EQ(1u, iree_task_topology_query_node_count());

  iree_task_topology_t topology;
  // Only even CPUs have topology → 4 physical cores.
  EXPECT_EQ(4u, InitPhysicalCoreCount(/*node_id=*/0, 8, &topology));
  iree_task_topology_deinitialize(&topology);
}

//===----------------------------------------------------------------------===//
// G13 — empty cpulist on one online node
//===----------------------------------------------------------------------===//

TEST_F(TopologySysfsFixtureTest, G13_EmptyCpulistNode) {
  UseFixture("x86_empty_cpulist_node");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  // Both node dirs readable → count 2; empty node maps no CPUs.
  EXPECT_EQ(2u, iree_task_topology_query_node_count());

  iree_task_topology_t topology;
  EXPECT_EQ(0u, InitPhysicalCoreCount(/*node_id=*/0, 8, &topology));
  iree_task_topology_deinitialize(&topology);
  EXPECT_EQ(8u, InitPhysicalCoreCount(/*node_id=*/1, 8, &topology));
  iree_task_topology_deinitialize(&topology);
}

//===----------------------------------------------------------------------===//
// G14 — cluster ids ≥ 64 must not become node count
//===----------------------------------------------------------------------===//

TEST_F(TopologySysfsFixtureTest, G14_LargeClusterIdsNotNodes) {
  UseFixture("x86_large_cluster_ids");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  EXPECT_EQ(1u, iree_task_topology_query_node_count());

  iree_task_topology_t topology;
  EXPECT_EQ(8u, InitPhysicalCoreCount(/*node_id=*/0, 8, &topology));
  EXPECT_TRUE(AffinityGroupsInclude(topology, {64, 80, 96, 112}));
  for (iree_host_size_t i = 0; i < topology.group_count; ++i) {
    EXPECT_GE(topology.groups[i].ideal_thread_affinity.group, 64u);
  }
  iree_task_topology_deinitialize(&topology);
}

//===----------------------------------------------------------------------===//
// G17 — uncovered CPUs degrade (kept when map fails)
//===----------------------------------------------------------------------===//

TEST_F(TopologySysfsFixtureTest, G17_UncoveredCpuDegradeKeep) {
  UseFixture("x86_numa_uncovered_cpu");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  EXPECT_EQ(2u, iree_task_topology_query_node_count());

  iree_task_topology_t topology;
  EXPECT_EQ(
      8u, InitPhysicalCoreCount(IREE_TASK_TOPOLOGY_NODE_ID_ANY, 8, &topology));
  iree_task_topology_deinitialize(&topology);

  // Mapped node0 CPUs (0-2) plus unmapped 6-7 kept by degrade path.
  EXPECT_EQ(5u, InitPhysicalCoreCount(/*node_id=*/0, 8, &topology));
  iree_task_topology_deinitialize(&topology);
  EXPECT_EQ(5u, InitPhysicalCoreCount(/*node_id=*/1, 8, &topology));
  iree_task_topology_deinitialize(&topology);
}

//===----------------------------------------------------------------------===//
// Table-driven smoke: every fixture resolves + node_count matches
//===----------------------------------------------------------------------===//

struct FixtureNodeExpectation {
  const char* name;
  iree_host_size_t node_count;
};

TEST_F(TopologySysfsFixtureTest, AllFixtures_NodeCountTable) {
  static const FixtureNodeExpectation kCases[] = {
      {"x86_hybrid_sparse_clusters", 1},
      {"x86_hybrid_smt_sparse_clusters", 1},
      {"x86_dual_numa_sparse_clusters", 2},
      {"x86_single_socket_multi_numa", 2},
      {"x86_sparse_kernel_numa", 2},
      {"x86_multi_package_no_numa", 2},
      {"x86_no_numa_single_package", 1},
      {"arm64_pixel6_tensor", 1},
      {"x86_missing_cluster_id", 2},
      {"x86_numa_missing_package", 2},
      {"x86_bare_minimal", 1},
      {"x86_partial_cpus", 1},
      {"x86_empty_cpulist_node", 2},
      {"x86_large_cluster_ids", 1},
      {"x86_numa_uncovered_cpu", 2},
  };
  for (const auto& c : kCases) {
    SCOPED_TRACE(c.name);
    UseFixture(c.name);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    EXPECT_EQ(c.node_count, iree_task_topology_query_node_count());
    iree_sysfs_set_root_path_for_testing(NULL);
  }
}

#else

TEST(TopologySysfsTest, PlatformDisabled) {}

#endif  // IREE_PLATFORM_LINUX && !IREE_PLATFORM_WASM

}  // namespace
