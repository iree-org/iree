// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <set>
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

//===----------------------------------------------------------------------===//
// A1 — ORACLE_TOPOLOGY.md §2 (issue #24761 hybrid paste)
//===----------------------------------------------------------------------===//
//
// MUST match oracle: 24 logical CPUs; 10 sparse cluster_ids listed below.
// DESIGN: node id = NUMA (else package), never raw cluster_id.
// Scaffolding: single node0 + package_id=0 + core_id/SMT layout (not reporter
// dumps — see ORACLE_TOPOLOGY.md §1/§3). Benchmark ms are NOT gtest gates.
//
// Properties: node_count ≠ |unique cluster_id|; affinity retains issue ids
// including ≥64 (mask-safety: large cluster_id is affinity, not node bit).

// Oracle unique cluster_id set — ORACLE_TOPOLOGY.md §2 / issue #24761 paste.
static const std::set<uint32_t> kIssue24761ClusterIds = {0,  8,  16, 24, 32,
                                                         40, 48, 56, 64, 72};

TEST_F(TopologySysfsFixtureTest, A1_IssueHybridOracle_NodeNotCluster) {
  UseFixture("x86_hybrid_sparse_clusters");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  const iree_host_size_t node_count = iree_task_topology_query_node_count();
  EXPECT_EQ(1u, node_count);
  // Property: must not treat each sparse cluster as a node.
  EXPECT_NE(node_count, kIssue24761ClusterIds.size());
  EXPECT_EQ(0u, iree_task_topology_query_current_node());

  iree_task_topology_t topology;
  EXPECT_EQ(16u, InitPhysicalCoreCount(/*node_id=*/0, 24, &topology));
  EXPECT_TRUE(AllGroupsIdAssigned(topology));
  const auto affinity = CollectAffinityGroups(topology);
  // Affinity retains issue cluster ids (incl. ≥64); not dense node ordinals.
  for (uint32_t cluster_id : kIssue24761ClusterIds) {
    EXPECT_TRUE(affinity.count(cluster_id))
        << "missing issue cluster_id=" << cluster_id;
  }
  for (uint32_t g : affinity) {
    EXPECT_TRUE(kIssue24761ClusterIds.count(g))
        << "unexpected affinity.group=" << g;
  }
  // Mask-safety discriminator: ids ≥64 appear as affinity.group only.
  EXPECT_TRUE(affinity.count(64));
  EXPECT_TRUE(affinity.count(72));
  iree_task_topology_deinitialize(&topology);

  EXPECT_EQ(16u, InitPhysicalCoreCount(IREE_TASK_TOPOLOGY_NODE_ID_ANY, 24,
                                       &topology));
  iree_task_topology_deinitialize(&topology);
}

//===----------------------------------------------------------------------===//
// A2a — mutation of A1: delete node/ → package fallback
//===----------------------------------------------------------------------===//
// Relative invariant only: single package ⇒ still one dense node; sparse
// cluster affinity preserved. Not a dual-socket dump.

TEST_F(TopologySysfsFixtureTest, A2a_NoNumaSinglePackage_FromA1) {
  UseFixture("x86_no_numa_single_package");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  EXPECT_EQ(1u, iree_task_topology_query_node_count());

  iree_task_topology_t topology;
  EXPECT_EQ(16u, InitPhysicalCoreCount(/*node_id=*/0, 24, &topology));
  const auto affinity = CollectAffinityGroups(topology);
  EXPECT_EQ(kIssue24761ClusterIds.size(), affinity.size());
  EXPECT_TRUE(affinity.count(0));
  EXPECT_TRUE(affinity.count(72));
  iree_task_topology_deinitialize(&topology);
}

//===----------------------------------------------------------------------===//
// A2b — mutation of A1: delete cluster_id files → affinity → package
//===----------------------------------------------------------------------===//
// Keeps single NUMA from A1 (does not invent dual-NUMA). Affinity.group falls
// back to physical_package_id (0) per DESIGN.

TEST_F(TopologySysfsFixtureTest, A2b_MissingClusterId_FromA1) {
  UseFixture("x86_missing_cluster_id");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  EXPECT_EQ(1u, iree_task_topology_query_node_count());

  iree_task_topology_t topology;
  EXPECT_EQ(16u, InitPhysicalCoreCount(/*node_id=*/0, 24, &topology));
  EXPECT_TRUE(AllGroupsIdAssigned(topology));
  for (iree_host_size_t i = 0; i < topology.group_count; ++i) {
    EXPECT_EQ(0u, topology.groups[i].ideal_thread_affinity.group);
  }
  iree_task_topology_deinitialize(&topology);
}

//===----------------------------------------------------------------------===//
// A0 — upstream Pixel6 capture (#22455)
//===----------------------------------------------------------------------===//
// Property: no node/ + single package → one node, not three dense clusters.

TEST_F(TopologySysfsFixtureTest, A0_Arm64PixelPackageFallback) {
  UseFixture("arm64_pixel6_tensor");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  const iree_host_size_t node_count = iree_task_topology_query_node_count();
  EXPECT_EQ(1u, node_count);

  iree_task_topology_t topology;
  EXPECT_EQ(
      8u, InitPhysicalCoreCount(IREE_TASK_TOPOLOGY_NODE_ID_ANY, 8, &topology));
  const auto affinity = CollectAffinityGroups(topology);
  // Pixel big.LITTLE clusters are dense 0/1/2 — still affinity-only.
  EXPECT_TRUE(affinity.count(0));
  EXPECT_TRUE(affinity.count(1));
  EXPECT_TRUE(affinity.count(2));
  EXPECT_NE(node_count, affinity.size());
  iree_task_topology_deinitialize(&topology);
}

//===----------------------------------------------------------------------===//
// P-* property test doubles — synthetic input to f(), NOT captured machines
//===----------------------------------------------------------------------===//
// See testdata/sysfs/README.md + process/CORPUS.md. Asserts are DESIGN
// invariants / membership properties of the mapping, not HW fidelity claims.

// P1 — two NUMA cpulists → dense nodes 0,1; filter preserves membership.
TEST_F(TopologySysfsFixtureTest, P1_DualNuma_DenseMembership) {
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

  EXPECT_EQ(
      8u, InitPhysicalCoreCount(IREE_TASK_TOPOLOGY_NODE_ID_ANY, 8, &topology));
  // Sparse clusters remain affinity-only; must not collapse to one cluster.
  EXPECT_GT(CollectAffinityGroups(topology).size(), 1u);
  EXPECT_NE(iree_task_topology_query_node_count(),
            CollectAffinityGroups(topology).size());
  iree_task_topology_deinitialize(&topology);
}

// P2 — sparse kernel node/online (0,2) → dense ordinals; raw kernel id ≠ bit.
TEST_F(TopologySysfsFixtureTest, P2_SparseKernelNuma_DenseRemap) {
  UseFixture("prop_sparse_kernel_numa");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  // online=0,2 → dense count 2 (kernel id 2 is not a third node).
  EXPECT_EQ(2u, iree_task_topology_query_node_count());

  const std::set<uint32_t> dense0 = {0, 1, 2, 3};  // kernel node0
  const std::set<uint32_t> dense1 = {4, 5, 6, 7};  // kernel node2

  iree_task_topology_t topology;
  EXPECT_EQ(4u, InitPhysicalCoreCount(/*node_id=*/0, 8, &topology));
  EXPECT_TRUE(SetIsSubset(CollectProcessors(topology), dense0));
  iree_task_topology_deinitialize(&topology);

  EXPECT_EQ(4u, InitPhysicalCoreCount(/*node_id=*/1, 8, &topology));
  EXPECT_TRUE(SetIsSubset(CollectProcessors(topology), dense1));
  iree_task_topology_deinitialize(&topology);

  // Misusing kernel id 2 as dense node id yields empty selection.
  EXPECT_EQ(0u, InitPhysicalCoreCount(/*node_id=*/2, 8, &topology));
  iree_task_topology_deinitialize(&topology);
}

// P3 — WITH node/: must NOT collapse to single package even if package_id=0.
TEST_F(TopologySysfsFixtureTest, P3_NumaOverPackage_NoCollapse) {
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

// P4 — no node/: multi-package dense fallback (package-only-when-no-node).
TEST_F(TopologySysfsFixtureTest, P4_PackageMulti_WhenNoNode) {
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

// P5 — empty cpulist on an online node: counted, maps no CPUs.
TEST_F(TopologySysfsFixtureTest, P5_EmptyCpulist_MapsNone) {
  UseFixture("prop_empty_cpulist");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  EXPECT_EQ(2u, iree_task_topology_query_node_count());

  iree_task_topology_t topology;
  EXPECT_EQ(0u, InitPhysicalCoreCount(/*node_id=*/0, 8, &topology));
  iree_task_topology_deinitialize(&topology);
  EXPECT_EQ(8u, InitPhysicalCoreCount(/*node_id=*/1, 8, &topology));
  iree_task_topology_deinitialize(&topology);
}

// P6 — uncovered CPUs: graceful degrade keeps them when map fails.
TEST_F(TopologySysfsFixtureTest, P6_UncoveredCpu_DegradeKeep) {
  UseFixture("prop_uncovered_cpu");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  EXPECT_EQ(2u, iree_task_topology_query_node_count());

  iree_task_topology_t topology;
  EXPECT_EQ(
      8u, InitPhysicalCoreCount(IREE_TASK_TOPOLOGY_NODE_ID_ANY, 8, &topology));
  iree_task_topology_deinitialize(&topology);

  // node0 maps 0-2; unmapped 6-7 kept by degrade → 5 physical cores.
  EXPECT_EQ(5u, InitPhysicalCoreCount(/*node_id=*/0, 8, &topology));
  {
    const auto cpus = CollectProcessors(topology);
    EXPECT_TRUE(cpus.count(6));
    EXPECT_TRUE(cpus.count(7));
  }
  iree_task_topology_deinitialize(&topology);

  // node1 maps 3-5; same unmapped 6-7 kept → 5.
  EXPECT_EQ(5u, InitPhysicalCoreCount(/*node_id=*/1, 8, &topology));
  {
    const auto cpus = CollectProcessors(topology);
    EXPECT_TRUE(cpus.count(6));
    EXPECT_TRUE(cpus.count(7));
  }
  iree_task_topology_deinitialize(&topology);
}

// P7 — NUMA path without physical_package_id (missing-package edge).
TEST_F(TopologySysfsFixtureTest, P7_NumaNoPackage_StillDense) {
  UseFixture("prop_numa_no_package");
  ASSERT_FALSE(::testing::Test::HasFatalFailure());

  EXPECT_EQ(2u, iree_task_topology_query_node_count());

  iree_task_topology_t topology;
  EXPECT_EQ(4u, InitPhysicalCoreCount(/*node_id=*/0, 8, &topology));
  EXPECT_TRUE(SetIsSubset(CollectProcessors(topology), {0, 1, 2, 3}));
  // Affinity falls back to package when package file absent → group unset/0.
  // Mapping itself must still succeed via NUMA.
  iree_task_topology_deinitialize(&topology);
  EXPECT_EQ(4u, InitPhysicalCoreCount(/*node_id=*/1, 8, &topology));
  EXPECT_TRUE(SetIsSubset(CollectProcessors(topology), {4, 5, 6, 7}));
  iree_task_topology_deinitialize(&topology);
}

#else

TEST(TopologySysfsTest, PlatformDisabled) {}

#endif  // IREE_PLATFORM_LINUX && !IREE_PLATFORM_WASM

}  // namespace
