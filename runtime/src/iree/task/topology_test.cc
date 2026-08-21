// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/task/topology.h"

#include <cstddef>

#include "iree/base/target_platform.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

// Linux sysfs topology tests against only compile on Linux.
#if defined(IREE_PLATFORM_LINUX) && !defined(IREE_PLATFORM_WASM)
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "iree/base/tooling/flags.h"
#include "iree/task/api.h"  // iree_task_topology_select_nodes
#define IREE_TASK_TOPOLOGY_TEST_HAVE_SYSFS_MOCK 1
#endif  // IREE_PLATFORM_LINUX && !IREE_PLATFORM_WASM

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
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_task_topology_push_group(&topology, &extra_group));

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
    // Cache sizes should be populated (either from platform queries or
    // conservative fallback values). The minimum acceptable values are the
    // fallback defaults (32KB L1, 128KB L2).
    EXPECT_GE(group->caches.l1_data, 32u * 1024u);
    EXPECT_GE(group->caches.l2_data, 128u * 1024u);
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
      IREE_TASK_TOPOLOGY_DISTRIBUTION_SCATTER, kMaxGroupCount, &topology));
  EnsureTopologyValid(kMaxGroupCount, &topology);
  iree_task_topology_deinitialize(&topology);
}

//===----------------------------------------------------------------------===//
// Sysfs backend fixture tests
//===----------------------------------------------------------------------===//
// These drive the sysfs backend against machines described by single-file
// snapshot manifests under testdata/sysfs/<name>.sysfs.txt.

#if defined(IREE_TASK_TOPOLOGY_TEST_HAVE_SYSFS_MOCK)

// Directory holding the snapshot trees.
IREE_FLAG(string, task_topology_test_snapshots, "",
          "Directory containing expanded sysfs snapshot trees. Set by the "
          "build; tests requiring a snapshot skip when it is empty.");

// Returns the directory for |name| within the snapshot directory.
static std::string SnapshotPath(const std::string& name) {
  const char* root = FLAG_task_topology_test_snapshots;
  if (!root || !*root) {
    fprintf(stderr, "FATAL: --task_topology_test_snapshots is not set.)\n");
    abort();
  }
  return std::string(root) + "/" + name;
}

// Set the snapshot root to the test snapshot for the duration of the test.
//
// Sets unimplemented, if not supported.
//
// The root is process-global: test cases must not run concurrently.
class SysfsSnapshot {
 public:
  explicit SysfsSnapshot(const std::string& name) : root_(SnapshotPath(name)) {
    if (access((root_ + "/cpu/present").c_str(), R_OK) != 0) {
      message_ = root_ + " is not a usable snapshot: broken data dependency.";
      return;
    }
    iree_status_t status = iree_task_topology_set_snapshot_path(root_.c_str());
    if (iree_status_is_unimplemented(status)) {
      unsupported_ = true;
      message_ = "topology backend does not read snapshots.";
    } else if (!iree_status_is_ok(status)) {
      message_ = root_ + " could not be installed as the topology source.";
    } else {
      ok_ = true;
    }
    iree_status_ignore(status);
  }
  ~SysfsSnapshot() {
    iree_status_ignore(iree_task_topology_set_snapshot_path(nullptr));
  }
  SysfsSnapshot(const SysfsSnapshot&) = delete;
  SysfsSnapshot& operator=(const SysfsSnapshot&) = delete;

  // True once the snapshot is the active topology source.
  bool ok() const { return ok_; }
  // True if the compiled-in backend reads no snapshots.
  bool unsupported() const { return unsupported_; }
  // Why the snapshot is unusable; empty when ok().
  const std::string& message() const { return message_; }

 private:
  std::string root_;
  std::string message_;
  bool ok_ = false;
  bool unsupported_ = false;
};

static iree_host_size_t PopcountMask(iree_task_topology_group_mask_t mask) {
  return iree_task_affinity_set_count_ones(mask);
}

// Regression for the hybrid "node = L2 cluster": on a single-package
// hybrid CPU the whole machine must be one NUMA node and node 0 must yield
// every physical core, not a single L2 module.
//
// synthetic_meteor_lake: 6 HT P-cores + 8 E-cores
// in two L2 modules + 2 LP-E cores with no L3, sparse cluster ids).
TEST(TopologySysfsTest, HybridSinglePackageUsesAllCores) {
  SysfsSnapshot snapshot("synthetic_meteor_lake");
  if (snapshot.unsupported()) GTEST_SKIP() << snapshot.message();
  ASSERT_TRUE(snapshot.ok()) << snapshot.message();

  // One package -> one NUMA node.
  EXPECT_EQ(1u, iree_task_topology_query_node_count());

  // Selecting node 0 must return all 16 physical cores (6 P + 8 E + 2 LP-E),
  // not the 2-4 cores of one L2 module.
  iree_task_topology_t topology;
  iree_task_topology_initialize(&topology);
  IREE_ASSERT_OK(iree_task_topology_initialize_from_physical_cores(
      /*node_id=*/0, IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_ANY,
      IREE_TASK_TOPOLOGY_DISTRIBUTION_SCATTER, /*max_core_count=*/64,
      &topology));
  EXPECT_EQ(16u, iree_task_topology_group_count(&topology));

  // No node hierarchy, so there is no kernel node id to bind memory to.
  EXPECT_EQ(IREE_TASK_TOPOLOGY_NODE_ID_ANY, topology.numa_node_id);

  // Every group carries a valid NUMA node hint of 0 (never the truncated 255).
  int lp_e_groups = 0;
  bool saw_p_core = false;
  for (iree_host_size_t i = 0; i < iree_task_topology_group_count(&topology);
       ++i) {
    const iree_task_topology_group_t* g =
        iree_task_topology_get_group(&topology, i);
    EXPECT_EQ(0u, g->ideal_thread_affinity.group);
    EXPECT_EQ(1u, g->ideal_thread_affinity.id_assigned);
    if (g->caches.l1_data == 48u * 1024u) saw_p_core = true;
    // LP-E cores (processors 20/21) have no L3 and thus form their own 2-core
    // last-level-cache sharing group; P/E cores share the 24M L3 across 14
    // cores.
    if (g->processor_index == 20 || g->processor_index == 21) {
      ++lp_e_groups;
      EXPECT_EQ(iree_host_size_t{2},
                PopcountMask(g->constructive_sharing_mask));
    } else {
      EXPECT_EQ(iree_host_size_t{14},
                PopcountMask(g->constructive_sharing_mask));
    }
  }
  EXPECT_TRUE(saw_p_core);
  EXPECT_EQ(2, lp_e_groups);

  iree_task_topology_deinitialize(&topology);
}

// Multi-socket NUMA: one node per socket, and node filtering restricts workers
// to the requested one.
//
// synthetic_dual_socket: two sockets of 4 single-threaded cores, one NUMA node
// each; per-socket core_ids.
TEST(TopologySysfsTest, MultiSocketNodePartitioning) {
  SysfsSnapshot snapshot("synthetic_dual_socket");
  if (snapshot.unsupported()) GTEST_SKIP() << snapshot.message();
  ASSERT_TRUE(snapshot.ok()) << snapshot.message();

  EXPECT_EQ(2u, iree_task_topology_query_node_count());

  for (uint32_t node = 0; node < 2; ++node) {
    iree_task_topology_t topology;
    iree_task_topology_initialize(&topology);
    IREE_ASSERT_OK(iree_task_topology_initialize_from_physical_cores(
        node, IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_ANY,
        IREE_TASK_TOPOLOGY_DISTRIBUTION_COMPACT, 64, &topology));
    EXPECT_EQ(4u, iree_task_topology_group_count(&topology));
    for (iree_host_size_t i = 0; i < iree_task_topology_group_count(&topology);
         ++i) {
      const iree_task_topology_group_t* g =
          iree_task_topology_get_group(&topology, i);
      const uint32_t lo = node * 4, hi = lo + 4;
      EXPECT_GE(g->processor_index, lo);
      EXPECT_LT(g->processor_index, hi);
      EXPECT_EQ(node, g->ideal_thread_affinity.group);
    }
    // Every core came from the requested node, so that is the node to bind this
    // executor's memory to.
    EXPECT_EQ(node, topology.numa_node_id);
    iree_task_topology_deinitialize(&topology);
  }

  // Without a node filter we get every core across both sockets.
  iree_task_topology_t all;
  iree_task_topology_initialize(&all);
  IREE_ASSERT_OK(iree_task_topology_initialize_from_physical_cores(
      IREE_TASK_TOPOLOGY_NODE_ID_ANY, IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_ANY,
      IREE_TASK_TOPOLOGY_DISTRIBUTION_COMPACT, 64, &all));
  EXPECT_EQ(8u, iree_task_topology_group_count(&all));
  iree_task_topology_deinitialize(&all);
}

// Regression: with NODE_ID_ANY the core scan spans both sockets at once, so
// deduplicating physical cores on the raw topology/core_id -- which restarts at
// 0 on each socket -- collapses socket 1 onto socket 0 and silently discards
// half the machine. Both identity sources must survive that.
// synthetic_dual_socket_no_siblings: two sockets of 4 single-threaded cores;
// per-socket core_ids.
TEST(TopologySysfsTest, MultiSocketAnyNodeKeepsBothSockets) {
  // The two manifests differ only in whether topology/core_cpus_list exists,
  // selecting which of the two physical-core identity sources the backend uses
  // (the sibling-list key vs the package+core_id fallback).
  for (const char* snapshot_name :
       {"synthetic_dual_socket", "synthetic_dual_socket_no_siblings"}) {
    SCOPED_TRACE(snapshot_name);
    SysfsSnapshot snapshot(snapshot_name);
    if (snapshot.unsupported()) GTEST_SKIP() << snapshot.message();
    ASSERT_TRUE(snapshot.ok()) << snapshot.message();

    iree_task_topology_t topology;
    iree_task_topology_initialize(&topology);
    IREE_ASSERT_OK(iree_task_topology_initialize_from_physical_cores(
        IREE_TASK_TOPOLOGY_NODE_ID_ANY,
        IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_ANY,
        IREE_TASK_TOPOLOGY_DISTRIBUTION_COMPACT, 64, &topology));

    // All 8 distinct physical cores, one group each.
    ASSERT_EQ(8u, iree_task_topology_group_count(&topology));
    bool seen_processor[8] = {false};
    int socket1_groups = 0;
    for (iree_host_size_t i = 0; i < iree_task_topology_group_count(&topology);
         ++i) {
      const iree_task_topology_group_t* g =
          iree_task_topology_get_group(&topology, i);
      ASSERT_LT(g->processor_index, 8u);
      EXPECT_FALSE(seen_processor[g->processor_index]);
      seen_processor[g->processor_index] = true;
      if (g->processor_index >= 4) {
        ++socket1_groups;
        EXPECT_EQ(1u, g->ideal_thread_affinity.group);  // socket 1 NUMA hint
      }
    }
    EXPECT_EQ(4, socket1_groups) << "socket 1 cores were dropped";

    // The selected cores span both nodes, so no single node can back this
    // executor's allocations.
    EXPECT_EQ(IREE_TASK_TOPOLOGY_NODE_ID_ANY, topology.numa_node_id);

    iree_task_topology_deinitialize(&topology);
  }
}

// Sub-NUMA clustering (Intel SNC / AMD NPS): one physical package split into
// multiple NUMA nodes. The backend must honor the /sys/devices/system/node/
// tables and report TWO nodes, partitioning cores by NUMA node -- something
// physical_package_id alone (all 0 here) could not do.
// synthetic_snc: one package of 8 cores split into two NUMA nodes (0-3, 4-7).
TEST(TopologySysfsTest, SubNumaClusterSplitsSinglePackage) {
  SysfsSnapshot snapshot("synthetic_snc");
  if (snapshot.unsupported()) GTEST_SKIP() << snapshot.message();
  ASSERT_TRUE(snapshot.ok()) << snapshot.message();

  // True NUMA sees two nodes even though there is a single physical package.
  EXPECT_EQ(2u, iree_task_topology_query_node_count());

  for (uint32_t node = 0; node < 2; ++node) {
    iree_task_topology_t topology;
    iree_task_topology_initialize(&topology);
    IREE_ASSERT_OK(iree_task_topology_initialize_from_physical_cores(
        node, IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_ANY,
        IREE_TASK_TOPOLOGY_DISTRIBUTION_COMPACT, 64, &topology));
    EXPECT_EQ(4u, iree_task_topology_group_count(&topology));
    for (iree_host_size_t i = 0; i < iree_task_topology_group_count(&topology);
         ++i) {
      const iree_task_topology_group_t* g =
          iree_task_topology_get_group(&topology, i);
      const uint32_t lo = node * 4, hi = lo + 4;
      EXPECT_GE(g->processor_index, lo);
      EXPECT_LT(g->processor_index, hi);
      EXPECT_EQ(node, g->ideal_thread_affinity.group);
    }
    // Every core came from the requested node, so that is the node to bind this
    // executor's memory to.
    EXPECT_EQ(node, topology.numa_node_id);
    iree_task_topology_deinitialize(&topology);
  }
}

// When physical_package_id is the -1 sentinel (UINT16_MAX), the affinity group
// hint must default to 0, never the truncated 255 from the 8-bit field, and the
// machine collapses to a single node.
// synthetic_invalid_package: 4 cores whose physical_package_id is the 65535
// sentinel, no node hierarchy.
TEST(TopologySysfsTest, InvalidNodeIdDefaultsToNodeZero) {
  SysfsSnapshot snapshot("synthetic_invalid_package");
  if (snapshot.unsupported()) GTEST_SKIP() << snapshot.message();
  ASSERT_TRUE(snapshot.ok()) << snapshot.message();

  EXPECT_EQ(1u, iree_task_topology_query_node_count());

  iree_task_topology_t topology;
  iree_task_topology_initialize(&topology);
  IREE_ASSERT_OK(iree_task_topology_initialize_from_physical_cores(
      IREE_TASK_TOPOLOGY_NODE_ID_ANY, IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_ANY,
      IREE_TASK_TOPOLOGY_DISTRIBUTION_COMPACT, 64, &topology));
  EXPECT_EQ(4u, iree_task_topology_group_count(&topology));
  for (iree_host_size_t i = 0; i < iree_task_topology_group_count(&topology);
       ++i) {
    EXPECT_EQ(0u, iree_task_topology_get_group(&topology, i)
                      ->ideal_thread_affinity.group);
  }
  iree_task_topology_deinitialize(&topology);
}

// big.LITTLE performance-level filtering keys off cpu_capacity (independent of
// the node change) and must still select big/little subsets on a single node.
// synthetic_big_little: one package of 4 cores, 2 big (capacity 1024) and 2
// little (capacity 256).
TEST(TopologySysfsTest, HeterogeneousPerformanceLevelFilter) {
  SysfsSnapshot snapshot("synthetic_big_little");
  if (snapshot.unsupported()) GTEST_SKIP() << snapshot.message();
  ASSERT_TRUE(snapshot.ok()) << snapshot.message();

  struct {
    iree_task_topology_performance_level_t level;
    iree_host_size_t expected;
  } cases[] = {
      {IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_ANY, 4},
      {IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_HIGH, 2},
      {IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_LOW, 2},
  };
  for (const auto& c : cases) {
    iree_task_topology_t topology;
    iree_task_topology_initialize(&topology);
    IREE_ASSERT_OK(iree_task_topology_initialize_from_physical_cores(
        IREE_TASK_TOPOLOGY_NODE_ID_ANY, c.level,
        IREE_TASK_TOPOLOGY_DISTRIBUTION_COMPACT, 64, &topology));
    EXPECT_EQ(c.expected, iree_task_topology_group_count(&topology))
        << "performance level " << static_cast<int>(c.level);
    iree_task_topology_deinitialize(&topology);
  }
}

// 8 HT P-cores each with its own cluster id + 8 E-cores in two
// modules; cluster ids {0,8,...,56,64,72}. One NUMA node, 16 physical cores.
TEST(TopologySysfsTest, RaptorLake) {
  SysfsSnapshot snapshot("synthetic_raptor_lake");
  if (snapshot.unsupported()) GTEST_SKIP() << snapshot.message();
  ASSERT_TRUE(snapshot.ok()) << snapshot.message();

  // One node from /sys/devices/system/node, not the ten cluster ids.
  EXPECT_EQ(1u, iree_task_topology_query_node_count());

  // 'current' is the default and 'all' the documented alternative; both must
  // select every physical core on this machine.
  for (const char* spec : {"current", "all"}) {
    SCOPED_TRACE(spec);
    iree_task_node_selection_t selection;
    IREE_ASSERT_OK(iree_task_topology_select_nodes(iree_make_cstring_view(spec),
                                                   &selection));
    ASSERT_EQ(1u, selection.count);

    iree_task_topology_t topology;
    iree_task_topology_initialize(&topology);
    IREE_ASSERT_OK(iree_task_topology_initialize_from_physical_cores(
        selection.ids[0], IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_ANY,
        IREE_TASK_TOPOLOGY_DISTRIBUTION_COMPACT, /*max_core_count=*/64,
        &topology));
    EXPECT_EQ(16u, iree_task_topology_group_count(&topology))
        << "--task_topology_nodes=" << spec << " must run on all 16 cores";
    // All 16 cores sit on the one node, so even 'all' resolves to it.
    EXPECT_EQ(0u, topology.numa_node_id);
    iree_task_topology_deinitialize(&topology);
  }

  // Node 0 yields all 16 physical cores: 8 P with HT collapsed + 8 E.
  iree_task_topology_t topology;
  iree_task_topology_initialize(&topology);
  IREE_ASSERT_OK(iree_task_topology_initialize_from_physical_cores(
      /*node_id=*/0, IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_ANY,
      IREE_TASK_TOPOLOGY_DISTRIBUTION_COMPACT, 64, &topology));
  ASSERT_EQ(16u, iree_task_topology_group_count(&topology));
  int p_cores = 0;
  int e_cores = 0;
  for (iree_host_size_t i = 0; i < iree_task_topology_group_count(&topology);
       ++i) {
    const iree_task_topology_group_t* g =
        iree_task_topology_get_group(&topology, i);
    EXPECT_EQ(0u, g->ideal_thread_affinity.group);
    // Everything shares the 30M L3, so all 16 groups form one LLC domain.
    EXPECT_EQ(iree_host_size_t{16}, PopcountMask(g->constructive_sharing_mask));
    if (g->caches.l1_data == 48u * 1024u) {
      ++p_cores;
      EXPECT_EQ(2048u * 1024u, g->caches.l2_data);
    } else if (g->caches.l1_data == 32u * 1024u) {
      ++e_cores;
      EXPECT_EQ(4096u * 1024u, g->caches.l2_data);
    }
  }
  EXPECT_EQ(8, p_cores);
  EXPECT_EQ(8, e_cores);
  iree_task_topology_deinitialize(&topology);
}

// Sparse node ids must pass through as the kernel's ids rather than as indices
// into the enumerated set. Node/online reads "0,4" here: remapping to a dense
// [0,N) space would make "1" mean node 4, leave node 4 unreachable by its real
// name, and put a nonexistent node in the affinity group hint fed to
// set_mempolicy. See synthetic_sparse_node_ids.sysfs.txt.
TEST(TopologySysfsTest, SparseNodeIdsKeepKernelIds) {
  SysfsSnapshot snapshot("synthetic_sparse_node_ids");
  if (snapshot.unsupported()) GTEST_SKIP() << snapshot.message();
  ASSERT_TRUE(snapshot.ok()) << snapshot.message();

  // Two nodes, enumerated by their kernel ids, not 0 and 1.
  iree_task_topology_node_id_t ids[8] = {0};
  const iree_host_size_t count =
      iree_task_topology_query_node_ids(IREE_ARRAYSIZE(ids), ids);
  ASSERT_EQ(2u, count);
  EXPECT_EQ(0u, ids[0]);
  EXPECT_EQ(4u, ids[1]);

  // Selecting the sparse id yields that node's cores, and the affinity group
  // hint carries the kernel id so memory policy targets a node that exists.
  iree_task_topology_t topology;
  iree_task_topology_initialize(&topology);
  IREE_ASSERT_OK(iree_task_topology_initialize_from_physical_cores(
      /*node_id=*/4, IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_ANY,
      IREE_TASK_TOPOLOGY_DISTRIBUTION_COMPACT, /*max_core_count=*/64,
      &topology));
  ASSERT_EQ(4u, iree_task_topology_group_count(&topology));
  for (iree_host_size_t i = 0; i < iree_task_topology_group_count(&topology);
       ++i) {
    const iree_task_topology_group_t* g =
        iree_task_topology_get_group(&topology, i);
    EXPECT_GE(g->processor_index, 4u);
    EXPECT_LT(g->processor_index, 8u);
    EXPECT_EQ(4u, g->ideal_thread_affinity.group);
  }
  iree_task_topology_deinitialize(&topology);

  // An id that only exists in a dense remapping must be rejected.
  iree_task_node_selection_t selection;
  iree_status_t status =
      iree_task_topology_select_nodes(IREE_SV("1"), &selection);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);

  // The kernel ids are accepted.
  IREE_EXPECT_OK(iree_task_topology_select_nodes(IREE_SV("0,4"), &selection));
  EXPECT_EQ(2u, selection.count);
  EXPECT_EQ(4u, selection.ids[1]);
}

// A memory-only NUMA node (readable but EMPTY cpulist, e.g. CXL/Optane/SNC)
// must not become a node with zero worker groups: node ids are derived from the
// CPUs that exist, so a CPU-less node is simply never selected.
// synthetic_memory_only_node: 4 cores on node 0; node 1 is online with an empty
// cpulist (no CPUs attached).
TEST(TopologySysfsTest, MemoryOnlyNumaNodeIgnored) {
  // 4 cores on node 0; node 1 is online with an empty cpulist (no CPUs
  // attached): see synthetic_memory_only_node.sysfs.txt.
  SysfsSnapshot snapshot("synthetic_memory_only_node");
  if (snapshot.unsupported()) GTEST_SKIP() << snapshot.message();
  ASSERT_TRUE(snapshot.ok()) << snapshot.message();

  // Only the CPU-bearing node is reported.
  iree_task_topology_node_id_t ids[8] = {0};
  const iree_host_size_t n = iree_task_topology_query_node_ids(8, ids);
  EXPECT_EQ(1u, n);
  EXPECT_EQ(0u, ids[0]);

  // Asking for the memory-only node explicitly yields an empty topology that
  // callers skip, rather than an executor with no workers.
  iree_task_topology_t t;
  iree_task_topology_initialize(&t);
  IREE_ASSERT_OK(iree_task_topology_initialize_from_physical_cores(
      /*node_id=*/1, IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_ANY,
      IREE_TASK_TOPOLOGY_DISTRIBUTION_COMPACT, 64, &t));
  EXPECT_EQ(0u, iree_task_topology_group_count(&t));
  iree_task_topology_deinitialize(&t);
}

// A long, fragmented node cpulist (comma-separated singles, as seen on some
// POWER/AMD/SNC enumerations) must be read in full. A buffer too small to hold
// one trips the out-of-range guard on the node read and aborts rather than
// silently reporting the wrong nodes, so this fixture is what keeps that buffer
// honest about real-world list lengths.
// synthetic_fragmented_cpulist: 200 CPUs, node0/cpulist listing each
// individually (~690 chars), package id 7 everywhere.
TEST(TopologySysfsTest, LargeFragmentedNodeCpulist) {
  // 200 CPUs, node0/cpulist listing each individually (~690 chars), package
  // id 7 everywhere: see synthetic_fragmented_cpulist.sysfs.txt.
  SysfsSnapshot snapshot("synthetic_fragmented_cpulist");
  if (snapshot.unsupported()) GTEST_SKIP() << snapshot.message();
  ASSERT_TRUE(snapshot.ok()) << snapshot.message();

  EXPECT_EQ(1u, iree_task_topology_query_node_count());
  // Node identity comes from node/, so it is 0; the fixture sets
  // physical_package_id to 7 everywhere so confusing the two would be visible.
  EXPECT_EQ(0u, iree_task_topology_query_current_node());
  iree_task_topology_node_id_t ids[8] = {0};
  iree_task_topology_query_node_ids(8, ids);
  EXPECT_EQ(0u, ids[0]);
}

//===----------------------------------------------------------------------===//
// Captured-hardware snapshot tests
//===----------------------------------------------------------------------===//
// Pixel 6 (Google Tensor GS101): 2x Cortex-X1 (capacity 1024) + 2x A76 (820) +
// 4x A55 (280), single package, no /sys/devices/system/node hierarchy, three
// L2 cluster ids (0/1/2).
TEST(TopologySysfsSnapshotTest, Pixel6Tensor) {
  SysfsSnapshot snapshot("arm64_pixel6_tensor");
  if (snapshot.unsupported()) GTEST_SKIP() << snapshot.message();
  ASSERT_TRUE(snapshot.ok()) << snapshot.message();

  // No node hierarchy at all, so ONE node, despite three L2 cluster ids. Under
  // the old cluster_id-as-node scheme this reported 3 and 'current' pinned all
  // work to one 2-core cluster.
  EXPECT_EQ(1u, iree_task_topology_query_node_count());

  // Node 0 is the only id query_node_ids reports, so it is selectable, but no
  // core has a kernel-confirmed node: there is nothing to bind memory to.
  {
    iree_task_topology_t named;
    iree_task_topology_initialize(&named);
    IREE_ASSERT_OK(iree_task_topology_initialize_from_physical_cores(
        /*node_id=*/0, IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_ANY,
        IREE_TASK_TOPOLOGY_DISTRIBUTION_COMPACT, 64, &named));
    EXPECT_EQ(8u, iree_task_topology_group_count(&named));
    EXPECT_EQ(IREE_TASK_TOPOLOGY_NODE_ID_ANY, named.numa_node_id);
    iree_task_topology_deinitialize(&named);
  }

  // All 8 physical cores, each with a node-0 affinity hint and per-cluster L2
  // sizes as captured (X1 512K / A76 256K / A55 128K); LLC (4M L3) sharing
  // groups follow the per-cluster shared_cpu_list (0-1, 2-3, 4-7).
  iree_task_topology_t topology;
  iree_task_topology_initialize(&topology);
  IREE_ASSERT_OK(iree_task_topology_initialize_from_physical_cores(
      IREE_TASK_TOPOLOGY_NODE_ID_ANY, IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_ANY,
      IREE_TASK_TOPOLOGY_DISTRIBUTION_COMPACT, 64, &topology));
  ASSERT_EQ(8u, iree_task_topology_group_count(&topology));
  for (iree_host_size_t i = 0; i < iree_task_topology_group_count(&topology);
       ++i) {
    const iree_task_topology_group_t* g =
        iree_task_topology_get_group(&topology, i);
    EXPECT_EQ(0u, g->ideal_thread_affinity.group);
    EXPECT_EQ(1u, g->ideal_thread_affinity.id_assigned);
    const uint32_t cpu = g->processor_index;
    const uint32_t expected_l2 =
        cpu < 2 ? 512u * 1024u : (cpu < 4 ? 256u * 1024u : 128u * 1024u);
    EXPECT_EQ(expected_l2, g->caches.l2_data) << "cpu " << cpu;
    const iree_host_size_t expected_llc_peers = cpu < 4 ? 2 : 4;
    EXPECT_EQ(expected_llc_peers, PopcountMask(g->constructive_sharing_mask))
        << "cpu " << cpu;
  }
  iree_task_topology_deinitialize(&topology);

  // big.LITTLE selection via cpu_capacity: threshold is 75% of max (768), so
  // HIGH = X1+A76 (0-3) and LOW = A55 (4-7).
  struct {
    iree_task_topology_performance_level_t level;
    uint32_t min_cpu;
    uint32_t max_cpu;
  } cases[] = {
      {IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_HIGH, 0, 3},
      {IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_LOW, 4, 7},
  };
  for (const auto& c : cases) {
    iree_task_topology_t filtered;
    iree_task_topology_initialize(&filtered);
    IREE_ASSERT_OK(iree_task_topology_initialize_from_physical_cores(
        IREE_TASK_TOPOLOGY_NODE_ID_ANY, c.level,
        IREE_TASK_TOPOLOGY_DISTRIBUTION_COMPACT, 64, &filtered));
    EXPECT_EQ(4u, iree_task_topology_group_count(&filtered));
    for (iree_host_size_t i = 0; i < iree_task_topology_group_count(&filtered);
         ++i) {
      const uint32_t cpu =
          iree_task_topology_get_group(&filtered, i)->processor_index;
      EXPECT_GE(cpu, c.min_cpu);
      EXPECT_LE(cpu, c.max_cpu);
    }
    iree_task_topology_deinitialize(&filtered);
  }
}

// AMD Ryzen 9 7950X (captured from real hardware): 16 cores / 32 SMT threads
// where sibling pairs are NON-adjacent (cpuN pairs with cpuN+16), two 8-core
// CCDs each with a private 32M L3 (0-7,16-23 and 8-15,24-31), one NUMA node,
// and cluster_id reporting the 65535 sentinel on every cpu. See
// x86_64_ryzen9_7950x.sysfs.txt.
TEST(TopologySysfsSnapshotTest, Ryzen9_7950X) {
  SysfsSnapshot snapshot("x86_64_ryzen9_7950x");
  if (snapshot.unsupported()) GTEST_SKIP() << snapshot.message();
  ASSERT_TRUE(snapshot.ok()) << snapshot.message();

  // One NUMA node; the all-sentinel cluster ids collapse to the single
  // fallback node rather than leaking 65535 as a cluster.
  EXPECT_EQ(1u, iree_task_topology_query_node_count());

  iree_task_topology_t topology;
  iree_task_topology_initialize(&topology);
  IREE_ASSERT_OK(iree_task_topology_initialize_from_physical_cores(
      IREE_TASK_TOPOLOGY_NODE_ID_ANY, IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_ANY,
      IREE_TASK_TOPOLOGY_DISTRIBUTION_COMPACT, 64, &topology));

  // 16 physical cores: the non-adjacent (cpuN, cpuN+16) sibling pairs must
  // collapse via core_cpus_list, keeping the first sibling of each pair.
  ASSERT_EQ(16u, iree_task_topology_group_count(&topology));
  for (iree_host_size_t i = 0; i < iree_task_topology_group_count(&topology);
       ++i) {
    const iree_task_topology_group_t* g =
        iree_task_topology_get_group(&topology, i);
    EXPECT_LT(g->processor_index, 16u);
    // NUMA node hint 0 -- never a truncated 255 from the 65535 cluster/package
    // sentinels.
    EXPECT_EQ(0u, g->ideal_thread_affinity.group);
    // Real caches: 32K L1d + 1M L2 per core; each 32M L3 is private to an
    // 8-core CCD, so every group shares its LLC with exactly 8 of the 16
    // groups.
    EXPECT_EQ(32u * 1024u, g->caches.l1_data);
    EXPECT_EQ(1024u * 1024u, g->caches.l2_data);
    EXPECT_EQ(32768u * 1024u, g->caches.l3_data);
    EXPECT_EQ(iree_host_size_t{8}, PopcountMask(g->constructive_sharing_mask));
  }
  iree_task_topology_deinitialize(&topology);
}

//===----------------------------------------------------------------------===//
// --task_topology_nodes grammar
//===----------------------------------------------------------------------===//
// iree_task_topology_select_nodes resolves the flag spec against the machine.
// Driving it with an explicit spec (rather than the global flag) against a mock
// tree keeps the grammar, sparse-id enumeration, and validation hermetic.

// Convenience: resolve |spec| and expect success.
static iree_task_node_selection_t SelectNodes(const char* spec) {
  iree_task_node_selection_t selection;
  IREE_EXPECT_OK(iree_task_topology_select_nodes(iree_make_cstring_view(spec),
                                                 &selection));
  return selection;
}

// Spec-to-id mapping on a machine with a single node; the fixture's core layout
// is irrelevant here, only its node count.
TEST(TopologySelectNodesTest, SingleNodeMachine) {
  SysfsSnapshot snapshot("synthetic_meteor_lake");
  if (snapshot.unsupported()) GTEST_SKIP() << snapshot.message();
  ASSERT_TRUE(snapshot.ok()) << snapshot.message();

  // 'current'/'' resolve to exactly one NUMA node.
  for (const char* spec : {"current", ""}) {
    SCOPED_TRACE(spec);
    const iree_task_node_selection_t s = SelectNodes(spec);
    EXPECT_EQ(1u, s.count);
    EXPECT_NE(IREE_TASK_TOPOLOGY_NODE_ID_ANY, s.ids[0]);
  }

  // 'all' is a single node-agnostic executor over every core.
  {
    const iree_task_node_selection_t s = SelectNodes("all");
    EXPECT_EQ(1u, s.count);
    EXPECT_EQ(IREE_TASK_TOPOLOGY_NODE_ID_ANY, s.ids[0]);
  }

  // 'numa' builds one executor per NUMA node.
  {
    const iree_task_node_selection_t s = SelectNodes("numa");
    EXPECT_EQ(1u, s.count);  // one package -> one node
  }
}
// A machine with more nodes than a selection can name must be rejected, not
// silently reduced to the first IREE_TASK_TOPOLOGY_MAX_NODES of them (which
// would leave the rest of the machine idle with no diagnostic).
TEST(TopologySelectNodesTest, TooManyNodesRejected) {
  SysfsSnapshot snapshot("synthetic_too_many_nodes");
  if (snapshot.unsupported()) GTEST_SKIP() << snapshot.message();
  ASSERT_TRUE(snapshot.ok()) << snapshot.message();

  // The backend reports the true node count even though it can only store
  // IREE_TASK_TOPOLOGY_MAX_NODES of the ids; that is what makes the overflow
  // detectable rather than invisible.
  EXPECT_EQ(65u, iree_task_topology_query_node_count());

  iree_task_node_selection_t selection;
  iree_status_t status =
      iree_task_topology_select_nodes(IREE_SV("numa"), &selection);
  EXPECT_TRUE(iree_status_is_resource_exhausted(status));
  iree_status_ignore(status);

  // Explicit ids cannot be validated against a node set we cannot enumerate,
  // so those are rejected too rather than mis-reporting a valid id as absent.
  status = iree_task_topology_select_nodes(IREE_SV("0"), &selection);
  EXPECT_TRUE(iree_status_is_resource_exhausted(status));
  iree_status_ignore(status);

  // 'current' and 'all' name a single node and stay usable.
  IREE_EXPECT_OK(iree_task_topology_select_nodes(IREE_SV("all"), &selection));
  EXPECT_EQ(1u, selection.count);
}

// Explicit ids are validated against the machine's enumerated node ids.
TEST(TopologySelectNodesTest, ExplicitIdsValidatedAgainstMachine) {
  // Two NUMA nodes (one per socket): ids 0 and 1 exist, nothing else does.
  SysfsSnapshot snapshot("synthetic_dual_socket");
  if (snapshot.unsupported()) GTEST_SKIP() << snapshot.message();
  ASSERT_TRUE(snapshot.ok()) << snapshot.message();

  {
    const iree_task_node_selection_t s = SelectNodes("0,1");
    EXPECT_EQ(2u, s.count);
    EXPECT_EQ(0u, s.ids[0]);
    EXPECT_EQ(1u, s.ids[1]);
  }

  // An id the machine does not have must be rejected up front: left to run, it
  // yields an empty topology and thus an executor-less driver far from here.
  iree_task_node_selection_t selection;
  iree_status_t status =
      iree_task_topology_select_nodes(IREE_SV("7"), &selection);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);

  // Non-numeric values are rejected rather than silently treated as node 0.
  status = iree_task_topology_select_nodes(IREE_SV("numa,0"), &selection);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

#endif  // IREE_TASK_TOPOLOGY_TEST_HAVE_SYSFS_MOCK

}  // namespace
