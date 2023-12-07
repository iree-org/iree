// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_STREAM_ANALYSIS_PARTITIONING_H_
#define IREE_COMPILER_DIALECT_STREAM_ANALYSIS_PARTITIONING_H_

#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler::IREE::Stream {

//===----------------------------------------------------------------------===//
// Data structures
//===----------------------------------------------------------------------===//

// A single slice of ops.
struct Partition {
  // Affinity compatible with all ops in the partition.
  IREE::Stream::AffinityAttr affinity;
  // SSA values defined outside of the partition.
  // All values not defined by ops in the partition must be declared.
  // Multiple partitions may capture the same value.
  SetVector<Value> ins;
  // SSA values defined by the partition with uses outside.
  // All values used by ops outside of the partition must be declared.
  // Only one partition may produce a new value.
  SetVector<Value> outs;
  // All ops covered by the partition. May contain ops that exist in other
  // partitions in cases where the op is to be duplicated. Not all ops are
  // streamable (such as constants and arithmetic).
  SetVector<Operation *> ops;

  void dump(AsmState &asmState);

  // Verifies that the partition meets the required conditions.
  LogicalResult verify(Location loc);
};

// A set of all partitions.
struct PartitionSet {
  // All partitions in an undefined topological order.
  SmallVector<Partition> partitions;

  // Total number of partitions in the set.
  size_t size() const { return partitions.size(); }
  // Returns true if the set is empty (no streamable ops).
  bool empty() const { return partitions.empty(); }

  void dump(AsmState &asmState);

  // Verifies that the partition set meets the required conditions.
  LogicalResult verify(Location loc);

  // Sorts all partitions in a topological order.
  void topologicalSort();
};

//===----------------------------------------------------------------------===//
// Stream partitioning algorithms
//===----------------------------------------------------------------------===//
//
// When these algorithms run all streamable operations have had an affinity
// assigned and are lowered out of tensor form. Some resources may have
// lifetimes associated but most will remain unassigned (`!stream.resource<*>`)
// until after partitioning. Note that there may already exist partitioned ops
// in stream.execute regions already.
//
// The intent is that we can use the information we have about each operation,
// the resources moving between them, and where they should execute to better
// partition the DAG. This could optimize for reducing memory transfer between
// devices, reducing latency by minimizing cuts, maximizing concurrency by
// separating non-interfering subgraphs, etc.
//
// This is a well-researched area and there are many algorithms to choose from.
// We'll mostly want to focus on ones that are able to handle multiple critera
// (like memory consumption, compute utilization, available capacity, etc).
//
// See for example:
//   dagP: https://github.com/GT-TDAlab/dagP
//     Multilevel Algorithms for Acyclic Partitioning of Directed Acyclic Graphs
//     https://hal.inria.fr/hal-02306566/document
//  METIS: https://github.com/KarypisLab/METIS
//     A Fast and High Quality Multilevel Scheme for Partitioning Ireegular
//     Graphs
//     http://glaros.dtc.umn.edu/gkhome/metis/metis/publications
// SCOTCH: https://www.labri.fr/perso/pelegrin/scotch/
//     Contributions to Parallel Multilevel Graph Partitioning
//     https://www.labri.fr/perso/pelegrin/papers/hdr.pdf
// Zoltan: https://cs.sandia.gov/Zoltan/
//     https://cs.sandia.gov/Zoltan/Zoltan_pubs.html
//     https://cs.sandia.gov/Zoltan/papers/zoltan_tutorial_dagstuhl09.pdf
//
// And some good papers/overviews:
// - Edge Partitioning of Large Graphs
//   https://tel.archives-ouvertes.fr/tel-01956979/document
//

// Partitions the ops in |block| such that all streamable ops are in one or more
// partitions (with >1 implying duplication). Partitions may contain
// non-streamable ops if it is safe to do so (such as std arithmetic). Not all
// ops in the block will be covered by a partition.
PartitionSet partitionStreamableOps(IREE::Stream::PartitioningConfigAttr config,
                                    Block *block);
PartitionSet
partitionRegionConcurrency(IREE::Stream::PartitioningConfigAttr config,
                           Block *block);

//===----------------------------------------------------------------------===//
// Reference partitioning
//===----------------------------------------------------------------------===//

// Naive clustering based solely on correctness with no cost model or weighting.
// Produces the largest possible streams for any given block. Unsatisfactory.
PartitionSet
partitionStreamableOpsReference(IREE::Stream::PartitioningConfigAttr config,
                                Block *block);

// Similarly poor algorithm to partitionStreamableOpsReference but for use
// within partitioned streams to produce waves of concurrently executable work.
PartitionSet
partitionRegionConcurrencyReference(IREE::Stream::PartitioningConfigAttr config,
                                    Block *block);

} // namespace mlir::iree_compiler::IREE::Stream

#endif // IREE_COMPILER_DIALECT_STREAM_ANALYSIS_PARTITIONING_H_
