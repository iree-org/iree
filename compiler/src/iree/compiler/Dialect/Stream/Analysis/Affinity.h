// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_STREAM_ANALYSIS_AFFINITY_H_
#define IREE_COMPILER_DIALECT_STREAM_ANALYSIS_AFFINITY_H_

#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Solver.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"

namespace mlir::iree_compiler::IREE::Stream {

//===----------------------------------------------------------------------===//
// Affinity analysis
//===----------------------------------------------------------------------===//

// Performs whole-program analysis of resource and tensor value affinity.
// All `!stream.resource` and `tensor` SSA values will be analyzed and their
// affinities where used will be available for querying via the lookup
// functions.
class AffinityAnalysis {
public:
  explicit AffinityAnalysis(Operation *rootOp);
  ~AffinityAnalysis();

  // Runs analysis and populates the resource usage map.
  // May fail if analysis cannot be completed due to unsupported or unknown IR.
  LogicalResult run();

  // Returns the affinity of the global |op| based on its loads.
  // The global storage should be allocated with this affinity and available for
  // fast access from any compatible affinity.
  //
  // If an explicit affinity is provided via a stream.affinity attribute then
  // that will be used in place of analysis. If there are more than one consumer
  // (such as multiple loads) with differing affinities or analysis fails then
  // no affinity is returned. If all affinities are compatible one will be
  // chosen in an unspecified way.
  IREE::Stream::AffinityAttr lookupGlobalAffinity(Operation *op);

  // Populates all potential affinities of the global |op| in |affinities|.
  // Returns false if analysis failed and the set of affinities is unknown.
  bool tryLookupGlobalAffinity(
      Operation *op, SmallVectorImpl<IREE::Stream::AffinityAttr> &affinities);

  // Returns the affinity of the executable |op| based on the op-specific rules
  // as to whether its operands or results control placement. The operation
  // should be scheduled to execute with this affinity and efficiently consume
  // or produce resources that share a compatible affinity.
  //
  // If an explicit affinity is provided via stream.affinity attrs or the
  // affinity op interface then that will be used in place of analysis. If there
  // are multiple possible affinities or analysis fails no affinity is returned.
  // If all affinities are compatible one will be chosen in an unspecified way.
  IREE::Stream::AffinityAttr lookupExecutionAffinity(Operation *op);

  // Populates all potential execution affinities of |op| in |affinities|.
  // Returns false if analysis failed and the set of affinities is unknown.
  bool tryLookupExecutionAffinity(
      Operation *op, SmallVectorImpl<IREE::Stream::AffinityAttr> &affinities);

  // Returns the affinity of |value| based on its producers.
  // The resource should be allocated with this affinity and be usable by any
  // compatible affinity.
  //
  // If there are more than one producer of the value (such as multiple callers)
  // with differing affinities or analysis fails then no affinity is returned.
  // If all affinities are compatible one will be chosen in an unspecified way.
  IREE::Stream::AffinityAttr lookupResourceAffinity(Value value);

  // Populates all potential affinities of |value| in |affinities|.
  // Returns false if analysis failed and the set of affinities is unknown.
  bool tryLookupResourceAffinity(
      Value value, SmallVectorImpl<IREE::Stream::AffinityAttr> &affinities);

private:
  Explorer explorer;
  llvm::BumpPtrAllocator allocator;
  DFX::Solver solver;
};

} // namespace mlir::iree_compiler::IREE::Stream

#endif // IREE_COMPILER_DIALECT_STREAM_ANALYSIS_AFFINITY_H_
