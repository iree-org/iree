// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Analysis/ResourceHazards.h"

#include <numeric>
#include <utility>

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Util/Analysis/IntegerDivisibilityAnalysis.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"

#define DEBUG_TYPE "iree-util-dfx"

namespace mlir::iree_compiler::IREE::Stream {

//===----------------------------------------------------------------------===//
// Hazard analysis
//===----------------------------------------------------------------------===//

ResourceHazardAnalysis::ResourceHazardAnalysis(Operation *rootOp)
    : rootOp(rootOp) {
  LLVM_DEBUG({
    asmState = std::make_unique<AsmState>(
        rootOp->getParentWithTrait<OpTrait::IsIsolatedFromAbove>());
  });
}

ResourceHazardAnalysis::~ResourceHazardAnalysis() = default;

LogicalResult ResourceHazardAnalysis::run() { return success(); }

DataFlowSolver *ResourceHazardAnalysis::getDivisibilitySolver() {
  if (solver || solverFailed) {
    return solver.get();
  }
  // Boundaries are often defined above |rootOp|.
  Operation *scopeOp =
      rootOp->hasTrait<OpTrait::IsIsolatedFromAbove>()
          ? rootOp
          : rootOp->getParentWithTrait<OpTrait::IsIsolatedFromAbove>();
  solver = std::make_unique<DataFlowSolver>();
  solver->load<dataflow::SparseConstantPropagation>();
  solver->load<dataflow::DeadCodeAnalysis>();
  solver->load<IREE::Util::IntegerDivisibilityAnalysis>();
  if (failed(solver->initializeAndRun(scopeOp ? scopeOp : rootOp))) {
    solver.reset();
    solverFailed = true;
  }
  return solver.get();
}

// Returns the unit, in bytes, in which |op| accesses storage.
static int64_t getAccessGranularity(Operation *op) {
  if (isa<IREE::Stream::AsyncFillOp>(op)) {
    return IREE::Stream::ResourceConfigAttr::lookup(op).getMinFillGranularity();
  }
  // Each of these lowers to a copy.
  if (isa<IREE::Stream::AsyncCopyOp, IREE::Stream::AsyncUpdateOp,
          IREE::Stream::AsyncSliceOp, IREE::Stream::AsyncCloneOp,
          IREE::Stream::AsyncTransferOp>(op)) {
    return IREE::Stream::ResourceConfigAttr::lookup(op).getMinCopyGranularity();
  }
  if (isa<IREE::Stream::AsyncDispatchOp>(op)) {
    return IREE::Stream::ResourceConfigAttr::lookup(op)
        .getMinAccessGranularity();
  }
  return 1;
}

bool ResourceHazardAnalysis::hasHazard(Operation *producerOp,
                                       Operation *consumerOp) {
  // We only perform analysis on ops implementing the async access interface.
  auto producerAccessOp =
      dyn_cast<IREE::Stream::AsyncAccessOpInterface>(producerOp);
  auto consumerAccessOp =
      dyn_cast<IREE::Stream::AsyncAccessOpInterface>(consumerOp);
  if (!producerAccessOp || !consumerAccessOp) {
    // Fallback to default whole resource checks.
    return llvm::is_contained(producerOp->getUsers(), consumerOp);
  }

  // Query the access ranges of each op.
  SmallVector<AsyncAccessRange> allProducerRanges;
  producerAccessOp.getAsyncAccessRanges(allProducerRanges);
  SmallVector<AsyncAccessRange> allConsumerRanges;
  consumerAccessOp.getAsyncAccessRanges(allConsumerRanges);

  LLVM_DEBUG({
    llvm::dbgs() << "producer: ";
    producerOp->print(llvm::dbgs(), *asmState);
    llvm::dbgs() << "\n";
    llvm::interleave(
        allProducerRanges, llvm::dbgs(),
        [&](auto range) {
          llvm::dbgs() << "  ";
          range.print(llvm::dbgs(), *asmState);
        },
        "\n");
    llvm::dbgs() << "\n";
    llvm::dbgs() << "consumer: ";
    consumerOp->print(llvm::dbgs(), *asmState);
    llvm::dbgs() << "\n";
    llvm::interleave(
        allConsumerRanges, llvm::dbgs(),
        [&](auto range) {
          llvm::dbgs() << "  ";
          range.print(llvm::dbgs(), *asmState);
        },
        "\n");
    llvm::dbgs() << "\n";
  });

  // Touching ranges are disjoint only at a whole unit of both ops' accesses.
  std::optional<int64_t> granularity;
  auto mayOverlapPhysically = [this, &granularity, producerOp,
                               consumerOp](const AsyncAccessRange &lhs,
                                           const AsyncAccessRange &rhs) {
    if (AsyncAccessRange::mayOverlap(lhs, rhs)) {
      return true;
    }
    if (!granularity) {
      granularity = std::lcm(getAccessGranularity(producerOp),
                             getAccessGranularity(consumerOp));
    }
    if (*granularity <= 1) {
      return false;
    }
    Value boundary = lhs.end == rhs.start ? lhs.end : lhs.start;
    DataFlowSolver *divisibilitySolver = getDivisibilitySolver();
    return !divisibilitySolver ||
           !IREE::Util::isProvenMultipleOf(boundary, *granularity,
                                           *divisibilitySolver);
  };
  for (auto &producerRange : allProducerRanges) {
    for (auto &consumerRange : allConsumerRanges) {
      if (producerRange.resource != consumerRange.resource) {
        continue;
      }
      if (producerRange.isReadOnly() && consumerRange.isReadOnly()) {
        continue;
      }
      // TODO(#6972): use adjacency tracking sets to handle out-of-order
      // ranges. The basic overlap check only handles perfectly adjacent ranges.
      if (!mayOverlapPhysically(producerRange, consumerRange)) {
        continue;
      }
      return true;
    }
  }

  return false;
}

} // namespace mlir::iree_compiler::IREE::Stream
