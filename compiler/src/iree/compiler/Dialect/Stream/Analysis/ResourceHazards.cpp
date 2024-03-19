// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Analysis/ResourceHazards.h"

#include <utility>

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
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

ResourceHazardAnalysis::ResourceHazardAnalysis(Operation *rootOp) {
  LLVM_DEBUG({
    asmState = std::make_unique<AsmState>(
        rootOp->getParentWithTrait<OpTrait::IsIsolatedFromAbove>());
  });
}

ResourceHazardAnalysis::~ResourceHazardAnalysis() = default;

LogicalResult ResourceHazardAnalysis::run() { return success(); }

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

  for (auto &producerRange : allProducerRanges) {
    for (auto &consumerRange : allConsumerRanges) {
      if (producerRange.resource == consumerRange.resource) {
        // TODO(#6972): use adjacency tracking sets to handle out-of-order
        // ranges. The basic overlap check only handles perfectly adjacent
        // ranges.
        if (!IREE::Stream::AsyncAccessRange::mayOverlap(producerRange,
                                                        consumerRange)) {
          // No overlap - no hazard.
          continue;
        }
        if (producerRange.isReadOnly() && consumerRange.isReadOnly()) {
          // Read-read is not a hazard.
          continue;
        }
        return true;
      }
    }
  }

  return false;
}

} // namespace mlir::iree_compiler::IREE::Stream
