// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-stream-hoist-allocation"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {
namespace {

//===----------------------------------------------------------------------===//
// -iree-stream-hoist-allocation
//===----------------------------------------------------------------------===//

// Returns a set of values tied to |values| through tied operands.
static llvm::SetVector<Value>
getTiedValues(IREE::Stream::ResourceAllocOp alloc) {
  llvm::SetVector<Value> tiedValued;
  tiedValued.insert(alloc.result_begin(), alloc.result_end());

  SmallVector<Value> worklist;
  worklist.append(alloc.result_begin(), alloc.result_end());

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    for (auto &use : value.getUses()) {
      if (auto tiedOp = dyn_cast<Util::TiedOpInterface>(use.getOwner())) {
        auto tiedResults = tiedOp.getOperandTiedResults(use.getOperandNumber());
        tiedValued.insert(tiedResults.begin(), tiedResults.end());
        worklist.append(tiedResults.begin(), tiedResults.end());
      }
    }
  }

  return tiedValued;
}

class HoistAllocationPass : public HoistAllocationBase<HoistAllocationPass> {
public:
  void runOnOperation() override {
    auto callableOp = getOperation();

    // Skip callable regions with empty or trivial CFG.
    Region *region = callableOp.getCallableRegion();
    if (!region || region->hasOneBlock())
      return;

    auto &domInfo = getAnalysis<DominanceInfo>();
    auto &liveness = getAnalysis<Liveness>();

    CFGLoopInfo loopInfo(domInfo.getDomTree(region));

    // Checks if storage size is known at compile time and below a threshold.
    auto isBelowMaxHoistedAllocSize = [&](Value storageSize) -> bool {
      llvm::APInt constantSize;
      return matchPattern(storageSize, m_ConstantInt(&constantSize)) &&
             constantSize.getSExtValue() < maxHoistedAllocSize;
    };

    // TODO(ezhulenev): It should be possible to split single alloc operation
    // into multiple ones if only some of the allocations are compatible with
    // hoisting. Today we require that all allocation results are hoistable.
    for (auto alloc : llvm::make_early_inc_range(
             region->getOps<IREE::Stream::ResourceAllocOp>())) {

      // Check that allocation is inside a loop CFG
      auto *loop = loopInfo.getLoopFor(alloc->getBlock());
      if (!loop)
        continue;

      auto *predecessor = loop->getOutermostLoop()->getLoopPredecessor();
      if (!predecessor)
        continue;

      // All resources storage size must be below a threshold.
      if (!llvm::all_of(alloc.getStorageSizes(), isBelowMaxHoistedAllocSize))
        continue;

      // Find all tied values derived from alloc results.
      llvm::SetVector<Value> tiedValues = getTiedValues(alloc);

      // Check that all tied values defined in the same block.
      bool sameBlock = llvm::all_of(tiedValues, [&](Value value) {
        return value.getDefiningOp()->getBlock() == alloc->getBlock();
      });
      if (!sameBlock)
        continue;

      // Check that none of the tied values leave the block via termiantor.
      auto *terminator = alloc->getBlock()->getTerminator();
      bool doNotLeak = llvm::all_of(tiedValues, [&](Value value) {
        return llvm::find(terminator->getOperands(), value) ==
               terminator->getOperands().end();
      });
      if (!doNotLeak)
        continue;

      // Check that all values tied to allocation die in the allocation block.
      auto *livenessBlockInfo = liveness.getLiveness(alloc->getBlock());
      if (!livenessBlockInfo)
        continue;

      bool doNotLiveOut = llvm::any_of(tiedValues, [&](Value value) {
        return !livenessBlockInfo->isLiveOut(value);
      });
      if (!doNotLiveOut)
        continue;

      // It's safe to move allocation to the predecessor block.
      alloc->moveBefore(predecessor->getTerminator());
    }
  }
};

} // namespace

std::unique_ptr<InterfacePass<CallableOpInterface>>
createHoistAllocationPass() {
  return std::make_unique<HoistAllocationPass>();
}

} // namespace Stream
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
