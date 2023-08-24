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

      // Check that allocation dies in the same block where it's defined.
      auto *livenessBlockInfo = liveness.getLiveness(alloc->getBlock());
      if (!livenessBlockInfo)
        continue;

      bool liveOut = llvm::any_of(alloc->getResults(), [&](Value result) {
        return livenessBlockInfo->isLiveOut(result);
      });

      if (!liveOut)
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
