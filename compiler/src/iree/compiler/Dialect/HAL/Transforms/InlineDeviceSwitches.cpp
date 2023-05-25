// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Inlines a condition region from a switch op into the function at the given
// point. This assumes that the insertion point will only be reached if the
// condition the region is predicated on is true.
static void inlineConditionRegion(Region &conditionRegion, Block *exitBlock,
                                  OpBuilder funcBuilder) {
  assert(!conditionRegion.empty() && "source regions must not be empty");
  assert(conditionRegion.front().getNumArguments() == 0 &&
         "switch does not capture");

  // Splice in the region blocks.
  auto *insertBlock = funcBuilder.getBlock();
  auto postInsertBlockIt = std::next(insertBlock->getIterator())->getIterator();
  auto *insertRegion = insertBlock->getParent();
  insertRegion->getBlocks().splice(postInsertBlockIt,
                                   conditionRegion.getBlocks());
  auto newBlocks = llvm::make_range(std::next(insertBlock->getIterator()),
                                    postInsertBlockIt);
  auto *firstNewBlock = &*newBlocks.begin();

  // Handle the hal.return ops which will transfer control to the exitBlock.
  for (auto &newBlock : newBlocks) {
    if (auto returnOp =
            dyn_cast<IREE::HAL::ReturnOp>(newBlock.getTerminator())) {
      OpBuilder branchBuilder(returnOp);
      branchBuilder.create<cf::BranchOp>(returnOp.getLoc(), exitBlock,
                                         returnOp.getOperands());
      returnOp.erase();
    }
  }

  // Splice the instructions of the inlined entry block into the insert block.
  insertBlock->getOperations().splice(insertBlock->end(),
                                      firstNewBlock->getOperations());
  firstNewBlock->erase();
}

// Inlines each switch condition region into the parent function predicated on
// the switch condition expression.
//
// Since switch conditions are evaluated in the order they are defined we can
// trivially turn the switch into a chain of if-else blocks.
//   if condition_0_match:
//     <inlined condition_0>
//   else
//     if condition_1_match:
//       <inlined condition_1>
//     else ...
static void buildConditionDispatchTable(IREE::HAL::DeviceSwitchOp switchOp,
                                        OpBuilder funcBuilder) {
  // Split the block containing the switch op such that all ops before the
  // switch are before and the switch and the following ops are after.
  // We'll have all of our inlined regions bounce over to the afterBlock with
  // the results of the call and use that to replace the switch op.
  auto *beforeBlock = funcBuilder.getBlock();
  auto *afterBlock = beforeBlock->splitBlock(switchOp);
  SmallVector<Location> locs(switchOp.getNumResults(), switchOp.getLoc());
  auto finalValues = llvm::to_vector<4>(
      afterBlock->addArguments(switchOp.getResultTypes(), locs));

  // Create the blocks we'll use for all our conditions so that we can
  // reference them when inserting the branch ops.
  SmallVector<Block *, 4> conditionMatchBlocks(
      switchOp.getConditionRegions().size());
  SmallVector<Block *, 4> conditionFallthroughBlocks(
      switchOp.getConditionRegions().size());
  for (int i = 0; i < conditionMatchBlocks.size(); ++i) {
    conditionMatchBlocks[i] = funcBuilder.createBlock(afterBlock);
    conditionFallthroughBlocks[i] = funcBuilder.createBlock(afterBlock);
  }

  funcBuilder.setInsertionPoint(beforeBlock, beforeBlock->end());
  for (auto condition :
       llvm::enumerate(llvm::zip_equal(switchOp.getConditions().getValue(),
                                       switchOp.getConditionRegions()))) {
    auto conditionAttr = llvm::cast<IREE::HAL::MatchAttrInterface>(
        std::get<0>(condition.value()));
    auto &conditionRegion = std::get<1>(condition.value());

    // Insert the branch based on the match. We either match and jump to a
    // block that will contain the inlined region or don't match and need to
    // fall through.
    auto isMatch = conditionAttr.buildConditionExpression(
        switchOp.getLoc(), switchOp.getDevice(), funcBuilder);
    auto *matchBlock = conditionMatchBlocks[condition.index()];
    auto *fallthroughBlock = conditionFallthroughBlocks[condition.index()];
    funcBuilder.create<cf::CondBranchOp>(switchOp.getLoc(), isMatch, matchBlock,
                                         fallthroughBlock);

    // Block that contains the inlined region and then jumps out of the chain.
    funcBuilder.setInsertionPointToStart(matchBlock);
    inlineConditionRegion(conditionRegion, afterBlock, funcBuilder);

    // Block that we enter to check the next condition.
    funcBuilder.setInsertionPointToStart(fallthroughBlock);
    if (condition.index() + 1 < conditionFallthroughBlocks.size()) {
      // Just continue on - the next loop iteration for the following
      // condition will add its IR to the block.
    } else {
      // Fallthrough of all expressions; die if we expected return values.
      funcBuilder.create<IREE::Util::UnreachableOp>(
          switchOp.getLoc(),
          "device not supported in the compiled configuration");
    }
  }

  // Remove the switch op and replace its results with the final joined
  // results.
  switchOp.replaceAllUsesWith(finalValues);
}

class InlineDeviceSwitchesPass
    : public PassWrapper<InlineDeviceSwitchesPass, OperationPass<void>> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Util::UtilDialect>();
  }

  StringRef getArgument() const override {
    return "iree-hal-inline-device-switches";
  }

  StringRef getDescription() const override {
    return "Inlines hal.device.switch condition regions";
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    SmallVector<IREE::HAL::DeviceSwitchOp, 4> switchOps;
    funcOp->walk([&](IREE::HAL::DeviceSwitchOp switchOp) {
      switchOps.push_back(switchOp);
    });
    for (auto switchOp : switchOps) {
      OpBuilder funcBuilder(switchOp);
      buildConditionDispatchTable(switchOp, funcBuilder);
      switchOp.erase();
    }
  }
};

std::unique_ptr<OperationPass<void>> createInlineDeviceSwitchesPass() {
  return std::make_unique<InlineDeviceSwitchesPass>();
}

static PassRegistration<InlineDeviceSwitchesPass> pass;

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
