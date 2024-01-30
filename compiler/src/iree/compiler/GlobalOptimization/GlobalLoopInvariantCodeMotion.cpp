// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#include "llvm/Support/Debug.h"

namespace mlir::iree_compiler::GlobalOptimization {

namespace {

struct GlobalLoopInvariantCodeMotionPass
    : public GlobalLoopInvariantCodeMotionBase<
          GlobalLoopInvariantCodeMotionPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    FunctionOpInterface funcOp = getOperation();
    if (funcOp.isDeclaration())
      return;

    SmallVector<scf::WhileOp> worklist;
    for (Operation &op : funcOp.getCallableRegion()->getOps()) {
      if (auto whileOp = llvm::dyn_cast<scf::WhileOp>(op)) {
        worklist.push_back(whileOp);
      }
    }

    for (auto whileOp : worklist) {
      SmallVector<Operation *> hoistedOps;
      moveLoopInvariantCode(
          whileOp.getLoopRegions(),
          [&](Value value, Region *region) {
            return whileOp.isDefinedOutsideOfLoop(value);
          },
          [&](Operation *invOp, Region *region) {
            return isMemoryEffectFree(invOp) && isSpeculatable(invOp);
          },
          [&](Operation *invOp, Region *region) {
            if (region == &whileOp.getAfter()) {
              hoistedOps.push_back(invOp);
            }
            whileOp.moveOutOfLoop(invOp);
            return;
          });

      if (hoistedOps.empty()) {
        continue;
      }

      IRRewriter rewriter(context);
      rewriter.setInsertionPoint(whileOp);

      IRMapping mapper;
      Block *srcBlock = whileOp.getBeforeBody();
      for (auto [arg, init] :
           llvm::zip_equal(srcBlock->getArguments(), whileOp.getInits())) {
        mapper.map(arg, init);
      }

      auto cloneOptions =
          Operation::CloneOptions::all().cloneRegions(false).cloneOperands(
              false);
      SmallVector<Operation *> clonedOps;
      for (auto &op : *srcBlock) {
        if (isa<scf::ConditionOp>(op)) {
          break;
        }
        auto clonedOp = op.clone(mapper, cloneOptions);
        rewriter.insert(clonedOp);
        clonedOps.push_back(clonedOp);
      }

      SmallVector<Value> operands;
      for (auto [clonedOp, srcOp] : llvm::zip(clonedOps, *srcBlock)) {
        operands.resize(srcOp.getNumOperands());
        llvm::transform(
            srcOp.getOperands(), operands.begin(),
            [&](Value operand) { return mapper.lookupOrDefault(operand); });
        clonedOp->setOperands(operands);
        for (auto [srcRegion, clonedRegion] :
             llvm::zip_equal(srcOp.getRegions(), clonedOp->getRegions())) {
          srcRegion.cloneInto(&clonedRegion, mapper);
        }
      }

      auto condOp = cast<scf::ConditionOp>(srcBlock->getTerminator());
      auto clonedCondition = mapper.lookupOrDefault(condOp.getCondition());
      SmallVector<Value> clonedArgs =
          llvm::map_to_vector(condOp.getArgs(), [&](Value result) {
            return mapper.lookupOrDefault(result);
          });
      auto loc = condOp.getLoc();
      auto ifOp = rewriter.create<scf::IfOp>(
          loc, clonedCondition,
          [&](OpBuilder &builder, Location loc) {
            auto rotatedWhileOp = builder.create<scf::WhileOp>(
                whileOp.getLoc(), whileOp.getResultTypes(), clonedArgs,
                [&](OpBuilder &builder, Location loc, ValueRange args) {
                  // Before block is created below.
                },
                [&](OpBuilder &builder, Location loc, ValueRange args) {
                  // Pass-through values to the before block.
                  builder.create<scf::YieldOp>(loc, args);
                });

            IRRewriter rewriter(builder);
            Block *beforeBlock = rotatedWhileOp.getBeforeBody();
            rewriter.mergeBlocks(whileOp.getAfterBody(), beforeBlock,
                                 beforeBlock->getArguments());
            auto yieldOp = cast<scf::YieldOp>(beforeBlock->getTerminator());
            rewriter.mergeBlocks(whileOp.getBeforeBody(), beforeBlock,
                                 yieldOp.getResults());
            rewriter.eraseOp(yieldOp);

            for (auto hoistedOp : hoistedOps) {
              rewriter.moveOpBefore(hoistedOp, rotatedWhileOp);
            }

            builder.create<scf::YieldOp>(loc, rotatedWhileOp->getResults());
          },
          [&](OpBuilder &builder, Location loc) {
            builder.create<scf::YieldOp>(loc, clonedArgs);
          });

      rewriter.replaceOp(whileOp, ifOp);
    }
  }
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGlobalLoopInvariantCodeMotionPass() {
  return std::make_unique<GlobalLoopInvariantCodeMotionPass>();
}
} // namespace mlir::iree_compiler::GlobalOptimization
