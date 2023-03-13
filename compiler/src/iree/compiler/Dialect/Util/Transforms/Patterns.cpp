// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Transforms/Patterns.h"

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/BitVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {

namespace {

// Erases all |operands| that have a bit set in |elidedOperands|.
static void eraseOperands(MutableOperandRange &operands,
                          llvm::BitVector &elidedOperands) {
  for (int i = elidedOperands.size() - 1; i >= 0; --i) {
    if (elidedOperands.test(i)) {
      operands.erase(i);
    }
  }
}

// Folds block arguments that are always known to have the same value at all
// branch source sites. This is like CSE applied to blocks.
//
// Example:
//   br ^bb1(%0, %0 : index, index)
// ^bb1(%arg0: index, %arg1: index):
// ->
//   br ^bb1(%0 : index)
// ^bb1(%arg0: index):  // %arg1 remapped to %arg0
struct FoldBlockArgumentsPattern
    : public OpInterfaceRewritePattern<CallableOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(CallableOpInterface op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallableRegion()) return failure();
    auto &region = *op.getCallableRegion();
    if (region.empty()) return failure();

    // Analyze all branches in the op to compute the information we'll need to
    // analyze across branch sources.
    struct BlockSource {
      // Branch operation targeting the block.
      mutable BranchOpInterface branchOp;
      // Which successor this source represents.
      unsigned successorIndex;
      // Successor operand index -> index mapping for duplicates.
      // Base/non-duplicated values will be identity.
      // Example: (%a, %b, %a, %b) -> (0, 1, 0, 1)
      SmallVector<int> dupeIndexMap;
    };
    static const int kUnassigned = -1;
    DenseMap<Block *, SmallVector<BlockSource>> blockSourceMap;
    bool hasAnyDupes = false;
    for (auto branchOp : region.getOps<BranchOpInterface>()) {
      for (unsigned successorIndex = 0;
           successorIndex < branchOp->getNumSuccessors(); ++successorIndex) {
        auto *block = branchOp->getSuccessor(successorIndex);
        auto operands = branchOp.getSuccessorOperands(successorIndex);
        BlockSource blockSource;
        blockSource.branchOp = branchOp;
        blockSource.successorIndex = successorIndex;
        blockSource.dupeIndexMap.resize(operands.size(), kUnassigned);
        for (int i = 0; i < operands.size(); ++i) {
          blockSource.dupeIndexMap[i] = i;
          for (int j = 0; j < i; ++j) {
            if (operands[j] == operands[i]) {
              blockSource.dupeIndexMap[i] = j;
              hasAnyDupes |= true;
              break;
            }
          }
        }
        blockSourceMap[block].push_back(blockSource);
      }
    }
    if (!hasAnyDupes) {
      return failure();  // no dupes at all
    }

    rewriter.startRootUpdate(op);

    // Iterate over all blocks after the entry block. We can't change the entry
    // block as it is part of the function signature.
    bool didChange = false;
    for (auto &block : llvm::make_range(++region.getBlocks().begin(),
                                        region.getBlocks().end())) {
      unsigned numArgs = block.getNumArguments();
      if (numArgs == 0) continue;
      auto blockSources = llvm::ArrayRef(blockSourceMap[&block]);
      if (blockSources.size() == 0) continue;

      // Which args we'll end up erasing.
      // We need to do the actual removal after we've done the remapping below
      // as we need the values to still be live and indices consistent with the
      // analysis above.
      llvm::BitVector elidedArgs(numArgs);

      // See if each block argument is foldable across all block sources.
      // In order to fold we need each source to have the same index in its
      // duplication map refering back to the given block argument.
      llvm::BitVector sameValues(numArgs);
      for (unsigned argIndex = 0; argIndex < numArgs; ++argIndex) {
        // Each bit represents an argument that duplicates the arg at argIndex.
        // We walk all the sources and AND their masks together to get the safe
        // set of duplicate operands.
        // Example for %0: (%a, %b, %a) -> b001
        // Example for %1: (%a, %b, %a) -> b000
        sameValues.set();  // note reused
        for (auto &blockSource : blockSources) {
          for (unsigned i = 0; i < numArgs; ++i) {
            if (i == argIndex || blockSource.dupeIndexMap[i] != argIndex) {
              sameValues.reset(i);
            }
          }
        }
        if (sameValues.none()) continue;
        elidedArgs |= sameValues;

        // Replace all of the subsequent duplicate arguments with the first.
        auto baseArg = block.getArgument(argIndex);
        for (unsigned dupeIndex : sameValues.set_bits()) {
          rewriter.replaceAllUsesWith(block.getArgument(dupeIndex), baseArg);
        }
      }

      // Erase all the block arguments we've deduplicated.
      if (elidedArgs.any()) {
        for (auto &blockSource : blockSources) {
          auto successorOperands = blockSource.branchOp.getSuccessorOperands(
              blockSource.successorIndex);
          auto operands = successorOperands.slice(
              successorOperands.getProducedOperandCount(),
              successorOperands.size());
          rewriter.updateRootInPlace(blockSource.branchOp, [&]() {
            eraseOperands(operands, elidedArgs);
          });
        }
        block.eraseArguments(elidedArgs);
        didChange |= !elidedArgs.none();
      }
    }

    if (didChange) {
      rewriter.finalizeRootUpdate(op);
      return success();
    } else {
      rewriter.cancelRootUpdate(op);
      return failure();
    }
  }
};

// Finds branch arguments that come from dominating branch source operands on
// all incoming edges and elides the arguments.
//
// Example:
//  func.func @foo(%arg0: index) {
//    br ^bb1(%arg0 : index)
//  ^bb1(%0: index):
// ->
//  func.func @foo(%arg0: index) {
//    br ^bb1
//  ^bb1:  // %0 remapped to %arg0
struct ElideBranchOperandsPattern
    : public OpInterfaceRewritePattern<CallableOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(CallableOpInterface op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallableRegion()) return failure();
    auto &region = *op.getCallableRegion();
    if (region.empty()) return failure();
    DominanceInfo dominance(op);

    // Analyze all branches to build a map of blocks to their sources.
    struct BlockSource {
      // Branch operation targeting the block.
      mutable BranchOpInterface branchOp;
      // Which successor this source represents.
      unsigned successorIndex;
    };
    DenseMap<Block *, SmallVector<BlockSource>> blockSourceMap;
    for (auto branchOp : region.getOps<BranchOpInterface>()) {
      for (unsigned successorIndex = 0;
           successorIndex < branchOp->getNumSuccessors(); ++successorIndex) {
        auto *block = branchOp->getSuccessor(successorIndex);
        auto operands = branchOp.getSuccessorOperands(successorIndex);
        BlockSource blockSource;
        blockSource.branchOp = branchOp;
        blockSource.successorIndex = successorIndex;
        blockSourceMap[block].push_back(blockSource);
      }
    }

    rewriter.startRootUpdate(op);

    // Iterate over all blocks after the entry block. We can't change the entry
    // block as it is part of the function signature.
    bool didChange = false;
    for (auto &block : llvm::make_range(++region.getBlocks().begin(),
                                        region.getBlocks().end())) {
      unsigned numArgs = block.getNumArguments();
      if (numArgs == 0) continue;
      auto blockSources = llvm::ArrayRef(blockSourceMap[&block]);
      if (blockSources.size() == 0) continue;

      // Which args we'll end up erasing.
      // We need to do the actual removal after we've done the remapping below
      // as we need the values to still be live and indices consistent with the
      // analysis above.
      llvm::BitVector elidedArgs(numArgs);

      for (unsigned argIndex = 0; argIndex < numArgs; ++argIndex) {
        // Find the uniform value passed for the operand of all branches.
        Value uniformValue = nullptr;
        for (auto &blockSource : blockSources) {
          auto operands = blockSource.branchOp.getSuccessorOperands(
              blockSource.successorIndex);
          auto operand = operands[argIndex];
          if (!uniformValue) {
            // First usage.
            uniformValue = operand;
            continue;
          } else if (uniformValue == operand) {
            // Operands match between all previous and the current source.
            continue;
          }

          // Operand for this source differs from previous. This is either
          // because it's non-uniform _or_ that it's a cycle.
          if (auto sourceArg = operand.dyn_cast<BlockArgument>()) {
            // Operand comes from a block argument. If that is the block
            // argument we are analyzing it means there's a cycle (%0 -> %0) and
            // we can ignore it for the purposes of this analysis.
            if (sourceArg.getOwner() == &block &&
                sourceArg.getArgNumber() == argIndex) {
              continue;
            }
          }

          // Non-uniform; can't elide.
          uniformValue = nullptr;
          break;
        }
        if (!uniformValue) continue;

        // See if the uniform value dominates this block; if so we can use it.
        if (!uniformValue.getDefiningOp() ||
            dominance.dominates(uniformValue.getDefiningOp()->getBlock(),
                                &block)) {
          rewriter.replaceAllUsesWith(block.getArgument(argIndex),
                                      uniformValue);
          elidedArgs.set(argIndex);
        }
      }
      if (elidedArgs.none()) continue;

      // Erase all the block arguments we remapped.
      for (auto &blockSource : blockSources) {
        auto successorOperands = blockSource.branchOp.getSuccessorOperands(
            blockSource.successorIndex);
        auto operands =
            successorOperands.slice(successorOperands.getProducedOperandCount(),
                                    successorOperands.size());
        rewriter.updateRootInPlace(blockSource.branchOp, [&]() {
          eraseOperands(operands, elidedArgs);
        });
      }
      block.eraseArguments(elidedArgs);
      didChange |= !elidedArgs.none();
    }

    if (didChange) {
      rewriter.finalizeRootUpdate(op);
      return success();
    } else {
      rewriter.cancelRootUpdate(op);
      return failure();
    }
  }
};

}  // namespace

void populateCommonPatterns(MLIRContext *context, RewritePatternSet &patterns) {
  context->getOrLoadDialect<IREE::Util::UtilDialect>()
      ->getCanonicalizationPatterns(patterns);

  // TODO(benvanik): same as branch folding but for calls.
  patterns.insert<FoldBlockArgumentsPattern, ElideBranchOperandsPattern>(
      context);
}

}  // namespace Util
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
