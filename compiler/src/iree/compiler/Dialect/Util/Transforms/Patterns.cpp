// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Transforms/Patterns.h"

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::iree_compiler::IREE::Util {

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

// Returns true if a block contains only unreachable code.
// This handles both util.unreachable (terminator) and util.scf.unreachable
// (non-terminator) when inside of an scf op region.
static bool isBlockUnreachable(Block *block) {
  if (!block || block->empty()) {
    return false;
  }

  // Check if block contains only util.unreachable terminator.
  if (block->getOperations().size() == 1 &&
      isa<IREE::Util::UnreachableOp>(block->getTerminator())) {
    return true;
  }

  // Check if block contains util.scf.unreachable (possibly followed by yield).
  for (auto &op : *block) {
    if (isa<IREE::Util::SCFUnreachableOp>(op)) {
      return true;
    }
    // Stop if we hit a non-side-effect-free op before unreachable.
    if (!mlir::isMemoryEffectFree(&op) &&
        !op.hasTrait<OpTrait::IsTerminator>()) {
      return false;
    }
  }

  return false;
}

// Replaces an SCF op that has become entirely unreachable with
// util.scf.unreachable and poison values for its results.
static void replaceSCFOpWithUnreachable(PatternRewriter &rewriter,
                                        Operation *op, StringRef message) {
  Location loc = op->getLoc();

  // First insert util.scf.unreachable before the op.
  // The existing folder patterns will convert this to util.unreachable
  // if we're in a CFG context.
  rewriter.setInsertionPoint(op);
  IREE::Util::SCFUnreachableOp::create(rewriter, loc, message);

  // Now create poison values for all results and replace the op.
  rewriter.replaceOp(op, IREE::Util::SCFUnreachableOp::createPoisonValues(
                             rewriter, loc, op->getResultTypes()));
}

// Inlines a block's contents before the given operation and handles any yield
// terminator by replacing the operation with the yield's operands.
//
// This handles the common pattern of:
//   1. Save yield operands (if present)
//   2. Inline the block
//   3. Erase the yield (if it was present)
//   4. Replace the op with yield operands (or just erase if no yield)
static void inlineSCFBlockAndReplaceOp(PatternRewriter &rewriter, Block &block,
                                       Operation *op) {
  if (auto yieldOp = dyn_cast<scf::YieldOp>(block.getTerminator())) {
    // Get yield operands before any modifications.
    SmallVector<Value> yieldOperands(yieldOp.getOperands());
    // First inline the block (with yield still in it).
    rewriter.inlineBlockBefore(&block, op);
    // Now erase the yield that was inlined.
    rewriter.eraseOp(yieldOp);
    // Finally replace the op with the yield operands.
    rewriter.replaceOp(op, yieldOperands);
  } else {
    // No yield, no results.
    rewriter.inlineBlockBefore(&block, op);
    rewriter.eraseOp(op);
  }
}

// Folds block arguments that are always known to have the same value at all
// branch source sites. This is like CSE applied to block arguments.
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
    if (!op.getCallableRegion())
      return failure();
    auto &region = *op.getCallableRegion();
    if (region.empty() || region.hasOneBlock())
      return failure();

    // Analyze all branches in the op to compute the information we'll need to
    // analyze across branch sources.
    struct BlockSource {
      // Branch operation targeting the block.
      mutable BranchOpInterface branchOp;
      // Which successor this source represents.
      unsigned successorIndex;
      // Equivalence classes for all arguments indicating which have the same
      // value at the source. Base/non-duplicated values will be identity.
      // Example: (%a, %b, %a, %b, %c) -> (0, 2), (1, 3), (4)
      llvm::EquivalenceClasses<unsigned> duplicates;
    };
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
        for (int i = 0; i < operands.size(); ++i) {
          blockSource.duplicates.insert(i);
          for (int j = 0; j < i; ++j) {
            if (operands[j] == operands[i]) {
              blockSource.duplicates.unionSets(i, j);
              hasAnyDupes |= true;
              break;
            }
          }
        }
        blockSourceMap[block].push_back(std::move(blockSource));
      }
    }
    if (!hasAnyDupes) {
      return failure(); // no dupes at all
    }

    rewriter.startOpModification(op);

    // Iterate over all blocks after the entry block. We can't change the entry
    // block as it is part of the function signature.
    bool didChange = false;
    for (auto &block : llvm::make_range(++region.getBlocks().begin(),
                                        region.getBlocks().end())) {
      unsigned numArgs = block.getNumArguments();
      if (numArgs == 0)
        continue;
      auto blockSources = llvm::ArrayRef(blockSourceMap[&block]);
      if (blockSources.size() == 0)
        continue;

      // Which args we'll end up erasing.
      // We need to do the actual removal after we've done the remapping below
      // as we need the values to still be live and indices consistent with the
      // analysis above.
      llvm::BitVector elidedArgs(numArgs);

      // See if each block argument is foldable across all block sources.
      // In order to fold we need each source to share some duplicates but note
      // that the sources may not have identical sets.
      llvm::BitVector sameValues(numArgs);   // reused
      llvm::BitVector sourceValues(numArgs); // reused
      for (unsigned argIndex = 0; argIndex < numArgs; ++argIndex) {
        // Each bit represents an argument that duplicates the arg at argIndex.
        // We walk all the sources and AND their masks together to get the safe
        // set of duplicate operands.
        // Example for %0: (%a, %b, %a) -> b001
        // Example for %1: (%a, %b, %a) -> b000
        sameValues.set(); // note reused
        for (auto &blockSource : blockSources) {
          sourceValues.reset();
          for (auto mit = blockSource.duplicates.findLeader(argIndex);
               mit != blockSource.duplicates.member_end(); ++mit) {
            sourceValues.set(*mit);
          }
          sameValues &= sourceValues;
        }
        if (sameValues.none()) {
          continue; // arg unused/not duplicated
        }

        // Remove the base argument from the set so we don't erase it and can
        // point all duplicate args at it.
        int baseArgIndex = sameValues.find_first();
        sameValues.reset(baseArgIndex);
        elidedArgs |= sameValues;

        // Replace all of the subsequent duplicate arguments with the first.
        auto baseArg = block.getArgument(baseArgIndex);
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
          rewriter.modifyOpInPlace(blockSource.branchOp, [&]() {
            eraseOperands(operands, elidedArgs);
          });
        }
        block.eraseArguments(elidedArgs);
        didChange |= !elidedArgs.none();
      }
    }

    if (didChange) {
      rewriter.finalizeOpModification(op);
      return success();
    } else {
      rewriter.cancelOpModification(op);
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
    if (!op.getCallableRegion())
      return failure();
    auto &region = *op.getCallableRegion();
    if (region.empty())
      return failure();
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

    rewriter.startOpModification(op);

    // Iterate over all blocks after the entry block. We can't change the entry
    // block as it is part of the function signature.
    bool didChange = false;
    for (auto &block : llvm::make_range(++region.getBlocks().begin(),
                                        region.getBlocks().end())) {
      unsigned numArgs = block.getNumArguments();
      if (numArgs == 0)
        continue;
      auto blockSources = llvm::ArrayRef(blockSourceMap[&block]);
      if (blockSources.size() == 0)
        continue;

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
          if (auto sourceArg = dyn_cast<BlockArgument>(operand)) {
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
        if (!uniformValue)
          continue;

        // See if the uniform value dominates this block; if so we can use it.
        if (!uniformValue.getDefiningOp() ||
            dominance.dominates(uniformValue.getDefiningOp()->getBlock(),
                                &block)) {
          rewriter.replaceAllUsesWith(block.getArgument(argIndex),
                                      uniformValue);
          elidedArgs.set(argIndex);
        }
      }
      if (elidedArgs.none())
        continue;

      // Erase all the block arguments we remapped.
      for (auto &blockSource : blockSources) {
        auto successorOperands = blockSource.branchOp.getSuccessorOperands(
            blockSource.successorIndex);
        auto operands =
            successorOperands.slice(successorOperands.getProducedOperandCount(),
                                    successorOperands.size());
        rewriter.modifyOpInPlace(blockSource.branchOp, [&]() {
          eraseOperands(operands, elidedArgs);
        });
      }
      block.eraseArguments(elidedArgs);
      didChange |= !elidedArgs.none();
    }

    if (didChange) {
      rewriter.finalizeOpModification(op);
      return success();
    } else {
      rewriter.cancelOpModification(op);
      return failure();
    }
  }
};

// Converts an scf.index_switch with a single case into an scf.if.
//
// Example:
//  scf.index_switch %case : i32
//  case 0 {
//    %foo = ...
//    scf.yield %foo : i32
//  }
//  default {
//    %default = ...
//    scf.yield %default : i32
//  }
// ->
//  %case_0 = arith.cmpi eq, %case, %c0 : index
//  scf.if %case_0 -> i32 {
//    %foo = ...
//    scf.yield %foo : i32
//  } else {
//    %default = ...
//    scf.yield %default : i32
//  }
struct IndexSwitchToIfPattern : public OpRewritePattern<scf::IndexSwitchOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(scf::IndexSwitchOp switchOp,
                                PatternRewriter &rewriter) const override {
    if (switchOp.getNumCases() != 1)
      return failure();
    Value caseValue = arith::ConstantIndexOp::create(
        rewriter, switchOp.getLoc(), switchOp.getCases().front());
    Value isCaseValue = rewriter.createOrFold<arith::CmpIOp>(
        switchOp.getLoc(), arith::CmpIPredicate::eq, switchOp.getArg(),
        caseValue);
    auto ifOp = scf::IfOp::create(rewriter, switchOp.getLoc(),
                                  switchOp.getResultTypes(), isCaseValue);
    rewriter.inlineRegionBefore(switchOp.getCaseRegions().front(),
                                ifOp.getThenRegion(),
                                ifOp.getThenRegion().begin());
    rewriter.inlineRegionBefore(switchOp.getDefaultRegion(),
                                ifOp.getElseRegion(),
                                ifOp.getElseRegion().begin());
    rewriter.replaceOp(switchOp, ifOp);
    return success();
  }
};

// Combines adjacent and compatible scf.index_switch ops into one.
// We sink any ops from the first switch into the second and erase the first.
// Note that since the second may implicitly capture the results of the first
// we have to handle mapping the scf.yield results to their new local values in
// the second.
//
// Example:
//  scf.index_switch %case
//  case 0 {
//    foo
//    scf.yield
//  }
//  default {
//    foo_default
//    scf.yield
//  }
//  scf.index_switch %case
//  case 0 {
//    bar
//    scf.yield
//  }
//  default {
//    bar_default
//    scf.yield
//  }
// ->
//  scf.index_switch %case
//  case 0 {
//    foo
//    bar
//    scf.yield
//  }
//  default {
//    foo_default
//    bar_default
//    scf.yield
//  }
struct MergeIndexSwitchPattern : public OpRewritePattern<scf::IndexSwitchOp> {
  MergeIndexSwitchPattern(MLIRContext *context)
      : OpRewritePattern(context, 1000) {}
  LogicalResult matchAndRewrite(scf::IndexSwitchOp nextOp,
                                PatternRewriter &rewriter) const override {
    // Inspect the previous op to see if it's also a switch.
    auto prevOp =
        dyn_cast_if_present<scf::IndexSwitchOp>(nextOp->getPrevNode());
    if (!prevOp)
      return failure();

    // Require that the cases line up exactly. There's probably some merging
    // we could do in other cases but it'd be best to leave other patterns to
    // hoist/CSE cases/etc instead.
    if (prevOp.getNumCases() != nextOp.getNumCases())
      return rewriter.notifyMatchFailure(nextOp, "number of cases differ");
    if (!llvm::equal(prevOp.getCases(), nextOp.getCases()))
      return rewriter.notifyMatchFailure(nextOp, "case values differ");

    // Create a new switch to replace nextOp that contains the same cases but
    // combined results from both ops.
    SmallVector<Type> newResultTypes;
    llvm::append_range(newResultTypes, prevOp.getResultTypes());
    llvm::append_range(newResultTypes, nextOp.getResultTypes());
    auto newOp = scf::IndexSwitchOp::create(
        rewriter, rewriter.getFusedLoc({prevOp.getLoc(), nextOp.getLoc()}),
        newResultTypes, prevOp.getArg(), prevOp.getCases(),
        prevOp.getNumCases());
    SmallVector<std::pair<Value, Value>> resultReplacements;
    for (auto [oldResult, newResult] :
         llvm::zip_equal(prevOp.getResults(),
                         newOp.getResults().slice(0, prevOp.getNumResults()))) {
      resultReplacements.push_back(std::make_pair(oldResult, newResult));
    }
    for (auto [oldResult, newResult] :
         llvm::zip_equal(nextOp.getResults(),
                         newOp.getResults().slice(prevOp.getNumResults(),
                                                  nextOp.getNumResults()))) {
      resultReplacements.push_back(std::make_pair(oldResult, newResult));
    }

    // NOTE: the results of prevOp may be implicitly captured by nextOp and we
    // need to build a mapping of the prevOp results to their cloned values in
    // nextOp once merged.

    auto cloneRegions = [&](Region &regionA, Region &regionB, Region &target) {
      SmallVector<Value> yieldValues;
      IRMapping localMapping;
      auto targetBuilder = OpBuilder::atBlockBegin(&target.emplaceBlock());

      // Clone regionA into target and map any results from prevOp to the cloned
      // values for the particular case.
      auto yieldA = *regionA.getOps<scf::YieldOp>().begin();
      for (auto &op : regionA.getOps()) {
        if (op.hasTrait<OpTrait::IsTerminator>())
          continue;
        // Clone each op and map its original value to the new local value.
        targetBuilder.clone(op, localMapping);
      }
      for (auto [yieldValue, resultValue] :
           llvm::zip_equal(yieldA.getOperands(), prevOp.getResults())) {
        // Track the new local values yielded by the region.
        auto newValue = localMapping.lookupOrDefault(yieldValue);
        yieldValues.push_back(newValue);
        localMapping.map(resultValue, newValue);
      }

      // Clone regionB into target.
      auto yieldB = *regionB.getOps<scf::YieldOp>().begin();
      for (auto &op : regionB.getOps()) {
        if (op.hasTrait<OpTrait::IsTerminator>())
          continue;
        // Clone each op and map its original value to the new local value.
        targetBuilder.clone(op, localMapping);
      }
      for (auto yieldValue : yieldB.getOperands()) {
        // Track the new local values yielded by the region, ensuring we check
        // what the first region may have produced and been captured implicitly.
        yieldValues.push_back(localMapping.lookupOrNull(yieldValue));
      }

      // Add the merged yield containing results from both regions.
      scf::YieldOp::create(
          targetBuilder,
          targetBuilder.getFusedLoc(yieldA.getLoc(), yieldB.getLoc()),
          yieldValues);
    };

    // Merge regions from both prevOp and nextOp into the newOp.
    for (auto [prevRegion, nextRegion, newRegion] :
         llvm::zip_equal(prevOp.getCaseRegions(), nextOp.getCaseRegions(),
                         newOp.getCaseRegions())) {
      cloneRegions(prevRegion, nextRegion, newRegion);
    }
    cloneRegions(prevOp.getDefaultRegion(), nextOp.getDefaultRegion(),
                 newOp.getDefaultRegion());

    // Update uses of the old results from both ops to the new ones.
    for (auto [oldResult, newResult] : resultReplacements) {
      rewriter.replaceAllUsesWith(oldResult, newResult);
    }
    rewriter.eraseOp(prevOp);
    rewriter.eraseOp(nextOp);

    return success();
  }
};

// Simplifies scf.if when one branch contains only util.unreachable.
//
// Example:
//  scf.if %cond {
//    util.unreachable
//  } else {
//    "some.op"() : () -> ()
//  }
// ->
//  // TODO: add assertion that !%cond when we have util.assume.* propagation
//  // to indicate that the opposite of `%cond` is true.
//  "some.op"() : () -> ()
struct SimplifyIfWithUnreachablePattern : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    // Check if either region contains an unreachable indicator.
    // If neither branch is unreachable there's nothing to do.
    bool thenIsUnreachable = false;
    if (!ifOp.getThenRegion().empty()) {
      thenIsUnreachable = isBlockUnreachable(&ifOp.getThenRegion().front());
    }
    bool elseIsUnreachable = false;
    if (!ifOp.getElseRegion().empty()) {
      elseIsUnreachable = isBlockUnreachable(&ifOp.getElseRegion().front());
    }
    if (!thenIsUnreachable && !elseIsUnreachable) {
      return failure();
    }

    // Both branches unreachable means the whole thing is unreachable.
    if (thenIsUnreachable && elseIsUnreachable) {
      replaceSCFOpWithUnreachable(rewriter, ifOp, "both branches unreachable");
      return success();
    }

    // If only one branch is unreachable, inline the other branch.
    // We need to properly map the scf.yield operands to the if op results.
    //
    // TODO(benvanik): add a util.assume that operates on i1 conditions and add
    // that here to the parent so that subsequent analysis can propagate it.
    if (thenIsUnreachable) {
      // Then is unreachable so condition must be false: inline else.
      if (!ifOp.getElseRegion().empty()) {
        Block &elseBlock = ifOp.getElseRegion().front();
        inlineSCFBlockAndReplaceOp(rewriter, elseBlock, ifOp);
      } else {
        // No else block. Create poison values for any results.
        if (ifOp.getNumResults() > 0) {
          rewriter.replaceOp(
              ifOp, IREE::Util::SCFUnreachableOp::createPoisonValues(
                        rewriter, ifOp.getLoc(), ifOp.getResultTypes()));
        } else {
          rewriter.eraseOp(ifOp);
        }
      }
      return success();
    }

    // Else is unreachable so condition must be true: inline then.
    assert(elseIsUnreachable);
    if (!ifOp.getThenRegion().empty()) {
      Block &thenBlock = ifOp.getThenRegion().front();
      inlineSCFBlockAndReplaceOp(rewriter, thenBlock, ifOp);
    } else {
      // No then block. Create poison values for any results.
      if (ifOp.getNumResults() > 0) {
        rewriter.replaceOp(ifOp,
                           IREE::Util::SCFUnreachableOp::createPoisonValues(
                               rewriter, ifOp.getLoc(), ifOp.getResultTypes()));
      } else {
        rewriter.eraseOp(ifOp);
      }
    }
    return success();
  }
};

// Simplifies scf.while when the body contains unreachable.
struct SimplifyWhileWithUnreachablePattern
    : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
    // Check if the after region (loop body) is unreachable.
    if (whileOp.getAfter().empty()) {
      return failure();
    }
    Block *afterBlock = whileOp.getAfterBody();
    if (!isBlockUnreachable(afterBlock)) {
      return failure();
    }

    // If the loop body is unreachable it means the loop condition must
    // always be false on the first iteration (otherwise we'd hit unreachable).
    // We can replace the entire loop with the values that would be returned
    // from the condition region when the condition is false.
    // For scf.while the results come from the scf.condition operands and
    // not from the inits directly.
    if (whileOp.getNumResults() > 0) {
      // The condition must evaluate to false on the first iteration so we
      // simulate executing just the before region with the init values.
      // We need to map the block arguments to the init values and then
      // get what the condition would yield.
      SmallVector<Value> resultValues;
      Block *beforeBlock = whileOp.getBeforeBody();
      auto condOp = cast<scf::ConditionOp>(beforeBlock->getTerminator());
      IRMapping mapping;
      for (auto [blockArg, initVal] :
           llvm::zip_equal(beforeBlock->getArguments(), whileOp.getInits())) {
        mapping.map(blockArg, initVal);
      }
      for (Value condOperand : condOp.getArgs()) {
        resultValues.push_back(mapping.lookupOrDefault(condOperand));
      }
      rewriter.replaceOp(whileOp, resultValues);
    } else {
      rewriter.eraseOp(whileOp);
    }
    return success();
  }
};

// Simplifies scf.index_switch when cases contain unreachable.
struct SimplifyIndexSwitchWithUnreachablePattern
    : public OpRewritePattern<scf::IndexSwitchOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::IndexSwitchOp switchOp,
                                PatternRewriter &rewriter) const override {
    // Collect which cases are unreachable.
    SmallVector<int64_t> remainingCases;
    SmallVector<Region *> remainingRegions;
    bool hasChanges = false;
    for (auto [idx, caseRegion] : llvm::enumerate(switchOp.getCaseRegions())) {
      if (!caseRegion.empty()) {
        Block &block = caseRegion.front();
        // Skip this case if it contains an unreachable indicator.
        if (isBlockUnreachable(&block)) {
          hasChanges = true;
          continue;
        }
      }
      remainingCases.push_back(switchOp.getCases()[idx]);
      remainingRegions.push_back(&caseRegion);
    }

    // Check default region.
    bool defaultIsUnreachable = false;
    if (!switchOp.getDefaultRegion().empty()) {
      Block &defaultBlock = switchOp.getDefaultRegion().front();
      if (isBlockUnreachable(&defaultBlock)) {
        defaultIsUnreachable = true;
        hasChanges = true;
      }
    }
    if (!hasChanges) {
      return failure();
    }

    // If all cases are unreachable replace with unreachable.
    if (remainingCases.empty() && defaultIsUnreachable) {
      replaceSCFOpWithUnreachable(rewriter, switchOp,
                                  "all switch cases unreachable");
      return success();
    }

    // If only default remains inline it.
    if (remainingCases.empty() && !defaultIsUnreachable) {
      if (!switchOp.getDefaultRegion().empty()) {
        Block &defaultBlock = switchOp.getDefaultRegion().front();
        inlineSCFBlockAndReplaceOp(rewriter, defaultBlock, switchOp);
      } else {
        // Empty default. Create poison values for any results.
        if (switchOp.getNumResults() > 0) {
          rewriter.replaceOp(
              switchOp,
              IREE::Util::SCFUnreachableOp::createPoisonValues(
                  rewriter, switchOp.getLoc(), switchOp.getResultTypes()));
        } else {
          rewriter.eraseOp(switchOp);
        }
      }
      return success();
    }

    // Create a new switch with only the non-unreachable cases.
    auto newSwitch = scf::IndexSwitchOp::create(
        rewriter, switchOp.getLoc(), switchOp.getResultTypes(),
        switchOp.getArg(), remainingCases, remainingCases.size());

    // Move the remaining case regions.
    for (auto [idx, region] : llvm::enumerate(remainingRegions)) {
      newSwitch.getCaseRegions()[idx].takeBody(*region);
    }

    // Move default region if it's not unreachable.
    if (!defaultIsUnreachable) {
      newSwitch.getDefaultRegion().takeBody(switchOp.getDefaultRegion());
    } else {
      // Default is unreachable - create util.scf.unreachable with proper yield.
      auto &defaultBlock = newSwitch.getDefaultRegion().emplaceBlock();
      rewriter.setInsertionPointToEnd(&defaultBlock);
      IREE::Util::SCFUnreachableOp::createRegionTerminator(
          rewriter, switchOp.getLoc(), switchOp.getResultTypes(),
          "default unreachable");
    }

    rewriter.replaceOp(switchOp, newSwitch.getResults());
    return success();
  }
};

// Simplifies scf.for when the body contains unreachable.
struct SimplifyForWithUnreachablePattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    // Check if the loop body ends with unreachable.
    if (forOp.getRegion().empty()) {
      return failure();
    }

    // We need to be careful - only optimize if we can prove the loop body
    // always hits unreachable on the first iteration.
    Block &body = forOp.getRegion().front();
    // Check if body contains an unreachable indicator.
    if (!isBlockUnreachable(&body)) {
      return failure();
    }

    // The loop will hit unreachable on first iteration if it runs at all.
    // For now, conservatively assume the loop may not execute and just
    // return the init values. Unlike scf.while, scf.for doesn't have a
    // condition region with potential side effects - the bounds are just
    // values.
    //
    // TODO(benvanik): use range analysis to determine if lb < ub and insert
    // util.scf.unreachable if we can prove the loop executes at least once.
    rewriter.replaceOp(forOp, forOp.getInitArgs());
    return success();
  }
};

// Simplifies unconditional branches to blocks that are unreachable.
// This is likely to happen via other patterns but we preserve this for defense.
struct SimplifyBranchToUnreachablePattern
    : public OpRewritePattern<cf::BranchOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(cf::BranchOp branchOp,
                                PatternRewriter &rewriter) const override {
    // Check if the destination block contains an unreachable indicator (_and_
    // no side-effecting ops we need to preserve).
    Block *dest = branchOp.getDest();
    if (!isBlockUnreachable(dest)) {
      return failure();
    }

    // Replace the branch with util.unreachable.
    rewriter.replaceOpWithNewOp<IREE::Util::UnreachableOp>(
        branchOp, rewriter.getStringAttr("branching to unreachable"));
    return success();
  }
};

// Simplifies conditional branches to blocks that are unreachable.
//
// Truth table for branch reachability:
//   ┌─────────────────┬─────────────────┬──────────────────────┐
//   │  True Branch    │  False Branch   │  Change              │
//   ├─────────────────┼─────────────────┼──────────────────────┤
//   │  reachable      │  reachable      │  no change           │
//   │  unreachable    │  unreachable    │  util.unreachable    │
//   │  unreachable    │  reachable      │  cf.br ^false        │
//   │  reachable      │  unreachable    │  cf.br ^true         │
//   └─────────────────┴─────────────────┴──────────────────────┘
struct SimplifyCondBranchToUnreachablePattern
    : public OpRewritePattern<cf::CondBranchOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(cf::CondBranchOp condBr,
                                PatternRewriter &rewriter) const override {
    // Check if either destination only contains an unreachable indicator and
    // replace the branch depending on reachability.
    Block *trueDest = condBr.getTrueDest();
    Block *falseDest = condBr.getFalseDest();
    const bool trueUnreachable = isBlockUnreachable(trueDest);
    const bool falseUnreachable = isBlockUnreachable(falseDest);
    if (!trueUnreachable && !falseUnreachable) {
      return failure();
    } else if (trueUnreachable && falseUnreachable) {
      // Both branches lead to unreachable.
      // This is rare but may occur if we've optimized both blocks independently
      // and they ended up as unreachable. By swapping the op with unreachable
      // we'll (eventually) end up with no incoming edges to the blocks and
      // they'll get cleaned up and pattern application will then start to
      // remove code from the parent block as it goes.
      rewriter.replaceOpWithNewOp<IREE::Util::UnreachableOp>(
          condBr, rewriter.getStringAttr("both branches unreachable"));
      return success();
    } else if (trueUnreachable) {
      // True branch is unreachable, so condition must be false.
      //
      // TODO(benvanik): add assertion about the condition (true or false) when
      // we have util.assume.int that could benefit from knowing the condition
      // is false (same with true branch).
      rewriter.replaceOpWithNewOp<cf::BranchOp>(condBr, falseDest,
                                                condBr.getFalseDestOperands());
      return success();
    } else {
      // False branch is unreachable, so condition must be true.
      rewriter.replaceOpWithNewOp<cf::BranchOp>(condBr, trueDest,
                                                condBr.getTrueDestOperands());
      return success();
    }
  }
};

} // namespace

void populateCommonPatterns(MLIRContext *context, RewritePatternSet &patterns) {
  context->getOrLoadDialect<IREE::Util::UtilDialect>()
      ->getCanonicalizationPatterns(patterns);

  patterns.insert<FoldBlockArgumentsPattern, ElideBranchOperandsPattern>(
      context);

  patterns.insert<IndexSwitchToIfPattern, MergeIndexSwitchPattern>(context);

  patterns.insert<
      SimplifyIfWithUnreachablePattern, SimplifyWhileWithUnreachablePattern,
      SimplifyIndexSwitchWithUnreachablePattern,
      SimplifyForWithUnreachablePattern, SimplifyBranchToUnreachablePattern,
      SimplifyCondBranchToUnreachablePattern>(context);
}

} // namespace mlir::iree_compiler::IREE::Util
