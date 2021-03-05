//===- LinalgTileToGenericPass.cpp - Tile and distribute to linalg.tile ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for tile and distribute to a tiled nested linalg
// abstraction.
//
//===----------------------------------------------------------------------===//

#include <iterator>
#include <memory>

#include "Transforms.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

#define DEBUG_TYPE "distribute"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

using namespace mlir;
using namespace mlir::linalg;

constexpr StringLiteral kTiledGenericOpName = "tiled_generic";
constexpr StringLiteral kTiledGenericYieldOpName = "tiled_generic_yield";
constexpr StringLiteral kTiledGenericPayloadOpName = "tiled_generic_payload";
constexpr StringLiteral kTiledGenericPayloadYieldOpName =
    "tiled_generic_payload_yield";

mlir::linalg::TileAndDistributePattern::TileAndDistributePattern(
    TileAndDistributeOptions options, LinalgTransformationFilter filter,
    PatternBenefit benefit)
    : RewritePattern(benefit, MatchAnyOpTypeTag()),
      filter(filter),
      options(options) {}

mlir::linalg::TileAndDistributePattern::TileAndDistributePattern(
    TileAndDistributeOptions options, StringRef opName, MLIRContext *context,
    LinalgTransformationFilter filter, PatternBenefit benefit)
    : RewritePattern(opName, {}, benefit, context),
      filter(filter),
      options(options) {}

LogicalResult mlir::linalg::TileAndDistributePattern::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter) const {
  LinalgOp linalgOp = dyn_cast<LinalgOp>(op);
  if (!linalgOp || !linalgOp.hasTensorSemantics()) return failure();
  if (failed(filter.checkAndNotify(rewriter, linalgOp))) return failure();

  Optional<TileAndDistributedLinalgOp> res =
      tileAndDistributeLinalgOp(rewriter, op, options);
  if (!res) return failure();
  if (res->tiledGenericOp->getNumResults() > 0)
    rewriter.replaceOp(op, res->tiledGenericOp->getResults());
  else
    rewriter.eraseOp(op);
  filter.replaceLinalgTransformationFilter(rewriter, res->tiledLinalgOp);
  return success();
}

static bool isProducedByOneOf(Value v,
                              llvm::SetVector<Operation *> &operations) {
  Operation *def = v.getDefiningOp();
  return def && operations.contains(def);
}

static bool hasAnyUseOutsideOf(Value v,
                               llvm::SetVector<Operation *> &operations) {
  return llvm::any_of(v.getUsers(),
                      [&](Operation *op) { return !operations.contains(op); });
}

static LinalgOp outline(PatternRewriter &rewriter, LinalgOp tiledRootOp,
                        llvm::SetVector<Operation *> &operations,
                        Operation *tiledGenericPayload) {
  LinalgOp tiledRootOpClone;

  Region &targetRegion = tiledGenericPayload->getRegion(0);
  Block *block = &targetRegion.front();

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(block);
  BlockAndValueMapping bvm;
  bvm.map(tiledGenericPayload->getOperands(), block->getArguments());
  llvm::SetVector<Operation *> blockOperations;
  for (Operation *op : operations) {
    Operation *cloned = rewriter.clone(*op, bvm);
    blockOperations.insert(cloned);
    if (op == tiledRootOp) tiledRootOpClone = cast<LinalgOp>(cloned);
  }

  llvm::SmallVector<Value> valuesToYield;  //(tiledRootOp->getResults());
  // Iterate over operations: if any result of any operation is used outside of
  // operations, it would leak outside of the region from blockOperations.
  // We need to yield such values outside the region.
  // TODO: we prob need to determine that set of values outside so we can do a
  // proper RAUW.
  for (auto it : llvm::zip(operations, blockOperations))
    for (OpResult result : std::get<0>(it)->getResults())
      if (hasAnyUseOutsideOf(result, operations))
        valuesToYield.push_back(
            std::get<1>(it)->getResult(result.getResultNumber()));

#if 1
  rewriter.create<linalg::YieldOp>(blockOperations.front()->getLoc(),
                                   valuesToYield);
#else
  OperationState state(tiledRootOpClone->getLoc(), kTiledGenericYieldOpName);
  state.addOperands(valuesToYield);
  state.addTypes(ValueRange{valuesToYield}.getTypes());
  state.addAttribute(
      "TODO",
      rewriter.getStringAttr(
          "should become linalg.yield_part %partial_tensor, %whole_tensor: "
          "(partial_tensor_t) -> (whole_tensor_t) where %whole_tensor must "
          "be "
          "`subtensor_insert %partial_tensor into ...`"));
#endif

  return tiledRootOpClone;
}

static TileAndDistributedLinalgOp buildTiledGenericPayload(
    PatternRewriter &rewriter, Operation *tiledGenericOp,
    LinalgOp tiledRootOp) {
  llvm::SetVector<Operation *> backwardSlice;
  // Get the backward slice limited by SubTensor ops and properly nested under
  // tiledGenericOp.
  getBackwardSlice(tiledRootOp, &backwardSlice, [&](Operation *op) {
    return !isa<SubTensorOp>(op) && tiledGenericOp->isProperAncestor(op);
  });
  backwardSlice.insert(tiledRootOp);

  // Compute used values defined outside of `operations` and use them to clone
  // in a new block.
  llvm::SetVector<Value> valuesFromOutside;
  for (Operation *op : backwardSlice)
    for (Value v : op->getOperands())
      if (!isProducedByOneOf(v, backwardSlice)) valuesFromOutside.insert(v);

  OperationState state(tiledRootOp->getLoc(), kTiledGenericPayloadOpName);
  state.addOperands(valuesFromOutside.getArrayRef());
  state.addTypes(tiledRootOp->getResultTypes());
  Region *region = state.addRegion();
  rewriter.createBlock(region, region->begin(),
                       ValueRange{valuesFromOutside.getArrayRef()}.getTypes());

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(tiledRootOp);
  Operation *tiledGenericPayload = rewriter.createOperation(state);
  LinalgOp tiledRootOpClone =
      outline(rewriter, tiledRootOp, backwardSlice, tiledGenericPayload);
  rewriter.replaceOp(tiledRootOp, tiledGenericPayload->getResults());

  // Erase the slice except tiledRootOp which was already replaced.
  backwardSlice.erase(std::prev(backwardSlice.end()));
  for (Operation *op : llvm::reverse(backwardSlice))
    if (op != tiledRootOp) rewriter.eraseOp(op);

  (void)simplifyRegions(tiledGenericPayload->getRegions());

  return TileAndDistributedLinalgOp{tiledGenericOp, tiledGenericPayload,
                                    tiledRootOpClone};
}

//
static void getUsedValuesDefinedOutsideOfLoopNest(
    llvm::SetVector<Operation *> loopNest,
    llvm::SetVector<Value> &valuesFromAbove) {
  scf::ForOp outerLoop = cast<scf::ForOp>(loopNest.front());
  scf::ForOp innerLoop = cast<scf::ForOp>(loopNest.back());
  innerLoop->walk([&](Operation *op) {
    for (OpOperand &operand : op->getOpOperands()) {
      // Values or BBArgs defined by an op outside of the loop nest.
      if (auto opResult = operand.get().dyn_cast<OpResult>()) {
        if (!outerLoop->isAncestor(opResult.getDefiningOp()))
          valuesFromAbove.insert(operand.get());
        continue;
      }
      if (auto bbArg = operand.get().dyn_cast<BlockArgument>()) {
        if (!outerLoop->isAncestor(bbArg.getOwner()->getParentOp()))
          valuesFromAbove.insert(operand.get());
        continue;
      }
      // Unexpected cases.
      llvm_unreachable("unexpected type of Value");
    }
  });
}

static void simplifyTensorGenericOpRegion(PatternRewriter &rewriter,
                                          Region &region,
                                          llvm::SetVector<int> keep) {
  Block *b = &region.front();
  Block *newBlock = rewriter.createBlock(&region, std::next(region.begin()));
  SmallVector<Value> argsRepl, newArgs;
  for (auto en : llvm::enumerate(b->getArguments())) {
    unsigned idx = en.index();
    BlockArgument bbarg = en.value();
    if (keep.contains(idx)) {
      newBlock->addArgument(bbarg.getType());
      argsRepl.push_back(newBlock->getArguments().back());
    } else {
      argsRepl.push_back(nullptr);
    }
  }
  rewriter.mergeBlocks(b, newBlock, argsRepl);
}

// TODO: use a map instead of linear scan when it matters.
template <typename ValueContainerType>
int lookupIndex(ValueContainerType &vector, Value target) {
  int pos = 0;
  for (Value v : vector) {
    if (target == v) return pos;
    ++pos;
  }
  return -1;
}

static void canonicalizeTensorGenericOp(PatternRewriter &rewriter,
                                        Operation *tiledGenericOp) {
  int64_t numLoops =
      tiledGenericOp->getAttr("num_loops_attr").cast<IntegerAttr>().getInt();
  unsigned numControlOperands = 3 * numLoops;
  unsigned numControlBlockArguments = numLoops;

  Block *block = &tiledGenericOp->getRegion(0).front();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(block);
  llvm::SetVector<Value> canonicalizedOperands;
  canonicalizedOperands.insert(
      tiledGenericOp->getOperands().begin(),
      tiledGenericOp->getOperands().begin() + numControlOperands);
  // Keep control bbArgs.
  llvm::SetVector<int> bbArgIdxToKeep;
  auto rangeNumControlBlockArguments =
      llvm::seq<int>(0, numControlBlockArguments);
  bbArgIdxToKeep.insert(rangeNumControlBlockArguments.begin(),
                        rangeNumControlBlockArguments.end());
  for (int idx = 0, e = tiledGenericOp->getNumOperands() - numControlOperands;
       idx < e; ++idx) {
    OpOperand &operand = tiledGenericOp->getOpOperand(numControlOperands + idx);
    BlockArgument bbArg = block->getArgument(numControlBlockArguments + idx);
    // Just drop bbargs without uses.
    if (bbArg.use_empty()) continue;

    if (canonicalizedOperands.contains(operand.get())) {
      int duplicateIdx = lookupIndex(canonicalizedOperands, operand.get());
      LLVM_DEBUG(DBGS() << "Duplicate: " << canonicalizedOperands[duplicateIdx]
                        << "\n");
      bbArg.replaceAllUsesWith(block->getArgument(
          duplicateIdx - numControlOperands + numControlBlockArguments));
      continue;
    }

    // Just pull constants in.
    if (Operation *constantOp = operand.get().getDefiningOp<ConstantOp>()) {
      LLVM_DEBUG(DBGS() << "Drop: " << *constantOp << "\n");
      bbArg.replaceAllUsesWith(rewriter.clone(*constantOp)->getResult(0));
      continue;
    }

    canonicalizedOperands.insert(operand.get());
    bbArgIdxToKeep.insert(numControlBlockArguments + idx);
  }

  simplifyTensorGenericOpRegion(rewriter, tiledGenericOp->getRegion(0),
                                bbArgIdxToKeep);
  tiledGenericOp->setOperands(canonicalizedOperands.getArrayRef());
}

static TileAndDistributedLinalgOp buildTiledGenericOp(
    PatternRewriter &rewriter, TiledLinalgOp &&tiledLinalgOp) {
  Location loc = tiledLinalgOp.op->getLoc();
  SmallVector<Value> lbs, ubs, steps, ivs;
  for (Operation *loop : tiledLinalgOp.loops) {
    scf::ForOp forOp = cast<scf::ForOp>(loop);
    lbs.push_back(forOp.lowerBound());
    ubs.push_back(forOp.upperBound());
    steps.push_back(forOp.step());
    ivs.push_back(forOp.getInductionVar());
  }

  auto outerLoop = cast<scf::ForOp>(tiledLinalgOp.loops.front());
  auto innerLoop = cast<scf::ForOp>(tiledLinalgOp.loops.back());
  llvm::SetVector<Value> valuesFromAbove;
  llvm::SetVector<Operation *> loopNest(tiledLinalgOp.loops.begin(),
                                        tiledLinalgOp.loops.end());
  getUsedValuesDefinedOutsideOfLoopNest(loopNest, valuesFromAbove);

  OperationState state(loc, kTiledGenericOpName);
  Region *region = state.addRegion();
  Block *block = new Block();
  region->push_back(block);

  // Results of TiledGenericOp comprise:
  //   1. the results of the outermost loop.
  state.addTypes(tiledLinalgOp.loops.front()->getResultTypes());

  // Operands of TiledGenericOp comprise:
  //   1. lbs/ubs/steps to reform loops.
  //   2. valuesFromAbove (TODO: filter out ivs and lbs/ubs/steps).
  state.addOperands(lbs);
  state.addOperands(ubs);
  state.addOperands(steps);
  // Assume that the outerLoop iter operands match the innerLoop bb iter args.
  // This is a property guaranteed by tileAndFuse on tensors.
  // In the future we may want to just directly emit TiledGenericOp to avoid
  // this assumption.
  state.addOperands(outerLoop.getIterOperands());
  state.addOperands(valuesFromAbove.getArrayRef());
  state.addAttribute("num_loops_attr", rewriter.getI32IntegerAttr(lbs.size()));

  // BBArgs of TiledGenericOp comprise:
  //   1. indices for each iterator (i.e. the IVs for all loops)
  //   2. the of the innermost loop.
  //   3. valuesFromAbove (TODO: filter out ivs and lbs/ubs/steps).
  SmallVector<Value> allValues;
  llvm::append_range(allValues, ivs);
  // Assume that the outerLoop iter operands match the innerLoop bb iter args.
  // This is a property guaranteed by tileAndFuse on tensors.
  // In the future we may want to just directly emit TiledGenericOp to avoid
  // this assumption.
  llvm::append_range(allValues, innerLoop.getRegionIterArgs());
  llvm::append_range(allValues, valuesFromAbove);
  block->addArguments(ValueRange{allValues}.getTypes());

  // TODO: handle ops in-between [inner, outer] loops (e.g. sink loop-invariants
  // and/or handle non-hyperrectangular cases).
  // In general, this will need an extra outlined function.
  // For best amortization, we will need one such function per dimension.
  // This is related to directly emitting TiledGenericOp to avoid this
  // assumption.

  // Propagate bbargs in the block before creating the TiledGeneric op.
  // We capture more than the innerLoop did and we cannot rely on the 1-1
  // replacement provided by `mergeBlocks`.
  for (auto it : llvm::zip(allValues, block->getArguments())) {
    std::get<0>(it).replaceUsesWithIf(std::get<1>(it), [&](OpOperand &operand) {
      return innerLoop->isProperAncestor(operand.getOwner());
    });
  }
  Block &innerLoopBlock = innerLoop->getRegion(0).front();
  assert(llvm::all_of(innerLoopBlock.getArguments(),
                      [](BlockArgument bbarg) { return bbarg.use_empty(); }));
  // Steal the ops and replace the loop nest by a new TileGenericOp.
  block->getOperations().splice(block->end(), innerLoopBlock.getOperations());

  Operation *tiledGenericOp = rewriter.createOperation(state);
  rewriter.replaceOp(tiledLinalgOp.loops.front(), tiledGenericOp->getResults());
  Operation *terminatorOp =
      tiledGenericOp->getRegion(0).front().getTerminator();

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(terminatorOp);
#if 1
  rewriter.replaceOpWithNewOp<linalg::YieldOp>(terminatorOp,
                                               terminatorOp->getOperands());
#else
  OperationState state(loc, kTiledGenericPayloadYieldOpName);
  state.addOperands(terminatorOp->getOperands());
  state.addTypes(terminatorOp->getResultTypes());
  state.addAttribute(
      "TODO",
      rewriter.getStringAttr(
          "should become linalg.yield_part %partial_tensor, %whole_tensor: "
          "(partial_tensor_t) -> (whole_tensor_t) where %whole_tensor must "
          "be "
          "`subtensor_insert %partial_tensor into ...`"));
#endif

  LLVM_DEBUG(DBGS() << "Pre-cleanup TiledGenericOp\n " << *tiledGenericOp);
  canonicalizeTensorGenericOp(rewriter, tiledGenericOp);
  LLVM_DEBUG(DBGS() << "Post-cleanup TiledGenericOp\n " << *tiledGenericOp);

  return buildTiledGenericPayload(rewriter, tiledGenericOp, tiledLinalgOp.op);
}

Optional<TileAndDistributedLinalgOp> mlir::linalg::tileAndDistributeLinalgOp(
    PatternRewriter &rewriter, LinalgOp linalgOp,
    const TileAndDistributeOptions &options) {
  auto tiledLinalgOp = tileLinalgOp(rewriter, linalgOp, options.tilingOptions);
  if (!tiledLinalgOp) return llvm::None;
  linalg::fuseProducerOfTensor(rewriter,
                               linalgOp.getOutputOpOperands()
                                   .front()
                                   .get()
                                   .getDefiningOp()
                                   ->getResults()
                                   .front(),
                               tiledLinalgOp->op.getOutputOpOperands().front());

  // Consider padding on the fly only if the op has tensor semantics.
  if (!options.tilingOptions.paddingValueComputationFunction ||
      !linalgOp.hasTensorSemantics())
    return buildTiledGenericOp(rewriter, std::move(*tiledLinalgOp));

  // Try to pad on the fly by rewriting tiledLinalgOp->op as a padded op.
  // TODO: This requires padding and bounding box to symbolic multiples.
  // (void)rewriteAsPaddedOp(rewriter, *tiledLinalgOp, options.tilingOptions);

  return buildTiledGenericOp(rewriter, std::move(*tiledLinalgOp));
}
