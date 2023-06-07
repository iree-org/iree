// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/Flow/Transforms/FormDispatchRegions.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-flow-collapse-dimensions"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {
/// Pass declaration.
struct CollapseDimensionsPass
    : public CollapseDimensionsBase<CollapseDimensionsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {}
  CollapseDimensionsPass() {}
  CollapseDimensionsPass(const CollapseDimensionsPass &pass)
      : CollapseDimensionsPass() {}
  void runOnOperation() override;
};
}  // namespace

/// Searches the same sequence in all the affine maps and collapses these
/// dimensions. It only applies these to "parallel" loops without mixing them
/// with "reduction" types.
static SmallVector<ReassociationIndices> getCollapsibleLoops(
    linalg::GenericOp genericOp) {
  SmallVector<ReassociationIndices> contiguousLoops;

  SmallVector<unsigned> pDims;
  genericOp.getParallelDims(pDims);
  if (pDims.size() < 2) return contiguousLoops;

  llvm::SmallDenseSet<unsigned> pLoops(pDims.begin(), pDims.end());

  auto hasAllMapsSameSequence = [&](AffineExpr preExpr, AffineExpr nextExpr) {
    for (AffineMap map : genericOp.getIndexingMapsArray()) {
      bool foundSeq = false;
      for (auto [index, resultExpr] : llvm::enumerate(map.getResults())) {
        if (resultExpr == nextExpr) {
          foundSeq = (index > 0 && preExpr == map.getResult(index - 1));
          break;
        }
      }
      if (!foundSeq) return false;
    }
    return true;
  };

  ReassociationIndices range;
  AffineExpr preExpr;
  for (auto nextExpr : genericOp.getIndexingMapsArray().front().getResults()) {
    unsigned pos = nextExpr.cast<AffineDimExpr>().getPosition();
    if (!range.empty()) {
      if (!hasAllMapsSameSequence(preExpr, nextExpr) || !pLoops.count(pos)) {
        if (range.size() > 1)
          contiguousLoops.push_back({range.begin(), range.end()});
        range.clear();
      }
    }
    preExpr = nextExpr;
    if (pLoops.count(pos)) range.push_back(pos);
  }
  if (range.size() > 1) contiguousLoops.push_back(range);

  LLVM_DEBUG({
    llvm::dbgs() << "Collapsing dimensions if possible: ";
    for (auto indices : contiguousLoops) {
      llvm::dbgs() << "[";
      for (auto idx : indices) llvm::dbgs() << idx << ",";
      llvm::dbgs() << "]\t";
    }
    llvm::dbgs() << "\n";
  });

  return contiguousLoops;
}

/// Collapse possible dimension of the given linalg.generic
static FailureOr<SmallVector<Value>> collapseLinalgGeneric(
    IRRewriter &rewriter, linalg::GenericOp genericOp,
    SmallVector<ReassociationIndices> &collapseIndices) {
  rewriter.setInsertionPoint(genericOp->getParentOp());
  FailureOr<SmallVector<Value>> replacements =
      mlir::linalg::collapseGenericOpIterationDims(genericOp, collapseIndices,
                                                   rewriter);
  if (failed(replacements) || replacements->empty()) {
    return rewriter.notifyMatchFailure(genericOp,
                                       "failed to collapse dimensions");
  }

  return replacements;
}

/// Returns true if the given op is collapsable.
static bool isEligibleForCollapse(linalg::GenericOp genericOp) {
  // TODO(guray) There is no mechanism to tell the collapsed indexes to
  // `tensor.expand_shape`. Once we have this support in MLIR, we can enable
  // dynamic tensor shapes.
  if (genericOp.hasDynamicShape()) return false;

  // TODO(guray) Currently we can only collapse when result of all the
  // AffineMaps are dimensions. Possible to collapse cases like
  // affine_map<d0, d1+d2> with affine_map<d0, d1+d2>, however, this is not
  // supported in collapsing mechanism in MLIR. Once we have this support,
  // we can remove this if statement.
  if (llvm::any_of(genericOp.getIndexingMapsArray(), [](AffineMap map) {
        return !map.isProjectedPermutation();
      })) {
    return false;
  }

  // TODO(guray) Collapsing caused performance regression in a cpu
  // benchmark, so we disable it.
  if (genericOp.hasIndexSemantics()) return false;

  return true;
}

/// Traverses all the the Ops in DispatchRegionOps and finds linalg.generic Op
/// without any producers.
static FailureOr<linalg::GenericOp> findRootGenericOp(
    DispatchRegionOp regionOp) {
  SmallVector<Operation *> computeOps;
  auto &ops = regionOp.getBody().front().getOperations();
  for (Operation &op : ops) {
    if (isa<TilingInterface>(op)) computeOps.push_back(&op);
  }
  // Looking for root without producer
  if (computeOps.size() != 1 || ops.size() != 2) return failure();
  auto genericOp = llvm::dyn_cast<linalg::GenericOp>(computeOps.front());
  if (!genericOp) return failure();
  return genericOp;
}

/// Generate a new dispatch.region and workload according with the collapsed
/// linalg Generic Op
static LogicalResult generateNewDispatchRegion(
    IRRewriter &rewriter, DispatchRegionOp regionOp,
    SmallVector<Value> collapseResults, linalg::GenericOp newGenericOp) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(regionOp->getParentOp());

  auto maybeRegionOp = Flow::wrapOpInDispatchRegion(rewriter, newGenericOp);
  if (failed(maybeRegionOp)) return failure();

  // Replace old regionOp with the result of collapse
  rewriter.replaceOp(regionOp, collapseResults);

  return success();
}

/// Traverses DispatchRegionOps to find linalg genericOps that has no
/// producers and tries to collapse its dimensions.
static LogicalResult collapseDimensions(IRRewriter &rewriter,
                                        DispatchRegionOp &regionOp) {
  // Step 1. Find the root linalg.generic Op with no producer
  std::optional<linalg::GenericOp> genericOp = findRootGenericOp(regionOp);
  if (!genericOp.has_value()) return success();

  // Step 2. Check whether it is possible to collapse
  if (!isEligibleForCollapse(genericOp.value())) return success();
  SmallVector<ReassociationIndices> collapseIndices;
  collapseIndices = getCollapsibleLoops(genericOp.value());
  if (collapseIndices.empty()) return success();

  // Step 3. Collapse dimensions
  auto maybeReplacements =
      collapseLinalgGeneric(rewriter, genericOp.value(), collapseIndices);
  if (failed(maybeReplacements)) return failure();
  auto expandshapeOp =
      maybeReplacements->front().getDefiningOp<tensor::ExpandShapeOp>();
  if (!expandshapeOp) return failure();
  auto newGenericOp =
      expandshapeOp.getOperand().getDefiningOp<linalg::GenericOp>();
  if (!newGenericOp) return failure();

  // Step 4. Generate new dispatch region and replace old one users
  if (failed(generateNewDispatchRegion(rewriter, regionOp, *maybeReplacements,
                                       newGenericOp)))
    return failure();

  return success();
}

void CollapseDimensionsPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();
  IRRewriter rewriter(funcOp->getContext());

  auto walkResult = funcOp->walk([&](DispatchRegionOp regionOp) {
    if (failed(collapseDimensions(rewriter, regionOp)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    funcOp->emitOpError("failed in collapsing dimensions pass");
    return signalPassFailure();
  }

  RewritePatternSet canonicalizationPatterns(&getContext());
  memref::populateResolveRankedShapedTypeResultDimsPatterns(
      canonicalizationPatterns);
  tensor::populateFoldTensorEmptyPatterns(canonicalizationPatterns);
  if (failed(applyPatternsAndFoldGreedily(
          funcOp, std::move(canonicalizationPatterns)))) {
    funcOp->emitOpError("failed to apply cleanup patterns");
    return signalPassFailure();
  }
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createCollapseDimensionsPass() {
  return std::make_unique<CollapseDimensionsPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
