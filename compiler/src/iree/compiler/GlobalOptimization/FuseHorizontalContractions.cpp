// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "iree/compiler/GlobalOptimization/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-global-opt-fuse-horizontal-contraction"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir::iree_compiler::GlobalOptimization {

static LogicalResult fuseGroup(RewriterBase &rewriter,
                               SmallVector<linalg::GenericOp> fusionGroup) {
  linalg::GenericOp base = fusionGroup[0];
  for (auto op : fusionGroup) {
    if (base->isBeforeInBlock(op)) {
      base = op;
    }
  }

  Location loc = base.getLoc();
  rewriter.setInsertionPointAfter(base);

  auto rhsType = cast<RankedTensorType>(base.getOperand(1).getType());
  auto outType = cast<RankedTensorType>(base.getResult(0).getType());

  SmallVector<ReassociationIndices> reassoc;
  reassoc.push_back({0, 1});
  for (int i = 0, e = rhsType.getRank() - 1; i < e; ++i) {
    reassoc.push_back({i + 2});
  }

  SmallVector<int64_t> rhsNewShape(rhsType.getShape());
  rhsNewShape.insert(rhsNewShape.begin(), 1);
  RankedTensorType concatRhsType =
      RankedTensorType::get(rhsNewShape, rhsType.getElementType());

  SmallVector<Value> rhsVals;
  for (auto op : fusionGroup) {
    Value thisRhs = op.getDpsInputOperand(1)->get();
    Value expanded = rewriter.create<tensor::ExpandShapeOp>(loc, concatRhsType,
                                                            thisRhs, reassoc);
    rhsVals.push_back(expanded);
  }

  Value newRhs = rewriter.create<tensor::ConcatOp>(loc, 0, rhsVals);

  SmallVector<int64_t> newShape(outType.getShape());
  newShape.insert(newShape.begin(), rhsVals.size());
  RankedTensorType concatOutType =
      RankedTensorType::get(newShape, outType.getElementType());

  Value baseOut = base.getDpsInitOperand(0)->get();
  auto origFill = baseOut.getDefiningOp<linalg::FillOp>();
  auto origEmpty =
      origFill.getDpsInitOperand(0)->get().getDefiningOp<tensor::EmptyOp>();

  auto newEmpty = rewriter.create<tensor::EmptyOp>(loc, concatOutType,
                                                   origEmpty.getDynamicSizes());
  auto newFill = rewriter.create<linalg::FillOp>(
      loc, concatOutType, origFill.getDpsInputOperand(0)->get(),
      newEmpty.getResult());

  Value lhs = base->getOperand(0);

  SmallVector<AffineMap> indexingMaps = base.getIndexingMapsArray();
  SmallVector<utils::IteratorType> iteratorTypes = base.getIteratorTypesArray();
  iteratorTypes.insert(iteratorTypes.begin(), utils::IteratorType::parallel);

  indexingMaps[0] = indexingMaps[0].shiftDims(1);
  indexingMaps[1] = indexingMaps[1].shiftDims(1).insertResult(
      rewriter.getAffineDimExpr(0), 0);
  indexingMaps[2] = indexingMaps[2].shiftDims(1).insertResult(
      rewriter.getAffineDimExpr(0), 0);

  linalg::GenericOp newGenericOp = rewriter.create<linalg::GenericOp>(
      loc, concatOutType, ValueRange{lhs, newRhs}, newFill.getResult(0),
      indexingMaps, iteratorTypes);
  rewriter.cloneRegionBefore(base.getRegion(), newGenericOp.getRegion(),
                             newGenericOp.getRegion().begin());

  Value fusedResult = newGenericOp.getResult(0);

  SmallVector<Value> newOuts;
  SmallVector<OpFoldResult> sizes = newEmpty.getMixedSizes();
  sizes[0] = rewriter.getIndexAttr(1);
  SmallVector<OpFoldResult> offsets(concatOutType.getRank(),
                                    rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(concatOutType.getRank(),
                                    rewriter.getIndexAttr(1));
  for (int i = 0, e = rhsVals.size(); i < e; ++i) {
    offsets[0] = rewriter.getIndexAttr(i);
    newOuts.push_back(rewriter.create<tensor::ExtractSliceOp>(
        loc, outType, fusedResult, offsets, sizes, strides));
  }

  for (auto [op, replacement] : llvm::zip_equal(fusionGroup, newOuts)) {
    rewriter.replaceOp(op, replacement);
  }
  return success();
}

namespace {

struct FuseHorizontalContractionsPass
    : public FuseHorizontalContractionsBase<FuseHorizontalContractionsPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, IREE::Flow::FlowDialect,
                    math::MathDialect>();
  }
  FuseHorizontalContractionsPass() {}
  FuseHorizontalContractionsPass(const FuseHorizontalContractionsPass &pass)
      : FuseHorizontalContractionsPass() {}

  void runOnOperation() override;
};

} // namespace

void FuseHorizontalContractionsPass::runOnOperation() {
  MLIRContext *context = &getContext();

  SmallVector<SmallVector<linalg::GenericOp>> horizontalFusionGroups;
  llvm::SmallDenseSet<linalg::GenericOp> groupedGenerics;

  getOperation()->walk([&](linalg::GenericOp generic) {
    if (groupedGenerics.contains(generic)) {
      return;
    }
    if (!linalg::isaContractionOpInterface(generic)) {
      return;
    }

    if (!generic->hasOneUse()) {
      return;
    }

    Value lhs = generic.getOperand(0);
    Value rhs = generic->getOperand(1);
    Value out = generic->getOperand(2);
    Type rhsType = rhs.getType();
    Type outType = generic->getResult(0).getType();
    Operation *user = *generic->user_begin();

    if (!isa<tensor::CollapseShapeOp>(user) &&
        !isa<tensor::ExpandShapeOp>(user)) {
      return;
    }

    Operation *rhsDef = rhs.getDefiningOp();
    if (!rhsDef) {
      return;
    }

    Block *block = generic->getBlock();

    SmallVector<linalg::GenericOp> fusionGroup;
    fusionGroup.push_back(generic);
    for (auto lhsUser : lhs.getUsers()) {
      if (lhsUser->getBlock() != block)
        continue;

      if (!lhsUser->isBeforeInBlock(user) || !rhsDef->isBeforeInBlock(lhsUser))
        continue;

      auto linalgUser = dyn_cast<linalg::GenericOp>(lhsUser);
      if (groupedGenerics.contains(linalgUser))
        continue;

      if (linalgUser == generic)
        continue;

      if (linalgUser->getOperand(2) != out)
        continue;

      if (!linalgUser || !linalg::isaContractionOpInterface(linalgUser))
        continue;

      if (linalgUser->getOperand(1).getType() != rhsType ||
          linalgUser->getResult(0).getType() != outType)
        continue;

      fusionGroup.push_back(linalgUser);
      groupedGenerics.insert(linalgUser);
    }

    if (fusionGroup.size() == 1)
      return;

    groupedGenerics.insert(generic);
    horizontalFusionGroups.push_back(fusionGroup);
  });

  IRRewriter rewriter(context);
  for (auto fusionGroup : horizontalFusionGroups) {
    if (failed(fuseGroup(rewriter, fusionGroup))) {
      return signalPassFailure();
    }
  }

  RewritePatternSet patterns(context);
  tensor::populateDecomposeTensorConcatPatterns(patterns);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createFuseHorizontalContractionsPass() {
  return std::make_unique<FuseHorizontalContractionsPass>();
}

} // namespace mlir::iree_compiler::GlobalOptimization
