// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/LinalgOpInfo.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmgpu-tensor-pad"

namespace mlir::iree_compiler {

namespace {

static FailureOr<SmallVector<int64_t>>
getPaddedShapeFromTensorLoad(IREE::Flow::DispatchTensorLoadOp tensorLoad,
                             ArrayRef<int64_t> origShape) {
  // Determine the padded shape from the load.
  SmallVector<int64_t> paddedShape(origShape.begin(), origShape.end());
  for (const auto &[index, size] :
       llvm::enumerate(tensorLoad.getMixedSizes())) {
    if (std::optional<int64_t> cst = getConstantIntValue(size)) {
      paddedShape[index] = cst.value();
    } else {
      FailureOr<int64_t> upperBound =
          ValueBoundsConstraintSet::computeConstantBound(
              presburger::BoundType::UB, size.get<Value>(),
              /*dim=*/std::nullopt,
              /*stopCondition=*/nullptr, /*closedUB=*/true);
      if (failed(upperBound))
        return failure();
      paddedShape[index] = *upperBound;
    }
  }
  return paddedShape;
}

static FailureOr<SmallVector<Value>>
rewriteAsPaddedOp(IRRewriter &rewriter, linalg::LinalgOp linalgOp,
                  linalg::LinalgOp &paddedOp) {
  Location loc = linalgOp.getLoc();

  IRRewriter::InsertionGuard g(rewriter);
  // Set IP after op because we also take the dims of the original output.
  rewriter.setInsertionPointAfter(linalgOp);

  // Pad each input operand in shared memory up to the targets bounding box
  // size. In this case, this corresponds with the maximum tile size from
  // distributing to workgroups.
  SmallVector<Value> paddedOperands;
  paddedOperands.reserve(linalgOp.getNumDpsInputs() +
                         linalgOp.getNumDpsInits());
  for (OpOperand &opOperand : linalgOp->getOpOperands()) {
    auto tensorLoad =
        opOperand.get().getDefiningOp<IREE::Flow::DispatchTensorLoadOp>();
    if (!tensorLoad) {
      return rewriter.notifyMatchFailure(linalgOp, "does not have tensor load");
    }
    FailureOr<SmallVector<int64_t>> maybePaddedShape =
        getPaddedShapeFromTensorLoad(tensorLoad, linalgOp.getShape(&opOperand));
    if (failed(maybePaddedShape))
      return failure();
    auto paddedShape = *maybePaddedShape;

    Value paddingValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(getElementTypeOrSelf(tensorLoad)));
    auto paddedTensorType =
        RankedTensorType::get(paddedShape, getElementTypeOrSelf(tensorLoad));
    Value paddedValue = linalg::makeComposedPadHighOp(
        rewriter, loc, paddedTensorType, tensorLoad, paddingValue,
        /*nofold=*/false);
    paddedOperands.push_back(paddedValue);
  }

  // Clone linalgOp to paddedOp with padded input/output shapes.
  auto resultTensorTypes = ValueRange(paddedOperands)
                               .take_back(linalgOp.getNumDpsInits())
                               .getTypes();
  paddedOp = mlir::clone(rewriter, linalgOp, resultTensorTypes, paddedOperands);

  // Slice out the original shape from the padded result to pass on to
  // consumers. The original linalg op is used to provide the dims for the reify
  // result shapes.
  SmallVector<SmallVector<OpFoldResult>> reifiedResultShapes;
  if (failed(cast<ReifyRankedShapedTypeOpInterface>(linalgOp.getOperation())
                 .reifyResultShapes(rewriter, reifiedResultShapes))) {
    return failure();
  }

  SmallVector<Value> paddedSubviewResults;
  paddedSubviewResults.reserve(paddedOp->getNumResults());
  for (const auto &[resultNumber, paddedResult] :
       llvm::enumerate(paddedOp->getResults())) {
    int64_t rank =
        llvm::cast<RankedTensorType>(paddedResult.getType()).getRank();
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes;
    for (OpFoldResult v : reifiedResultShapes[resultNumber])
      sizes.push_back(v);
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    paddedSubviewResults.push_back(rewriter.create<tensor::ExtractSliceOp>(
        loc, paddedResult, offsets, sizes, strides));
  }
  return paddedSubviewResults;
}

static FailureOr<Value> rewriteAsPaddedOp(IRRewriter &rewriter,
                                          tensor::UnPackOp op,
                                          tensor::UnPackOp &paddedOp) {
  Location loc = op.getLoc();

  // Set IP after op because we also take the dims of the original output.
  IRRewriter::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(op);
  auto tensorLoad =
      op.getDest().getDefiningOp<IREE::Flow::DispatchTensorLoadOp>();
  if (!tensorLoad) {
    return failure();
  }

  FailureOr<SmallVector<int64_t>> maybePaddedShape =
      getPaddedShapeFromTensorLoad(tensorLoad, op.getDestType().getShape());
  if (failed(maybePaddedShape))
    return failure();
  auto paddedShape = *maybePaddedShape;

  // Pad to the shape that makes tensor.unpack ops produce full tiles.
  SmallVector<int64_t> innerTiles = op.getStaticTiles();
  ArrayRef<int64_t> dimPos = op.getInnerDimsPos();
  for (auto [pos, size] : llvm::zip_equal(dimPos, innerTiles)) {
    paddedShape[pos] = llvm::divideCeil(paddedShape[pos], size) * size;
  }

  Value paddingValue = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(getElementTypeOrSelf(tensorLoad)));
  auto paddedTensorType =
      RankedTensorType::get(paddedShape, getElementTypeOrSelf(tensorLoad));
  Value paddedValue = linalg::makeComposedPadHighOp(
      rewriter, loc, paddedTensorType, tensorLoad, paddingValue,
      /*nofold=*/false);

  SmallVector<Value> paddedOperands = {op.getSource(), paddedValue};
  paddedOperands.append(op.getInnerTiles().begin(), op.getInnerTiles().end());
  paddedOp = rewriter.create<tensor::UnPackOp>(
      loc, TypeRange{paddedValue.getType()}, paddedOperands, op->getAttrs());

  // Slice out the original shape from the padded result to pass on to
  // consumers.
  SmallVector<SmallVector<OpFoldResult>> reifiedResultShapes;
  if (failed(op.reifyResultShapes(rewriter, reifiedResultShapes))) {
    return failure();
  }

  Value paddedSubviewResults;
  int64_t rank = paddedOp.getDestRank();
  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes = reifiedResultShapes[0];
  SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
  paddedSubviewResults = rewriter.create<tensor::ExtractSliceOp>(
      loc, paddedOp.getResult(), offsets, sizes, strides);
  return paddedSubviewResults;
}

static bool hasTwoOrThreeLoopsInfo(linalg::LinalgOp linalgOp) {
  return linalgOp.getNumParallelLoops() >= 2 &&
         linalgOp.getNumParallelLoops() <= 3;
}

struct LLVMGPUTensorPadPass
    : public LLVMGPUTensorPadBase<LLVMGPUTensorPadPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();

    IRRewriter rewriter(funcOp->getContext());
    rewriter.setInsertionPoint(funcOp);
    funcOp.walk([&](linalg::GenericOp linalgOp) {
      LinalgOpInfo opInfo(linalgOp, sharedMemTransposeFilter);
      // Checks preconditions for shared mem transpose. Only pad if op is
      // dynamic.
      if (!opInfo.isTranspose() || !opInfo.isDynamic() ||
          !hasTwoOrThreeLoopsInfo(linalgOp)) {
        funcOp.emitWarning("failed preconditions");
        return;
      }

      linalg::LinalgOp paddedOp;
      FailureOr<SmallVector<Value>> newResults =
          rewriteAsPaddedOp(rewriter, linalgOp, paddedOp);
      if (failed(newResults)) {
        return;
      }

      // Replace the original operation to pad.
      rewriter.replaceOp(linalgOp, *newResults);
    });

    funcOp.walk([&](tensor::UnPackOp unpackOp) {
      tensor::UnPackOp paddedOp;
      FailureOr<Value> newResult =
          rewriteAsPaddedOp(rewriter, unpackOp, paddedOp);
      if (failed(newResult)) {
        return;
      }

      // Replace the original operation to pad.
      rewriter.replaceOp(unpackOp, *newResult);
    });
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUTensorPadPass() {
  return std::make_unique<LLVMGPUTensorPadPass>();
}

} // namespace mlir::iree_compiler
