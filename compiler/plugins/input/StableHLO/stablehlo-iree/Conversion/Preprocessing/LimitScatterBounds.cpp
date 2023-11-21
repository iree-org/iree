// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements logic for lowering StableHLO einsum op to dot_general ops.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo-iree/Conversion/Preprocessing/Passes.h"
#include "stablehlo-iree/Conversion/Preprocessing/Rewriters.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_LIMITSCATTERBOUNDS
#include "stablehlo-iree/Conversion/Preprocessing/Passes.h.inc"

namespace {

LogicalResult replaceScatter(mlir::stablehlo::ScatterOp op) {
  IRRewriter builder(op.getContext());
  ImplicitLocOpBuilder b(op.getLoc(), builder);
  b.setInsertionPoint(op);

  auto dest = op.getInputs().front();
  auto indices = op.getScatterIndices();
  auto update0 = op.getUpdates().front();

  auto destTy = cast<ShapedType>(dest.getType());
  auto indicesTy = cast<ShapedType>(indices.getType());
  auto update0Ty = cast<ShapedType>(update0.getType());

  auto indicesETy = indicesTy.getElementType();
  auto indicesShape = indicesTy.getShape();

  auto dimnums = op.getScatterDimensionNumbers();
  const int64_t rank = destTy.getRank();

  int64_t indexVectorDim = dimnums.getIndexVectorDim();

  // Determine the bound limits for each index:
  auto updateWindowDims = dimnums.getUpdateWindowDims();
  auto toOperandDims = dimnums.getScatterDimsToOperandDims();
  auto insertedWindowDims = dimnums.getInsertedWindowDims();

  llvm::SmallVector<bool> isUpdatedWindowDim(rank, true);
  for (auto dim : insertedWindowDims) {
    isUpdatedWindowDim[dim] = false;
  }

  // This is grabbing all of the update shapes with indices removed:
  llvm::SmallVector<Value> updateWindowShape;
  for (auto dim : updateWindowDims) {
    updateWindowShape.push_back(b.create<tensor::DimOp>(update0, dim));
  }

  Value oneIdx = b.create<arith::ConstantIndexOp>(1);

  // We subtract the width of the update for each index. For inserted dimensions
  // this is 1.
  llvm::SmallVector<Value> windowBounds;
  {
    int j = 0;
    for (auto it : llvm::enumerate(isUpdatedWindowDim)) {
      Value dstSz = b.create<tensor::DimOp>(dest, it.index());
      if (!it.value()) {
        windowBounds.push_back(
            b.create<arith::SubIOp>(b.getIndexType(), dstSz, oneIdx));
        continue;
      }

      windowBounds.push_back(b.create<arith::SubIOp>(b.getIndexType(), dstSz,
                                                     updateWindowShape[j]));
      ++j;
    }
  }

  // Reorder / select the dimensions to the index ordering:
  llvm::SmallVector<Value> reorderedBounds;
  for (auto dim : toOperandDims) {
    reorderedBounds.push_back(
        b.create<arith::IndexCastOp>(indicesETy, windowBounds[dim]));
  }

  // Combine the selected dimensions into a small vector for the bounds check:
  Value dimLimits = b.create<tensor::FromElementsOp>(reorderedBounds);

  Value bounded = b.create<tensor::EmptyOp>(indicesShape, indicesETy);
  Value inbounds = b.create<tensor::EmptyOp>(indicesShape, b.getI1Type());
  AffineMap boundedMap = b.getMultiDimIdentityMap(indicesTy.getRank());

  llvm::SmallVector<AffineExpr> boundsExpr{
      b.getAffineDimExpr(indicesTy.getRank() - 1)};
  AffineMap boundsMap =
      AffineMap::get(indicesTy.getRank(), 0, boundsExpr, b.getContext());

  SmallVector<utils::IteratorType> indicesIter(indicesTy.getRank(),
                                               utils::IteratorType::parallel);
  auto applyLimits = b.create<linalg::GenericOp>(
      TypeRange{indicesTy, inbounds.getType()}, ValueRange{indices, dimLimits},
      ValueRange{bounded, inbounds},
      ArrayRef<AffineMap>{boundedMap, boundsMap, boundedMap, boundedMap},
      indicesIter, [](OpBuilder &bb, Location loc, ValueRange args) {
        ImplicitLocOpBuilder b(loc, bb);
        Value zero =
            b.create<arith::ConstantOp>(b.getZeroAttr(args[0].getType()));
        Value lower =
            b.create<arith::CmpIOp>(arith::CmpIPredicate::sge, args[0], zero);
        Value upper = b.create<arith::CmpIOp>(arith::CmpIPredicate::sle,
                                              args[0], args[1]);
        Value within = b.create<arith::AndIOp>(lower, upper);
        Value sel = b.create<arith::SelectOp>(within, args[0], zero);
        b.create<linalg::YieldOp>(ValueRange{sel, within});
      });

  bounded = applyLimits.getResult(0);
  inbounds = applyLimits.getResult(1);

  // If the index is not implicit we need to reduce away the index vector dim:
  if (indexVectorDim < indicesTy.getRank()) {
    ShapedType inTy = cast<ShapedType>(inbounds.getType());

    llvm::SmallVector<AffineExpr> inExprs;
    llvm::SmallVector<AffineExpr> outExprs;
    llvm::SmallVector<int64_t> outShape;
    for (int i = 0; i < inTy.getRank(); ++i) {
      if (i < indexVectorDim) {
        inExprs.push_back(b.getAffineDimExpr(outShape.size()));
        outExprs.push_back(b.getAffineDimExpr(outShape.size()));
        outShape.push_back(inTy.getDimSize(i));
      } else {
        // We need to do a reduction across the last dimension:
        inExprs.push_back(b.getAffineDimExpr(inTy.getRank() - 1));
      }
    }

    AffineMap inMap =
        AffineMap::get(inExprs.size(), 0, inExprs, b.getContext());
    AffineMap outMap =
        AffineMap::get(inExprs.size(), 0, outExprs, b.getContext());
    Value trueval = b.create<arith::ConstantIntOp>(1, b.getI1Type());
    Value empty = b.create<tensor::SplatOp>(inTy.clone(outShape), trueval);

    SmallVector<utils::IteratorType> reduceIter(outExprs.size(),
                                                utils::IteratorType::parallel);
    reduceIter.push_back(utils::IteratorType::reduction);

    inbounds =
        b.create<linalg::GenericOp>(
             TypeRange{empty.getType()}, ValueRange{inbounds},
             ValueRange{empty}, ArrayRef<AffineMap>{inMap, outMap}, reduceIter,
             [](OpBuilder &bb, Location loc, ValueRange args) {
               Value andOp = bb.create<arith::AndIOp>(loc, args[0], args[1]);
               bb.create<linalg::YieldOp>(loc, andOp);
             })
            .getResult(0);
  }

  if (cast<ShapedType>(inbounds.getType()).getRank() < update0Ty.getRank()) {
    llvm::SmallVector<bool> isBatchDim(update0Ty.getRank(), true);
    for (auto dim : updateWindowDims) {
      isBatchDim[dim] = false;
    }

    llvm::SmallVector<AffineExpr> inExprs;
    for (int i = 0; i < isBatchDim.size(); i++) {
      if (isBatchDim[i]) {
        inExprs.push_back(b.getAffineDimExpr(i));
      }
    }

    ShapedType outTy = update0Ty.clone(b.getI1Type());
    AffineMap inMap =
        AffineMap::get(outTy.getRank(), 0, inExprs, b.getContext());
    AffineMap outMap = b.getMultiDimIdentityMap(outTy.getRank());

    SmallVector<utils::IteratorType> iters(outTy.getRank(),
                                           utils::IteratorType::parallel);

    Value empty =
        b.create<tensor::EmptyOp>(outTy.getShape(), outTy.getElementType());
    inbounds =
        b.create<linalg::GenericOp>(
             TypeRange{empty.getType()}, ValueRange{inbounds},
             ValueRange{empty}, ArrayRef<AffineMap>{inMap, outMap}, iters,
             [](OpBuilder &bb, Location loc, ValueRange args) {
               bb.create<linalg::YieldOp>(loc, args[0]);
             })
            .getResult(0);
  }

  llvm::SmallVector<Value> newInputs(op.getInputs());

  ShapedType inputTy = cast<ShapedType>(newInputs.front().getType());
  Value emptyBool =
      b.create<tensor::EmptyOp>(inputTy.getShape(), b.getI1Type());
  newInputs.push_back(emptyBool);

  llvm::SmallVector<Value> newUpdates(op.getUpdates());
  newUpdates.push_back(inbounds);

  llvm::SmallVector<Type> newResultTys(op.getResultTypes());
  newResultTys.push_back(emptyBool.getType());

  auto scatter = b.create<mlir::stablehlo::ScatterOp>(
      newResultTys, newInputs, bounded, newUpdates, dimnums,
      op.getIndicesAreSorted(), op.getUniqueIndices());

  // Replace the existing yield with conditional results and the boolean update:
  for (int i = 0; i < op.getNumResults(); ++i) {
    op.getResult(i).replaceAllUsesWith(scatter.getResult(i));
  }

  // Clone the region into the new scatter and insert the new boolean udpates:
  Region &srcRegion = op.getUpdateComputation();
  Region &dstRegion = scatter.getUpdateComputation();

  IRMapping mapping;
  srcRegion.cloneInto(&dstRegion, mapping);

  const int64_t numUpdates = op.getUpdates().size();
  dstRegion.front().insertArgument(
      numUpdates * 2, RankedTensorType::get({}, b.getI1Type()), op.getLoc());
  dstRegion.front().insertArgument(
      numUpdates, RankedTensorType::get({}, b.getI1Type()), op.getLoc());

  // Update the return to only conditionally return the new value:
  Operation *terminator = dstRegion.front().getTerminator();
  b.setInsertionPoint(terminator);

  Value argInbounds = dstRegion.front().getArguments().back();
  llvm::SmallVector<Value> terminatorOperands;
  for (int i = 0; i < terminator->getNumOperands(); ++i) {
    auto newArg =
        b.create<arith::SelectOp>(argInbounds, terminator->getOperand(i),
                                  dstRegion.front().getArgument(i));

    terminatorOperands.push_back(newArg);
  }

  terminatorOperands.push_back(argInbounds);
  b.create<mlir::stablehlo::ReturnOp>(op.getLoc(), terminatorOperands);

  // Cleanup the no longer needed operations:
  terminator->erase();
  op.erase();

  return success();
}

struct LimitScatterBounds final
    : impl::LimitScatterBoundsBase<LimitScatterBounds> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, mlir::arith::ArithDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    auto operation = getOperation();
    llvm::SmallVector<mlir::stablehlo::ScatterOp> scatters;
    operation.walk([&scatters](mlir::stablehlo::ScatterOp scatter) {
      scatters.push_back(scatter);
    });

    for (auto scatter : scatters) {
      if (failed(replaceScatter(scatter))) {
        return signalPassFailure();
      }
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::stablehlo
