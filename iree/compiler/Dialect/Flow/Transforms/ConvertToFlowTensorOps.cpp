// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-flow-convert-to-flow-tensor-ops"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

/// An operation that uses `offsets`, `sizes` and `strides` (i.e. implements the
/// `OffsetSizeAndStrideInterface`) can be mapped to flow operations that
/// eventually map to DMA operations if the offsets/sizes/strides represent a
/// contiguous memory.
static bool isOffsetSizeAndStrideMappableToFlow(ArrayRef<OpFoldResult> offsets,
                                                ArrayRef<OpFoldResult> sizes,
                                                ArrayRef<OpFoldResult> strides,
                                                ArrayRef<int64_t> baseShape) {
  if (offsets.size() != baseShape.size()) {
    // Unhanded rank-reducing case.
    return false;
  }
  auto getVal = [](OpFoldResult valueOrAttr, int64_t dynamicVal) -> int64_t {
    auto attr = valueOrAttr.dyn_cast<Attribute>();
    return attr ? attr.cast<IntegerAttr>().getInt() : dynamicVal;
  };
  /// To ensure contiguity, start from the least signficant dimension. As long
  /// as the inner slices are "full slices", the current slice can be any offset
  /// and size. If the inner slices are not "full slices", the current slice
  /// must be of size 1. All strides must be one.

  bool fullSlices = true;
  for (size_t dim = offsets.size(); dim > 0; dim--) {
    int64_t staticOffset =
        getVal(offsets[dim - 1], ShapedType::kDynamicStrideOrOffset);
    int64_t staticSize = getVal(sizes[dim - 1], ShapedType::kDynamicSize);
    int64_t staticStride =
        getVal(strides[dim - 1], ShapedType::kDynamicStrideOrOffset);

    if (staticStride != 1) return false;
    // The offsets and sizes dont have to be static for all dimensions. When
    // `fullSlices` is true, the offset and sizes can be dynamic. But many
    // cases, the dynamic offset/size value is obtained by computing from
    // another tensor which lives on the device. To avoid host-round tripping
    // enforce that offset/size is also static.
    if (staticSize == ShapedType::kDynamicSize) return false;
    if (staticOffset == ShapedType::kDynamicStrideOrOffset) return false;

    if (fullSlices == false) {
      if (staticSize != 1) return false;
    } else {
      if (!(staticOffset == 0 && staticSize != ShapedType::kDynamicSize &&
            baseShape[dim - 1] != ShapedType::kDynamicSize &&
            staticSize == baseShape[dim - 1]))
        fullSlices = false;
    }
  }
  return true;
}

/// Returns the `Value`s for a list of `OpFoldResult` by generating std.constant
/// ops for the static values.
static SmallVector<Value, 4> getAsValues(
    OpBuilder &b, Location loc, ArrayRef<OpFoldResult> valueOrAttrList) {
  SmallVector<Value, 4> values;
  for (auto valueOrAttr : valueOrAttrList) {
    if (auto attr = valueOrAttr.dyn_cast<Attribute>()) {
      values.push_back(
          b.create<ConstantIndexOp>(loc, attr.cast<IntegerAttr>().getInt()));
    } else {
      values.push_back(valueOrAttr.get<Value>());
    }
  }
  return values;
}

/// Gets the list of non-static values from a list of `OpFoldResult`.
static SmallVector<Value, 4> getDynamicValues(
    ArrayRef<OpFoldResult> valueOrAttrList) {
  SmallVector<Value, 4> dynamicDims;
  for (auto valueOrAttr : valueOrAttrList) {
    if (auto value = valueOrAttr.dyn_cast<Value>()) {
      dynamicDims.push_back(value);
    }
  }
  return dynamicDims;
}

/// Get shape of the tensor given the sizes as a list of `OpFoldResult`.
static SmallVector<int64_t, 4> getShapeFromSizes(
    ArrayRef<OpFoldResult> valueOrAttrList) {
  return llvm::to_vector<4>(llvm::map_range(
      valueOrAttrList, [&](OpFoldResult valueOrAttr) -> int64_t {
        if (auto attr = valueOrAttr.dyn_cast<Attribute>()) {
          return attr.cast<IntegerAttr>().getInt();
        }
        return ShapedType::kDynamicSize;
      }));
}

/// Generates `memref.dim` operations to get the dynamic sizes of a value `v`.
static SmallVector<Value, 4> getDynamicDimValues(OpBuilder &b, Location loc,
                                                 Value v) {
  SmallVector<Value, 4> dynamicDims;
  for (auto dim : llvm::enumerate(v.getType().cast<ShapedType>().getShape())) {
    if (dim.value() != ShapedType::kDynamicSize) continue;
    dynamicDims.push_back(b.createOrFold<tensor::DimOp>(loc, v, dim.index()));
  }
  return dynamicDims;
}

namespace {

/// Converts linalg.tensor_reshape operations into flow.tensor.reshape
/// operations.
template <typename TensorReshapeOp>
struct LinalgTensorReshapeToFlowTensorReshape
    : public OpRewritePattern<TensorReshapeOp> {
  using OpRewritePattern<TensorReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TensorReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    if (reshapeOp->template getParentOfType<Flow::DispatchWorkgroupsOp>()) {
      return failure();
    }
    SmallVector<SmallVector<Value>> outputShape;
    if (failed(reshapeOp.reifyResultShapes(rewriter, outputShape))) {
      return failure();
    }
    SmallVector<Value> outputDynamicShapes;
    for (auto shape :
         llvm::zip(reshapeOp.getResultType().getShape(), outputShape[0])) {
      if (std::get<0>(shape) != ShapedType::kDynamicSize) continue;
      outputDynamicShapes.push_back(std::get<1>(shape));
    }
    rewriter.replaceOpWithNewOp<IREE::Flow::TensorReshapeOp>(
        reshapeOp, reshapeOp.getResultType(), reshapeOp.src(),
        outputDynamicShapes);
    return success();
  }
};

/// Convert subtensor insert operation flow.tensor.update where possible.
struct SubTensorInsertToTensorUpdate
    : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertOp,
                                PatternRewriter &rewriter) const override {
    if (insertOp->getParentOfType<Flow::DispatchWorkgroupsOp>()) {
      return failure();
    }
    SmallVector<OpFoldResult, 4> offsets = insertOp.getMixedOffsets();
    SmallVector<OpFoldResult, 4> sizes = insertOp.getMixedSizes();
    SmallVector<OpFoldResult, 4> strides = insertOp.getMixedStrides();
    ArrayRef<int64_t> dstShape = insertOp.getType().getShape();
    if (!isOffsetSizeAndStrideMappableToFlow(offsets, sizes, strides,
                                             dstShape)) {
      return failure();
    }
    Location loc = insertOp.getLoc();
    auto sourceDynamicDims = getDynamicValues(sizes);
    Value source = insertOp.source();
    ShapedType sourceType = insertOp.getSourceType();
    ShapedType destType = insertOp.getType();

    // Handle rank-reduced version.
    if (sourceType.getRank() < destType.getRank()) {
      // Get the un-rank-reduced shape of the source.
      auto unreducedShape = getShapeFromSizes(sizes);
      sourceType =
          RankedTensorType::get(unreducedShape, sourceType.getElementType());
      source = rewriter.create<IREE::Flow::TensorReshapeOp>(
          loc, sourceType, source, sourceDynamicDims, sourceDynamicDims);
    }

    auto offsetVals = getAsValues(rewriter, loc, offsets);
    Value dest = insertOp.dest();
    auto destDynamicDims = getDynamicDimValues(rewriter, loc, dest);
    rewriter.replaceOpWithNewOp<TensorUpdateOp>(
        insertOp, insertOp.getType(), dest, destDynamicDims, offsetVals, source,
        sourceDynamicDims, nullptr);
    return success();
  }
};

/// Convert subtensor operation to flow.tensor.slice where possible.
struct SubTensorToTensorSlice
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    if (sliceOp->getParentOfType<Flow::DispatchWorkgroupsOp>()) {
      return failure();
    }
    SmallVector<OpFoldResult, 4> offsets = sliceOp.getMixedOffsets();
    SmallVector<OpFoldResult, 4> sizes = sliceOp.getMixedSizes();
    SmallVector<OpFoldResult, 4> strides = sliceOp.getMixedStrides();
    ArrayRef<int64_t> srcShape = sliceOp.getSourceType().getShape();
    if (!isOffsetSizeAndStrideMappableToFlow(offsets, sizes, strides,
                                             srcShape)) {
      return failure();
    }
    Location loc = sliceOp.getLoc();

    ShapedType sourceType = sliceOp.getSourceType();
    ShapedType resultType = sliceOp.getType();

    // Handle rank reduced version.
    if (resultType.getRank() < sourceType.getRank()) {
      // Get the un-rank-reduced shape of the result.
      auto unreducedShape = getShapeFromSizes(sizes);
      resultType =
          RankedTensorType::get(unreducedShape, sourceType.getElementType());
    }

    auto offsetVals = getAsValues(rewriter, loc, offsets);
    auto sizeVals = getAsValues(rewriter, loc, sizes);
    auto sourceDynamicDims =
        getDynamicDimValues(rewriter, loc, sliceOp.source());
    auto resultDynamicDims = getDynamicValues(sizes);
    Value replacement = rewriter.create<TensorSliceOp>(
        loc, resultType, sliceOp.source(), sourceDynamicDims, offsetVals,
        sizeVals, resultDynamicDims);
    if (resultType.getRank() > sliceOp.getType().getRank()) {
      replacement = rewriter.create<IREE::Flow::TensorReshapeOp>(
          loc, sliceOp.getType(), replacement, resultDynamicDims,
          resultDynamicDims);
    }
    rewriter.replaceOp(sliceOp, replacement);
    return success();
  }
};

/// Converts operations that can map to flow.tensor.* operations.
struct ConvertToFlowTensorOpsPass
    : public ConvertToFlowTensorOpsBase<ConvertToFlowTensorOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Flow::FlowDialect, memref::MemRefDialect,
                    mlir::StandardOpsDialect>();
  }
  ConvertToFlowTensorOpsPass() = default;
  ConvertToFlowTensorOpsPass(const ConvertToFlowTensorOpsPass &pass) {}
  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    MLIRContext *context = funcOp->getContext();
    context->allowUnregisteredDialects(true);
    RewritePatternSet patterns(&getContext());
    patterns.insert<
        LinalgTensorReshapeToFlowTensorReshape<linalg::TensorCollapseShapeOp>,
        LinalgTensorReshapeToFlowTensorReshape<linalg::TensorExpandShapeOp>,
        SubTensorInsertToTensorUpdate, SubTensorToTensorSlice>(context);
    IREE::Flow::TensorReshapeOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createConvertToFlowTensorOpsPass() {
  return std::make_unique<ConvertToFlowTensorOpsPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
