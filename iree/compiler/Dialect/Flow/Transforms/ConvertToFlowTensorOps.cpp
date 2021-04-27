// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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
/// eventually map to DMA operations if
/// - all offsets apart from the first one are 0
/// - all the sizes apart from the first match the sizes of the source
/// - all strides are 1.
static bool isOffsetSizeAndStrideMappableToFlow(ArrayRef<OpFoldResult> offsets,
                                                ArrayRef<OpFoldResult> sizes,
                                                ArrayRef<OpFoldResult> strides,
                                                ArrayRef<int64_t> baseShape) {
  if (offsets.size() != baseShape.size()) {
    // Unhanded rank-reducing case.
    return false;
  }
  auto matchVal = [](OpFoldResult valueOrAttr, int64_t val) -> bool {
    auto attr = valueOrAttr.dyn_cast<Attribute>();
    return attr && attr.cast<IntegerAttr>().getInt() == val;
  };
  for (auto dim : llvm::seq<unsigned>(0, offsets.size())) {
    if ((dim != 0 && (!matchVal(offsets[dim], 0) ||
                      !matchVal(sizes[dim], baseShape[dim]))) ||
        !matchVal(strides[dim], 1)) {
      return false;
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

/// Generates `memref.dim` operations to get the dynamic sizes of a value `v`.
static SmallVector<Value, 4> getDynamicDimValues(OpBuilder &b, Location loc,
                                                 Value v) {
  SmallVector<Value, 4> dynamicDims;
  for (auto dim : llvm::enumerate(v.getType().cast<ShapedType>().getShape())) {
    if (dim.value() != ShapedType::kDynamicSize) continue;
    dynamicDims.push_back(b.createOrFold<memref::DimOp>(loc, v, dim.index()));
  }
  return dynamicDims;
}

namespace {

/// Converts linalg.tensor_reshape operations into flow.tensor.reshape
/// operations.
struct LinalgTensorReshapeToFlowTensorReshape
    : public OpRewritePattern<linalg::TensorReshapeOp> {
  using OpRewritePattern<linalg::TensorReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TensorReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    if (reshapeOp->getParentOfType<Flow::DispatchWorkgroupsOp>()) {
      return failure();
    }
    SmallVector<SmallVector<Value>> outputShape;
    if (failed(reshapeOp.reifyReturnTypeShapesPerResultDim(rewriter,
                                                           outputShape))) {
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
    : public OpRewritePattern<SubTensorInsertOp> {
  using OpRewritePattern<SubTensorInsertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubTensorInsertOp insertOp,
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
      // Pad the leading dimensions by 1. By construction, only leading sizes of
      // the subtensor can be 1 at this stage (and therefore can be
      // rank-reduced).
      SmallVector<int64_t> unreducedShape(
          destType.getRank() - sourceType.getRank(), 1);
      unreducedShape.append(sourceType.getShape().begin(),
                            sourceType.getShape().end());
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
struct SubTensorToTensorSlice : public OpRewritePattern<SubTensorOp> {
  using OpRewritePattern<SubTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubTensorOp subTensorOp,
                                PatternRewriter &rewriter) const override {
    if (subTensorOp->getParentOfType<Flow::DispatchWorkgroupsOp>()) {
      return failure();
    }
    SmallVector<OpFoldResult, 4> offsets = subTensorOp.getMixedOffsets();
    SmallVector<OpFoldResult, 4> sizes = subTensorOp.getMixedSizes();
    SmallVector<OpFoldResult, 4> strides = subTensorOp.getMixedStrides();
    ArrayRef<int64_t> srcShape = subTensorOp.getSourceType().getShape();
    if (!isOffsetSizeAndStrideMappableToFlow(offsets, sizes, strides,
                                             srcShape)) {
      return failure();
    }
    Location loc = subTensorOp.getLoc();

    ShapedType sourceType = subTensorOp.getSourceType();
    ShapedType resultType = subTensorOp.getType();

    // Handle rank reduced version.
    if (resultType.getRank() < sourceType.getRank()) {
      // Pad the leading dimensions by 1. By construction, only leading sizes of
      // the subtensor can be 1 at this stage (and therefore can be
      // rank-reduced).
      SmallVector<int64_t> unreducedShape(
          sourceType.getRank() - resultType.getRank(), 1);
      unreducedShape.append(resultType.getShape().begin(),
                            resultType.getShape().end());
      resultType =
          RankedTensorType::get(unreducedShape, sourceType.getElementType());
    }

    auto offsetVals = getAsValues(rewriter, loc, offsets);
    auto sizeVals = getAsValues(rewriter, loc, sizes);
    auto sourceDynamicDims =
        getDynamicDimValues(rewriter, loc, subTensorOp.source());
    auto resultDynamicDims = getDynamicValues(sizes);
    Value replacement = rewriter.create<TensorSliceOp>(
        loc, resultType, subTensorOp.source(), sourceDynamicDims, offsetVals,
        sizeVals, resultDynamicDims);
    if (resultType.getRank() > subTensorOp.getType().getRank()) {
      replacement = rewriter.create<IREE::Flow::TensorReshapeOp>(
          loc, subTensorOp.getType(), replacement, resultDynamicDims,
          resultDynamicDims);
    }
    rewriter.replaceOp(subTensorOp, replacement);
    return success();
  }
};

/// Converts operations that can map to flow.tensor.* operations.
struct ConvertToFlowTensorOpsPass
    : public PassWrapper<ConvertToFlowTensorOpsPass, OperationPass<FuncOp>> {
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
    patterns.insert<LinalgTensorReshapeToFlowTensorReshape,
                    SubTensorInsertToTensorUpdate, SubTensorToTensorSlice>(
        context);
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

static PassRegistration<ConvertToFlowTensorOpsPass> pass(
    "iree-flow-convert-to-flow-tensor-ops-pass",
    "Convert operations to equivalent flow.tensor.* operations",
    [] { return std::make_unique<ConvertToFlowTensorOpsPass>(); });

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
