// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/TensorDynamicDimAnalysis.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-block-dynamic-dimensions"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_BLOCKDYNAMICDIMENSIONSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

using TensorDivisibilityInfo =
    llvm::SmallDenseMap<unsigned, IREE::Util::ConstantIntDivisibility>;

namespace {

struct RemoveOptimizationBarrier final
    : public OpRewritePattern<IREE::Util::OptimizationBarrierOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::Util::OptimizationBarrierOp barrierOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOp(barrierOp, barrierOp.getOperands());
    return success();
  }
};

/// This pass is used to materialize information about dynamic dimensions of
/// `tensor` operands of an operation in the IR. If a dynamic dimension is
/// known to be a multiple of a compile-time constant value, this pass
/// expands the shape of the operands. For example if a `tensor` operand
/// is of shape `tensor<...x?x...>` and that dimension is known to be a
/// multiple of 16, this operand is expanded to `tensor<...x?x16x...>` where the
/// size of the new dynamic dimension is 1/16-th the size of the original
/// dynamic dimension size. This is done in two steps.
/// 1) Replace operands with such dynamic dimension with the result of a
///    `tensor.expand_shape/tensor.collapse_shape` pair
///    to materialize the new static dimension and immediately fold it away. A
///    optimization barrier is added in between to prevent these operations from
///    being folded.
/// 2) Use patterns that propagate the `tensor.collapse_shape` down to
///    manipulate the operation appropriately. This
///    allows re-using the (fairly complex) logic used to expand dimensions of
///    operations implemented in the propagation patterns.
/// At the end of the pass the optimization barriers are removed to fold away
/// any un-propagated `tensor.expand_shape/tensor.collapse_shape` patterns.
struct BlockDynamicDimensionsPass final
    : impl::BlockDynamicDimensionsPassBase<BlockDynamicDimensionsPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

/// Retrieve the divisibility information for dynamic dimensions of `v` if
/// known.
static TensorDivisibilityInfo
getTensorDivisibilityInfo(const TensorDynamicDimAnalysis &dynamicDimAnalysis,
                          Value v) {
  TensorDivisibilityInfo divisibilityInfo;
  auto tensorType = dyn_cast<RankedTensorType>(v.getType());
  if (!tensorType) {
    return divisibilityInfo;
  }

  for (auto [index, dim] : llvm::enumerate(tensorType.getShape())) {
    if (!tensorType.isDynamicDim(index))
      continue;
    std::optional<IREE::Util::ConstantIntDivisibility> dimDivisibility =
        dynamicDimAnalysis.getDivisibilityInfo(v, index);
    if (!dimDivisibility)
      continue;
    divisibilityInfo[index] = std::move(dimDivisibility.value());
  }

  return divisibilityInfo;
}

/// For a `v` if the dimension is known to be multiple of a compile-time static
/// value, insert
///
/// ```mlir
/// %v_expand = tensor.expand_shape %v
/// %barrier = util.optimization.barrier %v
/// %v_collapse = tensor.collapse_shape %barrier
/// ```
///
/// where the generated `tensor.expand_shape` and `tensor.collapse_shape` are
/// inverses of each other. The `util.optimization.barrier` avoid these from
/// getting folded away during reshape propagation. Return the result of the
/// `tensor.collapse_shape generated.
static std::optional<Value>
blockDynamicDimensionsOfValue(RewriterBase &rewriter,
                              const TensorDivisibilityInfo &divisibilityInfo,
                              Value v) {
  auto tensorType = dyn_cast<RankedTensorType>(v.getType());
  if (!tensorType) {
    return std::nullopt;
  }

  // Check if we know that the operands have a divisibility information.
  SmallVector<OpFoldResult> outputShape;
  SmallVector<ReassociationIndices> reassociation;
  Location loc = v.getLoc();

  for (auto [index, dim] : llvm::enumerate(tensorType.getShape())) {
    reassociation.emplace_back(ReassociationIndices{});

    // Check if this needs division.
    if (!tensorType.isDynamicDim(index) || !divisibilityInfo.contains(index)) {
      reassociation.back().push_back(outputShape.size());
      outputShape.push_back(rewriter.getIndexAttr(dim));
      continue;
    }

    // Split the dynamic based on the divisibility info.
    IREE::Util::ConstantIntDivisibility currDivisibility =
        divisibilityInfo.lookup(index);
    uint64_t factor = currDivisibility.sdiv();
    AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
    AffineExpr divExpr = s0.floorDiv(factor);
    Value sourceDim = rewriter.create<tensor::DimOp>(loc, v, index).getResult();
    OpFoldResult newDynamicDim = affine::makeComposedFoldedAffineApply(
        rewriter, loc, divExpr, ArrayRef<OpFoldResult>{sourceDim});
    OpFoldResult newStaticDim = rewriter.getIndexAttr(factor);

    reassociation.back().push_back(outputShape.size());
    reassociation.back().push_back(outputShape.size() + 1);

    outputShape.push_back(newDynamicDim);
    outputShape.push_back(newStaticDim);
  }

  auto staticOutputShape =
      llvm::map_to_vector(outputShape, [](OpFoldResult ofr) {
        if (auto staticShapeAttr = dyn_cast<Attribute>(ofr)) {
          return cast<IntegerAttr>(staticShapeAttr).getInt();
        }
        return ShapedType::kDynamic;
      });
  auto outputType = RankedTensorType::get(
      staticOutputShape, tensorType.getElementType(), tensorType.getEncoding());

  Value expandShape = rewriter.create<tensor::ExpandShapeOp>(
      loc, outputType, v, reassociation, outputShape);
  Value barrier =
      rewriter.create<IREE::Util::OptimizationBarrierOp>(loc, expandShape)
          .getResult(0);
  Value collapseShape = rewriter.create<tensor::CollapseShapeOp>(
      loc, tensorType, barrier, reassociation);
  return collapseShape;
}

/// For an operation, replace the operands at indices specified in
/// `limitToOperandIndices` with the result of
/// `tensor.expand_shape`/`tensor.collapse_shape` pair to materialize the
/// information about dynamic dimensions that are known to be a multiple of a
/// compile-time static value. For example,
///
/// ```mlir
/// %1 = <some_op>(..., %0, ...) : ... , tensor<4x?x6xf32>
/// ```
///
/// If the dynamic dimension is known to be a multiple of 16, then generate
///
/// ```mlir
/// %expanded = tensor.expand_shape %0 :
///    tensor<4x?x5xf32> into tensor<4x?x16x6xf32>
/// %barrier = util.optimization.barrier %expanded
/// %collapsed = tensor.collapse_shape %barrier
///     : tensor<4x?x16x5xf32> into tensor<4x?x5xf32>
/// %1 = <some_op>(..., %collaped, ...) : ... , tensor<4x?x6xf32>
/// ```
static LogicalResult blockDynamicDimensions(
    RewriterBase &rewriter, const TensorDynamicDimAnalysis &dynamicDimAnalysis,
    Operation *operation, llvm::SmallDenseSet<int64_t> limitToOperandIndices) {
  OpBuilder::InsertionGuard g(rewriter);

  for (OpOperand &operand : operation->getOpOperands()) {
    if (!limitToOperandIndices.contains(operand.getOperandNumber()))
      continue;
    if (operand.get().getDefiningOp<tensor::CollapseShapeOp>())
      continue;
    TensorDivisibilityInfo operandDivisibilityInfo =
        getTensorDivisibilityInfo(dynamicDimAnalysis, operand.get());
    if (operandDivisibilityInfo.empty())
      continue;
    std::optional<Value> newOperand = blockDynamicDimensionsOfValue(
        rewriter, operandDivisibilityInfo, operand.get());
    if (newOperand) {
      rewriter.modifyOpInPlace(operation,
                               [&]() { operand.set(newOperand.value()); });
    }
  }
  return success();
}

/// Insert `tensor.expand_shape` operations to materialize in IR information
/// about dynamic dimensions that are known to be a multiple of a compile-time
/// know value, for the operands of `iree_linalg_ext.attention` operation.
static LogicalResult
blockDynamicDimensions(RewriterBase &rewriter,
                       const TensorDynamicDimAnalysis &dynamicDimAnalysis,
                       IREE::LinalgExt::AttentionOp attentionOp) {
  // Only block the q and k values.
  llvm::SmallDenseSet<int64_t> prunedOperandsList;
  prunedOperandsList.insert(attentionOp.getQueryMutable().getOperandNumber());
  prunedOperandsList.insert(attentionOp.getKeyMutable().getOperandNumber());
  return blockDynamicDimensions(rewriter, dynamicDimAnalysis, attentionOp,
                                prunedOperandsList);
}

/// Generic method to block dynamic dimensions for all tensor operands.
/// Only used for testing for now
static LogicalResult
blockDynamicDimensions(RewriterBase &rewriter,
                       const TensorDynamicDimAnalysis &dynamicDimAnalysis,
                       Operation *operation, bool test) {
  return TypeSwitch<Operation *, LogicalResult>(operation)
      .Case<IREE::LinalgExt::AttentionOp>([&](auto attentionOp) {
        return blockDynamicDimensions(rewriter, dynamicDimAnalysis,
                                      attentionOp);
      })
      .Default([&](Operation *op) {
        if (!test) {
          return success();
        }
        // The default path here is for now only for testing.
        llvm::SmallDenseSet<int64_t> tensorOperandsList;
        for (OpOperand &opOperand : operation->getOpOperands()) {
          if (isa<RankedTensorType>(opOperand.get().getType())) {
            tensorOperandsList.insert(opOperand.getOperandNumber());
          }
        }
        return blockDynamicDimensions(rewriter, dynamicDimAnalysis, operation,
                                      tensorOperandsList);
      });
}

void BlockDynamicDimensionsPass::runOnOperation() {
  Operation *operation = getOperation();
  MLIRContext *context = &getContext();
  TensorDynamicDimAnalysis dynamicDimAnalysis(operation);
  if (failed(dynamicDimAnalysis.run())) {
    return signalPassFailure();
  }

  IRRewriter rewriter(context);
  auto walkResult = operation->walk([&](Operation *op) -> WalkResult {
    rewriter.setInsertionPoint(op);
    return blockDynamicDimensions(rewriter, dynamicDimAnalysis, op, test);
  });
  if (walkResult.wasInterrupted()) {
    return signalPassFailure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "After blocking dimensions:\n";
    operation->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n";
  });

  {
    RewritePatternSet bubbleExpandShapePatterns(context);
    // Add patterns to "push down" the `tensor.collapse_shape` patterns (which
    // are the dual of the patterns to "bubble up" `tensor.expand_shape`
    // patterns)
    linalg::ControlFusionFn controlFn = [](OpOperand *) { return true; };
    linalg::populateFoldReshapeOpsByExpansionPatterns(bubbleExpandShapePatterns,
                                                      controlFn);
    IREE::LinalgExt::populateFoldReshapeOpsByExpansionPatterns(
        bubbleExpandShapePatterns, controlFn);
    // Add patterns to fold the "bubbled-up" `tensor.expand_shape` operation and
    // "pushed-down" `tensor.collapse_shape` operation with their interface
    // bindings or `tensor.empty` operations.
    populateReshapeToInterfaceTensorPatterns(bubbleExpandShapePatterns);
    tensor::populateFoldTensorEmptyPatterns(bubbleExpandShapePatterns);
    // Add some additional patterns that can simplify the IR and remove dead
    // operations.
    memref::populateResolveRankedShapedTypeResultDimsPatterns(
        bubbleExpandShapePatterns);
    populateRemoveDeadMemAllocPatterns(bubbleExpandShapePatterns);
    if (failed(applyPatternsAndFoldGreedily(
            operation, std::move(bubbleExpandShapePatterns)))) {
      operation->emitOpError(
          "failed in application of bubble up expand shape patterns");
      return signalPassFailure();
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "After reshape propagation:\n";
    operation->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n";
  });

  // Delete the optimization barrier and run some further cleanup.
  {
    RewritePatternSet removeBarrierOpsPatterns(context);
    removeBarrierOpsPatterns.insert<RemoveOptimizationBarrier>(context);
    tensor::ExpandShapeOp::getCanonicalizationPatterns(removeBarrierOpsPatterns,
                                                       context);
    tensor::CollapseShapeOp::getCanonicalizationPatterns(
        removeBarrierOpsPatterns, context);
    if (failed(applyPatternsAndFoldGreedily(
            operation, std::move(removeBarrierOpsPatterns)))) {
      operation->emitOpError("failed in cleanup patterns");
      return signalPassFailure();
    }
  }

  return;
}

} // namespace mlir::iree_compiler
