// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-insert-batch-dim-for-batchless-conv"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_INSERTBATCHDIMFORBATCHLESSCONVPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

/// Detects if the given linalg op is a batch-less convolution (a convolution
/// where the batch dimension N=1 was stripped by IREE's unit dim folding).
///
/// Returns true if this is a batch-less convolution that should be
/// transformed, false otherwise.
static bool isBatchlessConv(linalg::LinalgOp op) {
  // Must be a convolution operation (uses upstream MLIR's convolution
  // detection).
  if (!linalg::isaConvolutionOpInterface(op)) {
    return false;
  }

  FailureOr<linalg::ConvolutionDimensions> maybeConvDims =
      linalg::inferConvolutionDims(op);
  if (failed(maybeConvDims)) {
    return false;
  }

  // Detect operation kind based on channel dimensions.
  bool isRegularConv = !maybeConvDims->inputChannel.empty() &&
                       !maybeConvDims->outputChannel.empty();
  bool isDepthwise = !maybeConvDims->depth.empty();
  bool isPooling = maybeConvDims->inputChannel.empty() &&
                   maybeConvDims->outputChannel.empty() &&
                   maybeConvDims->depth.empty();

  // Check if batch dimension is missing.
  bool isBatchless = false;

  if (isRegularConv || isDepthwise) {
    // For conv/depthwise: batch should be non-empty when present.
    isBatchless = maybeConvDims->batch.empty();
  } else if (isPooling) {
    // For pooling: batch contains [N, C], without N it's just [C].
    // So batch.size() == 1 means only channel, no real batch.
    isBatchless = (maybeConvDims->batch.size() == 1);
  }

  return isBatchless;
}

/// Builds reassociation indices for prepending a dimension at position 0.
/// For a tensor of rank R, produces: [[0, 1], [2], [3], ..., [R]]
static SmallVector<ReassociationIndices>
buildPrependDimReassociation(int64_t rank) {
  SmallVector<ReassociationIndices> reassoc;
  reassoc.push_back({0, 1}); // New dim groups with first existing dim.
  for (int64_t i = 1; i < rank; ++i) {
    reassoc.push_back({i + 1});
  }
  return reassoc;
}

/// Expands a tensor by prepending a unit dimension at position 0.
/// tensor<AxBxC> -> tensor<1xAxBxC>
static Value prependUnitDimToTensor(RewriterBase &rewriter, Location loc,
                                    Value tensor) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  SmallVector<int64_t> newShape = {1};
  llvm::append_range(newShape, tensorType.getShape());
  auto newType = RankedTensorType::get(newShape, tensorType.getElementType());
  auto reassoc = buildPrependDimReassociation(tensorType.getRank());
  return tensor::ExpandShapeOp::create(rewriter, loc, newType, tensor, reassoc);
}

/// Shifts all existing dimensions in an affine map by 1 and prepends d0.
/// (d0, d1, ...) -> (...) becomes (d0, d1, d2, ...) -> (d0, ...)
static AffineMap shiftAndPrependDimToMap(AffineMap oldMap, MLIRContext *ctx) {
  AffineMap shifted = oldMap.shiftDims(1);
  SmallVector<AffineExpr> newResults;
  newResults.push_back(getAffineDimExpr(0, ctx)); // New leading dim.
  llvm::append_range(newResults, shifted.getResults());
  return AffineMap::get(oldMap.getNumDims() + 1, 0, newResults, ctx);
}

/// Inserts a unit batch dimension into a batchless convolution operation.
///
/// Transforms:
///   linalg.generic (batchless conv, e.g., HWC -> HWF)
/// Into:
///   expand_shape(input) -> linalg.generic (with batch, NHWC -> NHWF) ->
///   collapse_shape
///
/// Returns the newly created conv op with batch dimension.
static linalg::GenericOp insertUnitBatchDimension(RewriterBase &rewriter,
                                                  linalg::GenericOp op) {
  Location loc = op.getLoc();
  MLIRContext *ctx = rewriter.getContext();

  // Process all input operands.
  // - Operand 0 (image): expand tensor and prepend dim to map.
  // - Other operands (filter, zero points, etc.): keep as-is, shift map.
  SmallVector<Value> newInputs;
  SmallVector<AffineMap> newMaps;

  for (OpOperand *inputOperand : op.getDpsInputOperands()) {
    Value input = inputOperand->get();
    AffineMap oldMap = op.getMatchingIndexingMap(inputOperand);

    if (inputOperand->getOperandNumber() == 0) {
      // Image input: expand and prepend dim to map.
      newInputs.push_back(prependUnitDimToTensor(rewriter, loc, input));
      newMaps.push_back(shiftAndPrependDimToMap(oldMap, ctx));
    } else {
      // Other inputs (filter, zero points): keep as-is, just shift map.
      newInputs.push_back(input);
      newMaps.push_back(oldMap.shiftDims(1));
    }
  }

  // Output: expand and add batch dim to map.
  OpOperand *outputOperand = op.getDpsInitOperand(0);
  Value output = outputOperand->get();
  auto outputType = cast<RankedTensorType>(output.getType());
  Value expandedOutput = prependUnitDimToTensor(rewriter, loc, output);
  AffineMap newOutputMap =
      shiftAndPrependDimToMap(op.getMatchingIndexingMap(outputOperand), ctx);
  newMaps.push_back(newOutputMap);

  // New iterator types: prepend parallel (batch) to existing types.
  SmallVector<utils::IteratorType> newIterTypes;
  newIterTypes.push_back(utils::IteratorType::parallel);
  llvm::append_range(newIterTypes, op.getIteratorTypesArray());

  // Create new generic with batch dimension.
  auto newOutputType = cast<RankedTensorType>(expandedOutput.getType());
  auto newConvOp = linalg::GenericOp::create(
      rewriter, loc, TypeRange{newOutputType}, newInputs,
      ValueRange{expandedOutput}, newMaps, newIterTypes,
      [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
        IRMapping mapping;
        for (auto [oldArg, newArg] :
             llvm::zip(op.getBody()->getArguments(), args)) {
          mapping.map(oldArg, newArg);
        }
        for (Operation &bodyOp : op.getBody()->without_terminator()) {
          b.clone(bodyOp, mapping);
        }
        auto yield = cast<linalg::YieldOp>(op.getBody()->getTerminator());
        linalg::YieldOp::create(b, nestedLoc,
                                mapping.lookup(yield.getOperand(0)));
      });

  // Collapse result to remove the batch dimension we added.
  auto reassoc = buildPrependDimReassociation(outputType.getRank());
  auto collapsed = tensor::CollapseShapeOp::create(
      rewriter, loc, outputType, newConvOp.getResult(0), reassoc);

  rewriter.replaceOp(op, collapsed);
  return newConvOp;
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct InsertBatchDimForBatchlessConvPass final
    : impl::InsertBatchDimForBatchlessConvPassBase<
          InsertBatchDimForBatchlessConvPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();

    // Find the batchless conv op. We assume there is only one convolution-like
    // op per function (typical for dispatches).
    linalg::GenericOp batchlessConv = nullptr;
    getOperation()->walk([&](linalg::GenericOp op) {
      if (isBatchlessConv(op)) {
        if (batchlessConv) {
          // Multiple batchless convs found - bail out.
          batchlessConv = nullptr;
          return WalkResult::interrupt();
        }
        batchlessConv = op;
      }
      return WalkResult::advance();
    });

    if (!batchlessConv) {
      return;
    }

    // Insert unit batch dimension into the convolution.
    IRRewriter rewriter(context);
    rewriter.setInsertionPoint(batchlessConv);
    linalg::GenericOp newConvOp =
        insertUnitBatchDimension(rewriter, batchlessConv);

    // Phase 1: Bubble up expand_shape (only for ops BEFORE conv).
    {
      RewritePatternSet reshapePatterns(context);
      populateReshapeToInterfaceTensorPatterns(reshapePatterns);

      linalg::ControlFusionFn controlFn = [&](OpOperand *fusedOperand) {
        Operation *op = fusedOperand->getOwner();
        if (op->getBlock() != newConvOp->getBlock()) {
          return false;
        }
        return op->isBeforeInBlock(newConvOp);
      };

      linalg::populateFoldReshapeOpsByExpansionPatterns(reshapePatterns,
                                                        controlFn);
      tensor::populateFoldTensorEmptyPatterns(reshapePatterns);
      tensor::populateBubbleUpExpandShapePatterns(reshapePatterns);
      linalg::FillOp::getCanonicalizationPatterns(reshapePatterns, context);
      tensor::ExpandShapeOp::getCanonicalizationPatterns(reshapePatterns,
                                                         context);

      if (failed(applyPatternsGreedily(getOperation(),
                                       std::move(reshapePatterns)))) {
        return signalPassFailure();
      }
    }

    // Phase 2: Sink down collapse_shape (only for ops AFTER conv).
    {
      RewritePatternSet reshapePatterns(context);
      populateReshapeToInterfaceTensorPatterns(reshapePatterns);

      linalg::ControlFusionFn controlFn = [&](OpOperand *fusedOperand) {
        Operation *op = fusedOperand->getOwner();
        if (op->getBlock() != newConvOp->getBlock()) {
          return false;
        }
        return newConvOp->isBeforeInBlock(op);
      };

      linalg::populateFoldReshapeOpsByExpansionPatterns(reshapePatterns,
                                                        controlFn);
      tensor::populateFoldTensorEmptyPatterns(reshapePatterns);
      tensor::CollapseShapeOp::getCanonicalizationPatterns(reshapePatterns,
                                                           context);

      if (failed(applyPatternsGreedily(getOperation(),
                                       std::move(reshapePatterns)))) {
        return signalPassFailure();
      }
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
