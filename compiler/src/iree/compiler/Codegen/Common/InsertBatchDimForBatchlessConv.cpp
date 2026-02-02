// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// InsertBatchDimForBatchlessConv Pass
//===----------------------------------------------------------------------===//
//
// This pass detects 2D convolution operations that have been generalized and
// had their batch dimension (N=1) stripped by IREE codegen transformations.
// It restores the batch dimension by inserting tensor.expand_shape and
// tensor.collapse_shape operations, enabling upstream convolution-specific
// patterns (like DownscaleConv and vectorization) to match and apply.
//
// Note: This pass currently only handles 2D convolutions (conv_2d, pooling_2d,
// depthwise_conv_2d).
//
// The reshape operations are expected to be folded into dispatch tensor
// load/store operations by the PropagateReshapesByExpansion pass, resulting
// in zero runtime cost.
//
//===----------------------------------------------------------------------===//

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

/// Detects if the given linalg op is a batch-less 2D convolution (a convolution
/// where the batch dimension N=1 was stripped by IREE's unit dim folding).
///
/// Returns success if this is a batch-less 2D convolution that should be
/// transformed, failure otherwise.
static LogicalResult isBatchlessConv(linalg::LinalgOp op) {
  // Must be a convolution operation (uses upstream MLIR's convolution
  // detection)
  if (!linalg::isaConvolutionOpInterface(op)) {
    return failure();
  }

  auto dimsOr = linalg::inferConvolutionDims(op);
  if (failed(dimsOr)) {
    return failure();
  }
  linalg::ConvolutionDimensions dims = *dimsOr;

  // Must be 2D spatial convolution
  if (dims.outputImage.size() != 2 || dims.filterLoop.size() != 2) {
    return failure();
  }

  // Detect operation kind based on channel dimensions
  bool isRegularConv =
      !dims.inputChannel.empty() && !dims.outputChannel.empty();
  bool isDepthwise = !dims.depth.empty();
  bool isPooling = dims.inputChannel.empty() && dims.outputChannel.empty() &&
                   dims.depth.empty();

  // Check if batch dimension is missing
  bool isBatchless = false;

  if (isRegularConv || isDepthwise) {
    // For conv/depthwise: batch should be non-empty when present
    isBatchless = dims.batch.empty();
  } else if (isPooling) {
    // For pooling: batch contains [N, C], without N it's just [C]
    // So batch.size() == 1 means only channel, no real batch
    isBatchless = (dims.batch.size() == 1);
  }

  if (!isBatchless) {
    return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Helper functions for batch dimension insertion
//===----------------------------------------------------------------------===//

/// Builds reassociation indices to prepend a unit batch dimension at position
/// 0. For a tensor of rank R, produces: [[0, 1], [2], [3], ..., [R]]
static SmallVector<ReassociationIndices> buildBatchReassociation(int64_t rank) {
  SmallVector<ReassociationIndices> reassoc;
  reassoc.push_back({0, 1}); // New batch dim groups with first existing dim
  for (int64_t i = 1; i < rank; ++i) {
    reassoc.push_back({i + 1});
  }
  return reassoc;
}

/// Expands a tensor by prepending a unit batch dimension at position 0.
/// tensor<AxBxC> -> tensor<1xAxBxC>
static Value expandWithBatchDim(PatternRewriter &rewriter, Location loc,
                                Value tensor) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  SmallVector<int64_t> newShape = {1};
  llvm::append_range(newShape, tensorType.getShape());
  auto newType = RankedTensorType::get(newShape, tensorType.getElementType());
  auto reassoc = buildBatchReassociation(tensorType.getRank());
  return tensor::ExpandShapeOp::create(rewriter, loc, newType, tensor, reassoc);
}

/// Transforms an indexing map to account for a new batch dimension.
/// Shifts all existing dimensions by 1 and prepends d0 (batch).
/// (d0, d1, ...) -> (...) becomes (d0, d1, d2, ...) -> (d0, ...)
static AffineMap prependBatchDimToMap(AffineMap oldMap, unsigned newNumLoops,
                                      MLIRContext *ctx) {
  AffineMap shifted = oldMap.shiftDims(1);
  SmallVector<AffineExpr> newResults;
  newResults.push_back(getAffineDimExpr(0, ctx)); // batch dim
  llvm::append_range(newResults, shifted.getResults());
  return AffineMap::get(newNumLoops, 0, newResults, ctx);
}

//===----------------------------------------------------------------------===//
// InsertBatchDimForConvPattern
//===----------------------------------------------------------------------===//

/// Pattern to insert batch dimension for batch-less convolutions.
///
/// This transforms a 6D linalg.generic (3 parallel + 3 reduction) representing
/// a batch-less conv into a 7D linalg.generic (4 parallel + 3 reduction) with
/// the batch dimension restored:
///
///   expand_shape(image_input) -> linalg.generic (with batch) ->
///   collapse_shape(output)
///
/// Note: The filter tensor does NOT get a batch dimension - it's the same
/// filter applied across the batch. Only the input and output tensors get
/// expanded.
struct InsertBatchDimForConvPattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(isBatchlessConv(op))) {
      return failure();
    }

    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    unsigned newNumLoops = op.getNumLoops() + 1;

    // --- Image input (operand 0): expand and add batch dim to map ---
    OpOperand *imageOperand = op.getDpsInputOperand(0);
    Value image = imageOperand->get();
    if (!isa<RankedTensorType>(image.getType())) {
      return rewriter.notifyMatchFailure(op,
                                         "image input is not a ranked tensor");
    }
    Value expandedImage = expandWithBatchDim(rewriter, loc, image);
    AffineMap newImageMap = prependBatchDimToMap(
        op.getMatchingIndexingMap(imageOperand), newNumLoops, ctx);

    // --- Filter (operand 1): keep tensor as-is, only shift map indices ---
    OpOperand *filterOperand = op.getDpsInputOperand(1);
    Value filter = filterOperand->get();
    AffineMap newFilterMap =
        op.getMatchingIndexingMap(filterOperand).shiftDims(1);

    // --- Output: expand and add batch dim to map ---
    OpOperand *outputOperand = op.getDpsInitOperand(0);
    Value output = outputOperand->get();
    auto outputType = dyn_cast<RankedTensorType>(output.getType());
    if (!outputType) {
      return rewriter.notifyMatchFailure(op, "output is not a ranked tensor");
    }
    Value expandedOutput = expandWithBatchDim(rewriter, loc, output);
    AffineMap newOutputMap = prependBatchDimToMap(
        op.getMatchingIndexingMap(outputOperand), newNumLoops, ctx);

    // Collect inputs and maps
    SmallVector<Value> newInputs = {expandedImage, filter};
    SmallVector<AffineMap> newMaps = {newImageMap, newFilterMap, newOutputMap};

    // New iterator types: prepend parallel (batch) to existing types
    SmallVector<utils::IteratorType> newIterTypes;
    newIterTypes.push_back(utils::IteratorType::parallel);
    llvm::append_range(newIterTypes, op.getIteratorTypesArray());

    // Create new generic with batch dimension
    auto newOutputType = cast<RankedTensorType>(expandedOutput.getType());
    auto newOp = linalg::GenericOp::create(
        rewriter, loc, TypeRange{newOutputType}, newInputs,
        ValueRange{expandedOutput}, newMaps, newIterTypes,
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
          // Clone the body
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

    // Collapse result to remove the batch dimension we added
    auto reassoc = buildBatchReassociation(outputType.getRank());
    auto collapsed = tensor::CollapseShapeOp::create(
        rewriter, loc, outputType, newOp.getResult(0), reassoc);

    rewriter.replaceOp(op, collapsed);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct InsertBatchDimForBatchlessConvPass final
    : impl::InsertBatchDimForBatchlessConvPassBase<
          InsertBatchDimForBatchlessConvPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<InsertBatchDimForConvPattern>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }

    // Run reshape propagation to push the expand/collapse shapes to boundaries
    {
      RewritePatternSet reshapePatterns(context);
      populateReshapeToInterfaceTensorPatterns(reshapePatterns);
      linalg::ControlFusionFn bubbleUpExpansionControlFn =
          [](OpOperand *fusedOperand) {
            // Allow fusion through all ops
            return true;
          };
      linalg::populateFoldReshapeOpsByExpansionPatterns(
          reshapePatterns, bubbleUpExpansionControlFn);
      tensor::CollapseShapeOp::getCanonicalizationPatterns(reshapePatterns,
                                                           context);
      tensor::ExpandShapeOp::getCanonicalizationPatterns(reshapePatterns,
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
