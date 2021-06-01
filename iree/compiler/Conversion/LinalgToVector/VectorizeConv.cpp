// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Conversion/LinalgToVector/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-vectorize-conv"

namespace mlir {
namespace iree_compiler {

namespace {

/// Vectorizes linalg.conv_2d_input_nhwc_filter_hwcf for a single GPU
/// invocation. Therefore, the linalg.conv op should have a very specific form;
/// other patterns are expected to tile and distribute larger convolutions into
/// this form for a single GPU invocation.
///
/// The linalg.conv op should follow:
/// - Filter: HfWfCiCo format
/// - Input : NHiWiCi format
/// - Output: NHoWoCo format
/// - For output:
///   - N must be 1.
///   - Co must be a multiple of 4.
/// - For filter:
///   - Hf must be 1.
///   - Hf must be 1.
/// - No dilation.
/// - No padding.
///
/// Output channel is requried to be a multiple of 4 so that we can process
/// them with load4/store4, which is native to GPUs. Similarly for the input
/// channel size requirement.
struct VectorizeLinalgConv
    : OpRewritePattern<linalg::ConvInputNHWCFilterHWCFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ConvInputNHWCFilterHWCFOp convOp,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "inspecting " << convOp << "\n");

    // This pattern does not handle convolutions with dilation.
    if (auto dilations = convOp.dilations()) {
      auto values = dilations.getIntValues();
      if (llvm::any_of(values, [](const APInt &value) {
            return value.getSExtValue() != 1;
          })) {
        return failure();
      }
    }

    auto inputViewOp =
        convOp.getInputOperand(0)->get().getDefiningOp<memref::SubViewOp>();
    auto filterViewOp =
        convOp.getInputOperand(1)->get().getDefiningOp<memref::SubViewOp>();
    auto outputViewOp =
        convOp.getOutputOperand(0)->get().getDefiningOp<memref::SubViewOp>();
    if (!filterViewOp || !inputViewOp || !outputViewOp) return failure();

    // The filter/input/output view should have static sizes to vectorize.
    if (!llvm::empty(filterViewOp.getDynamicSizes()) ||
        !llvm::empty(inputViewOp.getDynamicSizes()) ||
        !llvm::empty(outputViewOp.getDynamicSizes())) {
      return failure();
    }

    // The output batch dimension should be 1.
    if (outputViewOp.getStaticSize(0) != 1) return failure();

    // We addtionally expect the filter height/width dimensions are both 1 to
    // simplify vectorization. Other patterns can generate loops to create 1x1
    // filter subivews.
    if (filterViewOp.getStaticSize(0) != 1 ||
        filterViewOp.getStaticSize(1) != 1) {
      return failure();
    }

    int64_t numInputChannels = filterViewOp.getStaticSize(2);
    int64_t numOutputChannels = filterViewOp.getStaticSize(3);
    if (numInputChannels > 4 || numOutputChannels % 4 != 0) return failure();

    int64_t numOutputHeights = outputViewOp.getStaticSize(1);
    int64_t numOutputWidths = outputViewOp.getStaticSize(2);
    int64_t heightStride = convOp.strides().getValue<int64_t>({0});
    int64_t widthStride = convOp.strides().getValue<int64_t>({1});

    // This invocation handles a batch of
    // (numOutputHeights * numOutputWidths * numOutputChannels).
    LLVM_DEBUG({
      llvm::dbgs() << "# output height: " << numOutputHeights << "\n";
      llvm::dbgs() << "# output width: " << numOutputWidths << "\n";
      llvm::dbgs() << "# output channels: " << numOutputChannels << "\n";
      llvm::dbgs() << "height stride: " << heightStride << "\n";
      llvm::dbgs() << "width stride: " << widthStride << "\n";
    });

    MLIRContext *context = convOp.getContext();
    Location loc = convOp.getLoc();

    Type elementType = filterViewOp.getType().getElementType();
    auto filterVectorType =
        VectorType::get({numInputChannels, numOutputChannels}, elementType);
    auto vector1x4Type = VectorType::get({1, 4}, elementType);
    auto inputVectorType = VectorType::get({1, numInputChannels}, elementType);
    Value zero = rewriter.createOrFold<ConstantIndexOp>(loc, 0);

    // Load the entire filter subview.
    SmallVector<Value, 4> filterIndices(4, zero);
    Value wholeFilter = rewriter.create<vector::TransferReadOp>(
        loc, filterVectorType, filterViewOp, filterIndices);

    // Get filter slices so that later we can use them for dot product with the
    // input. Both the height and width dimensions are 1; so we just need to
    // loop over input and output channel dimensions.
    SmallVector<SmallVector<Value, 1>, 4> filterVectors(numInputChannels);
    for (int ic = 0; ic < numInputChannels; ++ic) {
      auto &thisInputChannel = filterVectors[ic];
      thisInputChannel.reserve(numOutputChannels / 4);
      for (int oc = 0; oc < numOutputChannels / 4; ++oc) {
        Value slice = rewriter.create<vector::ExtractStridedSliceOp>(
            loc, wholeFilter, /*offsets=*/ArrayRef<int64_t>({ic, oc * 4}),
            /*sizes=*/ArrayRef<int64_t>({1, 4}),
            /*strides=*/ArrayRef<int64_t>({1, 1}));
        thisInputChannel.push_back(slice);
      }
    }

    // Build indexing maps for a later vector contraction op.
    AffineExpr dim0 = getAffineDimExpr(0, context);  // M
    AffineExpr dim1 = getAffineDimExpr(1, context);  // N
    AffineExpr dim2 = getAffineDimExpr(2, context);  // K
    auto map02 = AffineMap::get(3, 0, {dim0, dim2}, context);
    auto map21 = AffineMap::get(3, 0, {dim2, dim1}, context);
    auto map01 = AffineMap::get(3, 0, {dim0, dim1}, context);
    ArrayAttr indexingMaps =
        rewriter.getAffineMapArrayAttr({map02, map21, map01});

    // Also build iterator types for the vector contraction op.
    ArrayAttr iterators = rewriter.getStrArrayAttr(
        {getParallelIteratorTypeName(), getParallelIteratorTypeName(),
         getReductionIteratorTypeName()});

    // Compute the (numOutputHeights * numOutputWidths * numOutputChannels)
    // batch. We only contribute numInputChannels accumulation along the
    // reduction dimension. So read in the result from the output, compose a
    // chain of numInputChannels vector dot operations, and then write out.
    for (int oh = 0; oh < numOutputHeights; ++oh) {
      for (int ow = 0; ow < numOutputWidths; ++ow) {
        // Read in the input vector for these 4 input channels a a batch. The
        // input vector are used for computing all output channels so data can
        // be reused.
        SmallVector<Value, 4> inputIndices(4, zero);
        inputIndices[1] =
            rewriter.createOrFold<ConstantIndexOp>(loc, oh * heightStride);
        inputIndices[2] =
            rewriter.createOrFold<ConstantIndexOp>(loc, ow * widthStride);
        Value inputVector = rewriter.create<vector::TransferReadOp>(
            loc, inputVectorType, inputViewOp, inputIndices);

        for (int oc = 0; oc < numOutputChannels / 4; ++oc) {
          // Read in the initial value for this output vector.
          SmallVector<Value, 4> outputIndices(4, zero);
          outputIndices[1] = rewriter.createOrFold<ConstantIndexOp>(loc, oh);
          outputIndices[2] = rewriter.createOrFold<ConstantIndexOp>(loc, ow);
          outputIndices[3] =
              rewriter.createOrFold<ConstantIndexOp>(loc, oc * 4);
          Value outputVector = rewriter.create<vector::TransferReadOp>(
              loc, vector1x4Type, outputViewOp, outputIndices);

          // Peform a chain of dot product and accumulation.
          for (int i = 0; i < numInputChannels; ++i) {
            auto inputSlice = rewriter.create<vector::ExtractStridedSliceOp>(
                loc, inputVector, /*offsets=*/ArrayRef<int64_t>({0, i}),
                /*sizes=*/ArrayRef<int64_t>({1, 1}),
                /*strides=*/ArrayRef<int64_t>({1, 1}));
            outputVector = rewriter.create<vector::ContractionOp>(
                loc, inputSlice, filterVectors[i][oc], outputVector,
                indexingMaps, iterators);
          }

          // Write out the output vector.
          rewriter.create<vector::TransferWriteOp>(loc, outputVector,
                                                   outputViewOp, outputIndices);
        }
      }
    }

    rewriter.eraseOp(convOp);
    return success();
  }
};

/// Vectorizes linalg.depthwise_conv_2d_input_nhwc_filter_hwc for a single GPU
/// invocation. Therefore, the linalg.depthwise_conv_2d_input_nhwc_filter_hwc op
/// should have a very specific form; other patterns are expected to tile and
/// distribute larger convolutions into this form for a single GPU invocation.
///
/// The linalg.depthwise_conv_2d_input_nhwc_filter_hwc op should follow:
/// - Filter: HfWfC format
/// - Input : NHiWiC format
/// - Output: NHoWoC format
/// - For output:
///   - N must be 1.
///   - C must be a multiple of 4.
/// - For filter:
///   - Hf must be 1.
///   - Hf must be 1.
/// - No dilation.
/// - No padding.
///
/// Channel is requried to be a multiple of 4 so that we can process them with
/// load4/store4, which is native to GPUs.
struct VectorizeLinalgDepthwiseConv
    : OpRewritePattern<linalg::DepthwiseConvInputNHWCFilterHWCOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::DepthwiseConvInputNHWCFilterHWCOp convOp,
      PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "inspecting " << convOp << "\n");

    auto inputViewOp =
        convOp.getInputOperand(0)->get().getDefiningOp<memref::SubViewOp>();
    auto filterViewOp =
        convOp.getInputOperand(1)->get().getDefiningOp<memref::SubViewOp>();
    auto outputViewOp =
        convOp.getOutputOperand(0)->get().getDefiningOp<memref::SubViewOp>();
    if (!filterViewOp || !inputViewOp || !outputViewOp) return failure();

    // The filter/input/output view should have static sizes to vectorize.
    if (!llvm::empty(filterViewOp.getDynamicSizes()) ||
        !llvm::empty(inputViewOp.getDynamicSizes()) ||
        !llvm::empty(outputViewOp.getDynamicSizes())) {
      return failure();
    }

    // The output batch dimension should be 1.
    if (outputViewOp.getStaticSize(0) != 1) return failure();

    // We addtionally expect the filter height/width dimensions are both 1 to
    // simplify vectorization. Other patterns can generate loops to create 1x1
    // filter subivews.
    if (filterViewOp.getStaticSize(0) != 1 ||
        filterViewOp.getStaticSize(1) != 1) {
      return failure();
    }

    int64_t numChannels = outputViewOp.getStaticSize(3);
    if (numChannels % 4 != 0) return failure();

    int64_t numOutputHeights = outputViewOp.getStaticSize(1);
    int64_t numOutputWidths = outputViewOp.getStaticSize(2);
    int64_t heightStride = convOp.strides().getValue<int64_t>({0});
    int64_t widthStride = convOp.strides().getValue<int64_t>({1});

    // This invocation handles a batch of (numOutputHeights * numOutputWidths *
    // numChannels).
    LLVM_DEBUG({
      llvm::dbgs() << "# output height: " << numOutputHeights << "\n";
      llvm::dbgs() << "# output width: " << numOutputWidths << "\n";
      llvm::dbgs() << "# channels: " << numChannels << "\n";
      llvm::dbgs() << "height stride: " << heightStride << "\n";
      llvm::dbgs() << "width stride: " << widthStride << "\n";
    });

    Location loc = convOp.getLoc();

    Type elementType = filterViewOp.getType().getElementType();
    auto vector4Type = VectorType::get(4, elementType);
    auto filterVectorType = VectorType::get({numChannels}, elementType);
    Value zero = rewriter.createOrFold<ConstantIndexOp>(loc, 0);

    // Load the entire filter subview.
    SmallVector<Value, 4> filterIndices(3, zero);
    Value wholeFilter = rewriter.create<vector::TransferReadOp>(
        loc, filterVectorType, filterViewOp, filterIndices);

    // Compute the (numOutputHeights * numOutputWidths * numChannels) output
    // batch. We only contribute numChannels accumulation along the reduction
    // dimension.
    for (int oc = 0; oc < numChannels / 4; ++oc) {
      Value filterVector = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, wholeFilter, /*offsets=*/oc * 4, /*sizes=*/4, /*strides=*/1);

      for (int oh = 0; oh < numOutputHeights; ++oh) {
        for (int ow = 0; ow < numOutputWidths; ++ow) {
          // Read in the initial value for this output vector.
          SmallVector<Value, 4> outputIndices(4, zero);
          outputIndices[1] = rewriter.createOrFold<ConstantIndexOp>(loc, oh);
          outputIndices[2] = rewriter.createOrFold<ConstantIndexOp>(loc, ow);
          outputIndices[3] =
              rewriter.createOrFold<ConstantIndexOp>(loc, oc * 4);
          Value outputVector = rewriter.create<vector::TransferReadOp>(
              loc, vector4Type, outputViewOp, outputIndices);

          // Read in the input vector for these 4 input channels a a batch.
          SmallVector<Value, 4> inputIndices(4, zero);
          inputIndices[1] =
              rewriter.createOrFold<ConstantIndexOp>(loc, oh * heightStride);
          inputIndices[2] =
              rewriter.createOrFold<ConstantIndexOp>(loc, ow * widthStride);
          inputIndices[3] = rewriter.createOrFold<ConstantIndexOp>(loc, oc * 4);
          Value inputVector = rewriter.create<vector::TransferReadOp>(
              loc, vector4Type, inputViewOp, inputIndices);

          // Peform element-wise product and accumulation.
          outputVector = rewriter.create<vector::FMAOp>(
              loc, inputVector, filterVector, outputVector);

          // Write out the output vector.
          rewriter.create<vector::TransferWriteOp>(loc, outputVector,
                                                   outputViewOp, outputIndices);
        }
      }
    }

    rewriter.eraseOp(convOp);
    return success();
  }
};

struct VectorizeLinalgConvPass
    : public PassWrapper<VectorizeLinalgConvPass, OperationPass<FuncOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    OwningRewritePatternList patterns(&getContext());
    patterns.insert<VectorizeLinalgConv, VectorizeLinalgDepthwiseConv>(context);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace

void populateVectorizeLinalgConvPatterns(MLIRContext *context,
                                         OwningRewritePatternList &patterns) {
  patterns.insert<VectorizeLinalgConv, VectorizeLinalgDepthwiseConv>(context);
}

std::unique_ptr<Pass> createVectorizeLinalgConvPass() {
  return std::make_unique<VectorizeLinalgConvPass>();
}

static PassRegistration<VectorizeLinalgConvPass> pass(
    "iree-codegen-vectorize-linalg-conv",
    "Vectorize a very specific form of linalg.conv");

}  // namespace iree_compiler
}  // namespace mlir
