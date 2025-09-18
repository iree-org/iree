// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

#define GEN_PASS_DEF_DECOMPOSEMAPSCATTERPASS
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h.inc"

/// Fold a subview op into the output of a map_scatter op. The offsets of the
/// subview op are folded into the body of the map_scatter, and each offset is
/// added to the corresponding yielded index. For this pattern to apply, the
/// map_scatter op must be vectorized and bufferized, and the subview must be
/// non-rank reducing with unit strides. The subview's result also should not
/// be collapsible, because the decomposition will work for collapsable memrefs,
/// and there is no need to fold the subview.
struct FoldSubViewIntoMapScatter final : OpRewritePattern<MapScatterOp> {
  using OpRewritePattern<MapScatterOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MapScatterOp mapScatterOp,
                                PatternRewriter &rewriter) const override {
    if (!mapScatterOp.isVectorized()) {
      return rewriter.notifyMatchFailure(mapScatterOp,
                                         "map_scatter op is not vectorized");
    }
    if (!mapScatterOp.hasPureBufferSemantics()) {
      return rewriter.notifyMatchFailure(
          mapScatterOp, "map_scatter op has non-buffer semantics");
    }
    auto subViewOp =
        mapScatterOp.getOutput().getDefiningOp<memref::SubViewOp>();
    if (!subViewOp) {
      return failure();
    }
    if (subViewOp.getSourceType().getRank() != subViewOp.getType().getRank()) {
      return rewriter.notifyMatchFailure(subViewOp,
                                         "subview op is rank reducing");
    }
    SmallVector<OpFoldResult> strides = subViewOp.getMixedStrides();
    if (!areAllConstantIntValue(strides, 1)) {
      return rewriter.notifyMatchFailure(subViewOp,
                                         "subview op has non-unit strides");
    }
    MemRefType subViewType = subViewOp.getType();
    SmallVector<ReassociationIndices> reassociations;
    reassociations.push_back(
        llvm::to_vector(llvm::seq<int64_t>(subViewType.getRank())));
    if (memref::CollapseShapeOp::isGuaranteedCollapsible(subViewType,
                                                         reassociations)) {
      return rewriter.notifyMatchFailure(subViewOp,
                                         "subview op is collapsible");
    }
    auto mapScatterBodyYield = cast<IREE::LinalgExt::YieldOp>(
        mapScatterOp.getTransformationRegion().front().getTerminator());
    SmallVector<Value> yieldedIndices = mapScatterBodyYield.getOperands();
    Value yieldedMask = yieldedIndices.pop_back_val();
    rewriter.setInsertionPoint(mapScatterBodyYield);
    for (auto [yieldedIdx, subViewOffset] :
         llvm::zip_equal(yieldedIndices, subViewOp.getMixedOffsets())) {
      Value subViewOffsetVal = getValueOrCreateConstantIndexOp(
          rewriter, subViewOp.getLoc(), subViewOffset);
      Value yieldedIdxVal = getValueOrCreateConstantIndexOp(
          rewriter, mapScatterBodyYield.getLoc(), yieldedIdx);
      yieldedIdx = arith::AddIOp::create(rewriter, subViewOp.getLoc(),
                                         yieldedIdxVal, subViewOffsetVal);
    }
    SmallVector<Value> newYieldedValues(yieldedIndices);
    newYieldedValues.push_back(yieldedMask);
    rewriter.modifyOpInPlace(mapScatterBodyYield, [&]() {
      mapScatterBodyYield.getOperandsMutable().assign(newYieldedValues);
    });
    Value subViewSource = subViewOp.getSource();
    rewriter.modifyOpInPlace(subViewOp, [&]() {
      mapScatterOp.getOutputMutable().assign(subViewSource);
    });
    return success();
  }
};

/// Flatten the output buffer, and populate `strides` with the strides of the
/// flattened buffer if needed. If the buffer is collapsible, then the strides
/// will remain empty. Otherwise, the strides from the original buffer will be
/// added to the `strides` list.
static Value createFlatOutputBuffer(RewriterBase &rewriter, Location loc,
                                    Value outputBuffer,
                                    ArrayRef<OpFoldResult> sizes,
                                    SmallVectorImpl<Value> &strides) {
  auto outputBufferType = cast<MemRefType>(outputBuffer.getType());
  SmallVector<ReassociationIndices> reassociations;
  reassociations.push_back(
      llvm::to_vector(llvm::seq<int64_t>(outputBufferType.getRank())));
  if (memref::CollapseShapeOp::isGuaranteedCollapsible(outputBufferType,
                                                       reassociations)) {
    return memref::CollapseShapeOp::create(rewriter, loc, outputBuffer,
                                           reassociations);
  }
  auto stridedMetadataOp =
      memref::ExtractStridedMetadataOp::create(rewriter, loc, outputBuffer);
  strides.append(stridedMetadataOp.getStrides().begin(),
                 stridedMetadataOp.getStrides().end());
  Value offset = stridedMetadataOp.getOffset();
  OpFoldResult collapsedSize =
      IREE::LinalgExt::computeProductUsingAffine(rewriter, loc, sizes);
  SmallVector<OpFoldResult> collapsedShape = {collapsedSize};
  SmallVector<OpFoldResult> collapsedStrides = {rewriter.getIndexAttr(1)};
  return memref::ReinterpretCastOp::create(rewriter, loc, outputBuffer, offset,
                                           collapsedShape, collapsedStrides);
}

/// Decompose an iree_linalg_ext.map_scatter op with a vector input, and a
/// memref output. The map_scatter op is lowered into a sequence of vector ops
/// to compute a vector of indices for the elements of the map_scatter input,
/// and then a vector.scatter op to scatter the input vector to the output
/// buffer at the indices in the computed index vector. The output buffer is
/// also flattened to a 1D memref. If the collapse is not possible due to non
/// collapsible strides, then the decomposition will fail.
static LogicalResult decomposeMapScatter(MapScatterOp mapScatterOp,
                                         RewriterBase &rewriter) {
  Location loc = mapScatterOp.getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(mapScatterOp);
  SmallVector<OpFoldResult> outputSizes =
      memref::getMixedSizes(rewriter, loc, mapScatterOp.getOutput());
  SmallVector<Value> strides;
  Value flatOutputBuffer = createFlatOutputBuffer(
      rewriter, loc, mapScatterOp.getOutput(), outputSizes, strides);

  auto inputType = cast<VectorType>(mapScatterOp.getInputType());
  auto bodyBuilder = [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
    auto inlineBodyBuilder = [&](OpBuilder inlineBuilder, Location inlineLoc,
                                 ArrayRef<Value> yieldedValues) {
      SmallVector<Value> outputIndices(yieldedValues);
      Value mask = outputIndices.pop_back_val();
      Value linearIdx;
      // If strides are empty, this means that the memref layout was contiguous,
      // so we can simply linearize the indices based on the shape. Otherwise,
      // use the strides to compute the linear index.
      if (strides.empty()) {
        linearIdx = affine::AffineLinearizeIndexOp::create(
            inlineBuilder, inlineLoc, outputIndices, outputSizes,
            /*disjoint=*/true);
      } else {
        linearIdx = arith::ConstantIndexOp::create(inlineBuilder, inlineLoc, 0);
        for (auto [outputIdx, stride] :
             llvm::zip_equal(outputIndices, strides)) {
          Value stridedOutputIdx = arith::MulIOp::create(
              inlineBuilder, inlineLoc, outputIdx, stride);
          linearIdx = arith::AddIOp::create(inlineBuilder, inlineLoc, linearIdx,
                                            stridedOutputIdx);
        }
      }
      linalg::YieldOp::create(inlineBuilder, inlineLoc,
                              ValueRange{linearIdx, mask});
    };
    SmallVector<Value> indices = llvm::map_to_vector(
        llvm::seq<int64_t>(inputType.getRank()), [&](int64_t dim) -> Value {
          return linalg::IndexOp::create(b, nestedLoc, b.getIndexType(), dim);
        });
    mapScatterOp.inlineMapScatterBody(b, nestedLoc, indices, inlineBodyBuilder);
  };
  auto idxInit = tensor::EmptyOp::create(rewriter, loc, inputType.getShape(),
                                         rewriter.getIndexType());
  auto maskInit = tensor::EmptyOp::create(rewriter, loc, inputType.getShape(),
                                          rewriter.getIntegerType(1));
  SmallVector<AffineMap> maps(
      2, rewriter.getMultiDimIdentityMap(inputType.getRank()));
  SmallVector<utils::IteratorType> iterTypes(inputType.getRank(),
                                             utils::IteratorType::parallel);
  SmallVector<Value> outs = {idxInit.getResult(), maskInit.getResult()};
  auto genericOp =
      linalg::GenericOp::create(rewriter, loc, TypeRange(outs), ValueRange(),
                                outs, maps, iterTypes, bodyBuilder);

  // Lower linearize and delinearize ops before vectorizing, because the
  // vectorizer can't hendle them.
  SmallVector<affine::AffineLinearizeIndexOp> linearizeOps(
      genericOp.getBody()->getOps<affine::AffineLinearizeIndexOp>());
  for (auto linearizeOp : linearizeOps) {
    rewriter.setInsertionPoint(linearizeOp);
    if (failed(affine::lowerAffineLinearizeIndexOp(rewriter, linearizeOp))) {
      return rewriter.notifyMatchFailure(
          linearizeOp, "failed to lower affine.linearize_index op");
    }
  }
  SmallVector<affine::AffineDelinearizeIndexOp> delinearizeOps(
      genericOp.getBody()->getOps<affine::AffineDelinearizeIndexOp>());
  for (auto delinearizeOp : delinearizeOps) {
    rewriter.setInsertionPoint(delinearizeOp);
    if (failed(
            affine::lowerAffineDelinearizeIndexOp(rewriter, delinearizeOp))) {
      return rewriter.notifyMatchFailure(
          delinearizeOp, "failed to lower affine.delinearize_index op");
    }
  }

  FailureOr<linalg::VectorizationResult> result =
      linalg::vectorize(rewriter, genericOp);
  if (failed(result)) {
    return rewriter.notifyMatchFailure(mapScatterOp,
                                       "failed to generate index vector");
  }

  auto indexWriteOp =
      result->replacements[0].getDefiningOp<vector::TransferWriteOp>();
  auto maskWriteOp =
      result->replacements[1].getDefiningOp<vector::TransferWriteOp>();
  if (!indexWriteOp || !maskWriteOp) {
    return failure();
  }
  Value indexVector = indexWriteOp.getVector();
  Value maskVector = maskWriteOp.getVector();
  // Erase unused tensor ops after vectorizing the linalg.generic.
  rewriter.eraseOp(indexWriteOp);
  rewriter.eraseOp(maskWriteOp);
  rewriter.eraseOp(genericOp);

  // Flatten all the vectors, since the scatter op lowering expects 1D vectors.
  int64_t flatVectorSize =
      std::reduce(inputType.getShape().begin(), inputType.getShape().end(), 1,
                  std::multiplies<int64_t>());
  rewriter.setInsertionPoint(mapScatterOp);
  auto flatIndexType =
      VectorType::get({flatVectorSize}, rewriter.getIndexType());
  indexVector =
      vector::ShapeCastOp::create(rewriter, loc, flatIndexType, indexVector);
  auto flatMaskType =
      VectorType::get({flatVectorSize}, rewriter.getIntegerType(1));
  maskVector =
      vector::ShapeCastOp::create(rewriter, loc, flatMaskType, maskVector);
  auto flatInputType =
      VectorType::get({flatVectorSize}, inputType.getElementType());
  Value inputVector = vector::ShapeCastOp::create(rewriter, loc, flatInputType,
                                                  mapScatterOp.getInput());

  SmallVector<Value> offsets = {
      arith::ConstantIndexOp::create(rewriter, loc, 0)};
  rewriter.replaceOpWithNewOp<vector::ScatterOp>(mapScatterOp, flatOutputBuffer,
                                                 offsets, indexVector,
                                                 maskVector, inputVector);
  return success();
}

namespace {
struct DecomposeMapScatterPass final
    : impl::DecomposeMapScatterPassBase<DecomposeMapScatterPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::FunctionOpInterface funcOp = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<FoldSubViewIntoMapScatter>(context);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }

    if (testPreprocessingPatterns) {
      return;
    }

    // Decomposition is only supported for map_scatter ops that are both
    // vectorized and bufferized. Bufferization is a requirement because
    // vector.scatter only takes memref destinations.
    // TODO(#21135): Allow tensor outputs when vector.scatter supports tensor
    // destinations.
    SmallVector<MapScatterOp> candidates;
    funcOp->walk([&](MapScatterOp op) {
      if (isa<VectorType>(op.getInputType()) && op.hasPureBufferSemantics()) {
        candidates.push_back(op);
      }
    });
    IRRewriter rewriter(context);
    for (auto mapScatterOp : candidates) {
      if (failed(decomposeMapScatter(mapScatterOp, rewriter))) {
        return signalPassFailure();
      }
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler::IREE::LinalgExt
