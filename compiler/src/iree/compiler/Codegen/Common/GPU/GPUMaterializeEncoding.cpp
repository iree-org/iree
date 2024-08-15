// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/IR/Operation.h"

#define DEBUG_TYPE "iree-codegen-gpu-materialize-encoding"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUMATERIALIZEDEVICEENCODINGPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

// TODO: Query the value from GPU attributes.
static std::optional<TileMxNxK> getIntrinsicSize(TypeRange elementTypes) {
  Type lhs = elementTypes[0];
  Type rhs = elementTypes[1];
  Type out = elementTypes[2];
  if (lhs.isF32() && rhs.isF32() && out.isF32()) {
    return TileMxNxK{16, 16, 4};
  }
  return std::nullopt;
}

// TODO: Query the value from GPU attributes.
// TODO: Define a struct with meaningful name for the pair.
SmallVector<int64_t> getIntrinsicVectorSize(TypeRange elementTypes,
                                            int64_t roleIdx) {
  Type lhs = elementTypes[0];
  Type rhs = elementTypes[1];
  Type out = elementTypes[2];
  if (lhs.isF32() && rhs.isF32() && out.isF32()) {
    if (roleIdx == 0 || roleIdx == 1) {
      return {1, 1};
    }
    if (roleIdx == 2) {
      return {4, 1};
    }
  }
  return {};
}

// Given encoding's role index and element types, return the transpose
// permutation used in GPU materialization.
SmallVector<int64_t> getTransposePermutation(int64_t roleIdx,
                                             TypeRange elementTypes) {
  // For now, check that all types are f32:
  Type lhs = elementTypes[0];
  Type rhs = elementTypes[1];
  Type out = elementTypes[2];
  if (!lhs.isF32() || !rhs.isF32() || !out.isF32()) {
    return {};
  }

  switch (roleIdx) {
  case 0: // A
  case 1: // B
    // OuterTileX x InnerTileX x OuterTileY x InnerTileY
    // -> OuterTileY x OuterTileX x InnerTileY x InnerTileX
    return {2, 0, 3, 1};
  case 2: // C
    // ACC:
    // OuterTileX x InnerTileX x OuterTileY x InnerTileY
    // -> OuterTileX x OuterTileY x InnerTileX x InnerTileY
    return {0, 2, 1, 3};
  default:
    return {};
  }
}

// TODO(hanchung): Pass an ExecutableTargetAttr attribute for the target
// encoding. Here we assume that every mfma op is available.
// TODO(hanchung): Handle wmma ops.
static SmallVector<TileMxNxK> enumerateMatmulTileMxNxK(TypeRange elementTypes) {
  assert(elementTypes.size() == 3);
  Type lhs = elementTypes[0];
  Type rhs = elementTypes[1];
  Type out = elementTypes[2];
  if (lhs.isF32() && rhs.isF32() && out.isF32()) {
    // TODO: Take subgroup_size into account, so we can have more unrolling.
    // TODO: Take the bitwidth of load into account, so we can have correct
    // unrolling factor for K-dimension.
    return {TileMxNxK{16, 16, 4}}; // Aim to use mfma_f32_16x16x4_f32 intrinsic.
  }

  // Fallback - no architecture-optimized tile size for this case.
  return {};
}

static FailureOr<MaterializeEncodingInfo>
materializeEncodingForTarget(RankedTensorType tensorType) {
  auto encoding =
      dyn_cast_or_null<IREE::Encoding::EncodingAttr>(tensorType.getEncoding());
  if (!encoding) {
    return failure();
  }
  // We only know about contractions with {Batch, M, N, K} <= 1 at the moment.
  auto cDims = getEncodingContractionDims(encoding);
  if (failed(cDims) || cDims->batch.size() > 1 || cDims->m.size() > 1 ||
      cDims->n.size() > 1 || cDims->k.size() > 1) {
    return failure();
  }
  // Enumerate available tile shapes for the given encoding and target.
  auto elementTypes = llvm::to_vector(
      llvm::map_range(encoding.getElementTypes().getValue(), [](Attribute a) {
        return cast<TypeAttr>(a).getValue();
      }));
  SmallVector<TileMxNxK> enumeratedTileMxNxK =
      enumerateMatmulTileMxNxK(elementTypes);
  if (enumeratedTileMxNxK.empty()) {
    return failure();
  }

  // Map the matmul TileMxNxK to an actual tile shape for the tensor at hand,
  // based on its operand index in the matmul.
  auto rank = tensorType.getRank();

  auto encodingInfo =
      getEncodingInfoForMatmul(encoding, rank, enumeratedTileMxNxK[0]);

  // insert inner tile shapes and permutation info
  auto roleIdx = encoding.getOperandIndex().getInt();
  auto intrinsicVectorSizes = getIntrinsicVectorSize(elementTypes, roleIdx);
  auto permutation = getTransposePermutation(roleIdx, elementTypes);
  encodingInfo.innerTileShapes = intrinsicVectorSizes;
  encodingInfo.permutation = permutation;
  return encodingInfo;
}

namespace {
struct GPUMaterializeDeviceEncodingPass final
    : impl::GPUMaterializeDeviceEncodingPassBase<
          GPUMaterializeDeviceEncodingPass> {
  using GPUMaterializeDeviceEncodingPassBase::
      GPUMaterializeDeviceEncodingPassBase;
  void runOnOperation() override;
};

/// Convert iree_linalg_ext.set_encoding op to pack + tile swizzling ops. We use
/// expand_shape + linalg.transpose to represent a tile swizzling op.
struct GPUSetEncodingOpLoweringConversion
    : public OpMaterializeEncodingPattern<IREE::Encoding::SetEncodingOp> {
  using OpMaterializeEncodingPattern<
      IREE::Encoding::SetEncodingOp>::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(IREE::Encoding::SetEncodingOp encodingOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = static_cast<const MaterializeEncodingTypeConverter *>(
        getTypeConverter());
    MaterializeEncodingFn materializeEncodingFn =
        converter->getMaterializeEncodingFn();

    auto packOp = lowerSetEncodingOpToPackOp(
        rewriter, encodingOp, adaptor.getSource(), materializeEncodingFn,
        this->materializeEncodingValueFn);
    if (failed(packOp)) {
      Value result = adaptor.getSource();
      Type targetType =
          getTypeConverter()->convertType(encodingOp.getResultType());
      if (targetType != result.getType()) {
        result = rewriter.create<tensor::CastOp>(encodingOp.getLoc(),
                                                 targetType, result);
      }
      rewriter.replaceOp(encodingOp, result);
      return success();
    }

    FailureOr<MaterializeEncodingInfo> maybeEncodingInfo =
        materializeEncodingFn(encodingOp.getResultType());
    if (failed(maybeEncodingInfo)) {
      return rewriter.notifyMatchFailure(encodingOp,
                                         "unhandled result encoding");
    }
    SmallVector<int64_t> innerTiles = maybeEncodingInfo->innerTileSizes;
    SmallVector<int64_t> intrinsicVectorShape =
        maybeEncodingInfo->innerTileShapes;

    // TODO(hanchung): Add a util to the encoding attribute, so we don't need
    // the map_to_vector method here.
    auto encoding = IREE::Encoding::getEncodingAttr(encodingOp.getResultType());
    int64_t roleIdx = encoding.getOperandIndex().getInt();
    auto elemTypes = llvm::map_to_vector(
        encoding.getElementTypes().getValue(),
        [](Attribute a) { return cast<TypeAttr>(a).getValue(); });
    auto loc = encodingOp.getLoc();

    std::optional<TileMxNxK> intrinsicShape = getIntrinsicSize(elemTypes);
    if (!intrinsicShape || intrinsicVectorShape.empty()) {
      return failure();
    }

    SmallVector<int64_t> targetShape; // for unrolling
    switch (roleIdx) {
    case 0: // A
      targetShape = {intrinsicShape->M, intrinsicShape->K};
      break;
    case 1: // B
      targetShape = {intrinsicShape->N, intrinsicShape->K};
      break;
    case 2: // C
      targetShape = {intrinsicShape->M, intrinsicShape->N};
      break;
    default:
      return failure();
    }

    assert(innerTiles.size() == targetShape.size());
    for (auto [packedShape, targetShape] :
         llvm::zip_equal(innerTiles, targetShape)) {
      // TODO(lialan): Relax the condition for unrolling when it is supported.
      (void)packedShape;
      (void)targetShape;
      assert(packedShape == targetShape);
    }

    // Create expand_shape op to tile the innermost two dimensions.
    auto sourceShape = packOp->getDestType().getShape();
    assert(intrinsicVectorShape.size() == 2); // TODO: relax this
    auto iT1 = intrinsicVectorShape[0];
    auto iT2 = intrinsicVectorShape[1];
    auto oT1 = sourceShape[2] / iT1;
    auto oT2 = sourceShape[3] / iT2;
    SmallVector<int64_t> expandShapeShape = {
        sourceShape[0], sourceShape[1], oT1, iT1, oT2, iT2};
    assert(expandShapeShape.size() == 6);
    auto expandShapeType = RankedTensorType::get(
        expandShapeShape, encodingOp.getSourceType().getElementType());

    std::optional<SmallVector<ReassociationIndices>> reassociationMap =
        getReassociationIndicesForReshape(packOp->getDestType(),
                                          expandShapeType);
    assert(reassociationMap.has_value());
    auto expandShapeOp = rewriter.create<tensor::ExpandShapeOp>(
        loc, expandShapeType, packOp->getResult(), *reassociationMap);

    // create linalg.transpose on expandShapeShape
    size_t origRank = encodingOp.getSourceType().getRank();

    SmallVector<int64_t> transposePerm;
    transposePerm.push_back(0);
    transposePerm.push_back(1);
    for (auto perm : maybeEncodingInfo->permutation) {
      transposePerm.push_back(origRank + perm);
    }
    SmallVector<int64_t> transposeResultDims = expandShapeShape;
    applyPermutationToVector(transposeResultDims, transposePerm);

    auto emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, transposeResultDims, encodingOp.getSourceType().getElementType());
    auto transposeOp = rewriter.create<linalg::TransposeOp>(
        loc, expandShapeOp, emptyTensor, transposePerm);

    // We want to make the shape consistent, so we need to append it with a
    // `collapse_shape` and a `expand_shape`, just to be conformant with how we
    // materialize for Flow and HAL op.

    // 1. collapse tiled dimensions into one dim
    SmallVector<int64_t> collapsedShape = {sourceShape[0], sourceShape[1],
                                           sourceShape[2] * sourceShape[3]};
    auto revertShapeType = RankedTensorType::get(
        collapsedShape, encodingOp.getSourceType().getElementType());

    std::optional<SmallVector<ReassociationIndices>> collapseReassoc =
        getReassociationIndicesForReshape(emptyTensor.getType(),
                                          revertShapeType);
    assert(collapseReassoc.has_value());

    auto collapseShapeOp = rewriter.create<tensor::CollapseShapeOp>(
        loc, revertShapeType, transposeOp->getResult(0), *collapseReassoc);

    // 2. expand the collapsed shape to the shape intended by the encoding
    assert(innerTiles.size() == 2); // TODO: relax this
    auto expandTileShapeType = RankedTensorType::get(
        {sourceShape[0], sourceShape[1], innerTiles[0], innerTiles[1]},
        encodingOp.getSourceType().getElementType());
    std::optional<SmallVector<ReassociationIndices>> tileAssoc =
        getReassociationIndicesForReshape(collapseShapeOp.getType(),
                                          expandTileShapeType);
    assert(tileAssoc.has_value());
    auto expandTileShapeOp = rewriter.create<tensor::ExpandShapeOp>(
        loc, expandTileShapeType, collapseShapeOp, *tileAssoc);

    rewriter.replaceOp(encodingOp, expandTileShapeOp);
    return success();
  }
};

} // namespace

void GPUMaterializeDeviceEncodingPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  FunctionOpInterface funcOp = getOperation();
  {
    RewritePatternSet patterns(ctx);
    MaterializeEncodingTypeConverter typeConverter(
        materializeEncodingForTarget);
    MaterializeEncodingConversionTarget target(*funcOp.getContext());
    MaterializeEncodingValueFn materializeEncodingValueFn =
        [](RankedTensorType, OpBuilder,
           Location) -> FailureOr<MaterializeEncodingValueInfo> { return {}; };
    populateIREEMaterializeEncodingIntoPackUnPackPatterns(
        patterns, target, typeConverter, materializeEncodingValueFn);

    patterns.insert<GPUSetEncodingOpLoweringConversion>(
        ctx, typeConverter, materializeEncodingValueFn);

    if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
      funcOp.emitOpError("materialization failed");
      return signalPassFailure();
    }
  }

  // Add patterns to fold pack/unpack ops with pad/extract_slice ops and
  // resolve dims ops.
  {
    RewritePatternSet patterns(ctx);
    tensor::populateFoldIntoPackAndUnpackPatterns(patterns);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitOpError("folding patterns failed");
      return signalPassFailure();
    }
  }
}

} // namespace mlir::iree_compiler
