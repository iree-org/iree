// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-materialize-encoding"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUMATERIALIZEDEVICEENCODINGPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

/// Returns the corresponding native vector sizes defined by the `mma`
/// intrinsic.
static SmallVector<int64_t> getIntrinsicVectorSize(IREE::GPU::MMAAttr mma,
                                                   int64_t roleIdx) {
  if (mma.getIntrinsic().getValue() ==
      IREE::GPU::MMAIntrinsic::MFMA_F32_16x16x4_F32) {
    // TODO: Query the value from GPU attributes.
    if (roleIdx == 0 || roleIdx == 1) {
      return {1, 1};
    }
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
static SmallVector<int64_t> getEncodingTransposePerm(IREE::GPU::MMAAttr mma,
                                                     int64_t roleIdx) {
  // TODO: Support other intrinsics.
  if (mma.getIntrinsic().getValue() !=
      IREE::GPU::MMAIntrinsic::MFMA_F32_16x16x4_F32) {
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

static std::optional<IREE::GPU::MMAAttr>
enumerateMmaIntrinsic(TypeRange elementTypes, IREE::GPU::TargetAttr target) {
  assert(elementTypes.size() == 3);
  Type lhs = elementTypes[0];
  Type rhs = elementTypes[1];
  Type out = elementTypes[2];
  for (IREE::GPU::MMAAttr mma : target.getWgp().getMma()) {
    IREE::GPU::MMAIntrinsic type = mma.getIntrinsic().getValue();
    // TODO: Drop this once all intrinsics are supported.
    if (type != IREE::GPU::MMAIntrinsic::MFMA_F32_16x16x4_F32) {
      continue;
    }

    auto [aType, bType, cType] = mma.getABCElementTypes();
    if (lhs != aType || rhs != bType || out != cType) {
      continue;
    }
    return mma;
  }

  // Fallback - no architecture-optimized tile size for this case.
  return std::nullopt;
}

static FailureOr<MaterializeEncodingInfo>
materializeEncodingForTarget(RankedTensorType tensorType,
                             IREE::HAL::ExecutableTargetAttr targetAttr) {
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
  IREE::GPU::TargetAttr gpuTargetAttr;
  if (targetAttr) {
    gpuTargetAttr = getGPUTargetAttr(targetAttr);
  } else {
    gpuTargetAttr = getCLGPUTarget(tensorType.getContext());
  }
  auto elementTypes = llvm::to_vector(
      llvm::map_range(encoding.getElementTypes().getValue(), [](Attribute a) {
        return cast<TypeAttr>(a).getValue();
      }));
  std::optional<IREE::GPU::MMAAttr> mma =
      enumerateMmaIntrinsic(elementTypes, gpuTargetAttr);
  if (!mma) {
    return failure();
  }

  // Map the matmul TileMxNxK to an actual tile shape for the tensor at hand,
  // based on its operand index in the matmul.
  // TODO: Support unrolling.
  auto rank = tensorType.getRank();
  TileMxNxK innerTile;
  std::tie(innerTile.M, innerTile.N, innerTile.K) = mma->getMNKShape();
  auto encodingInfo = getEncodingInfoForMatmul(encoding, rank, innerTile);

  // insert inner tile shapes and permutation info
  auto roleIdx = encoding.getOperandIndex().getInt();
  auto intrinsicVectorSizes = getIntrinsicVectorSize(*mma, roleIdx);
  auto permutation = getEncodingTransposePerm(*mma, roleIdx);
  encodingInfo.innerTileShapes = intrinsicVectorSizes;
  encodingInfo.intrinsicSize = {mma->getMSize(), mma->getNSize(),
                                mma->getKSize()};
  encodingInfo.permutation = permutation;
  return encodingInfo;
}

namespace {
struct GPUMaterializeDeviceEncodingPass final
    : impl::GPUMaterializeDeviceEncodingPassBase<
          GPUMaterializeDeviceEncodingPass> {
  using GPUMaterializeDeviceEncodingPassBase::
      GPUMaterializeDeviceEncodingPassBase;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, tensor::TensorDialect,
                    linalg::LinalgDialect, IREE::Encoding::IREEEncodingDialect,
                    IREE::GPU::IREEGPUDialect>();
  }
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
    auto packOp = lowerSetEncodingOpToPackOp(rewriter, encodingOp,
                                             adaptor.getSource(), *converter,
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
        converter->getEncodingInfo(encodingOp.getResultType());
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
    auto loc = encodingOp.getLoc();

    SmallVector<int64_t> targetShape; // for unrolling
    switch (roleIdx) {
    case 0: // A
      targetShape = {maybeEncodingInfo->intrinsicSize[0],
                     maybeEncodingInfo->intrinsicSize[2]};
      break;
    case 1: // B
      targetShape = {maybeEncodingInfo->intrinsicSize[1],
                     maybeEncodingInfo->intrinsicSize[2]};
      break;
    case 2: // C
      targetShape = {maybeEncodingInfo->intrinsicSize[0],
                     maybeEncodingInfo->intrinsicSize[1]};
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
    rewriter.replaceOp(encodingOp, transposeOp->getResult(0));

    return success();
  }
};

struct GPUUnsetEncodingOpLoweringConversion
    : public OpMaterializeEncodingPattern<IREE::Encoding::UnsetEncodingOp> {
  using OpMaterializeEncodingPattern<
      IREE::Encoding::UnsetEncodingOp>::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(IREE::Encoding::UnsetEncodingOp unsetEncodingOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = static_cast<const MaterializeEncodingTypeConverter *>(
        getTypeConverter());

    Location loc = unsetEncodingOp.getLoc();

    FailureOr<MaterializeEncodingInfo> maybeEncodingInfo =
        converter->getEncodingInfo(unsetEncodingOp.getSource().getType());
    if (failed(maybeEncodingInfo)) {
      return rewriter.notifyMatchFailure(unsetEncodingOp,
                                         "unhandled result encoding");
    }
    SmallVector<int64_t> innerTiles = maybeEncodingInfo->innerTileSizes;
    SmallVector<int64_t> intrinsicVectorShape =
        maybeEncodingInfo->innerTileShapes;

    // compute sourceShape:
    auto srcConvertedType =
        cast<RankedTensorType>(adaptor.getSource().getType());
    SmallVector<int64_t> unpackSrcShape(
        srcConvertedType.getShape().take_front(maybeEncodingInfo->srcRank));
    unpackSrcShape.append(maybeEncodingInfo->innerTileSizes.begin(),
                          maybeEncodingInfo->innerTileSizes.end());
    auto unpackSrcType = RankedTensorType::get(
        unpackSrcShape, unsetEncodingOp.getSourceType().getElementType());

    auto iT1 = intrinsicVectorShape[0];
    auto iT2 = intrinsicVectorShape[1];
    auto oT1 = unpackSrcShape[2] / iT1;
    auto oT2 = unpackSrcShape[3] / iT2;
    SmallVector<int64_t> transposeSourceDims = {
        unpackSrcShape[0], unpackSrcShape[1], oT1, iT1, oT2, iT2};
    assert(transposeSourceDims.size() == 6);

    size_t targetRank = unsetEncodingOp.getResultType().getRank();

    SmallVector<int64_t> transposePerm;
    transposePerm.push_back(0);
    transposePerm.push_back(1);
    for (auto perm : maybeEncodingInfo->permutation) {
      transposePerm.push_back(targetRank + perm);
    }
    SmallVector<int64_t> expandShapeResultDims = transposeSourceDims;
    applyPermutationToVector(expandShapeResultDims, transposePerm);
    auto invertedTransposePerm = invertPermutationVector(transposePerm);

    auto emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, transposeSourceDims,
        unsetEncodingOp.getSourceType().getElementType());
    auto transposeOp = rewriter.create<linalg::TransposeOp>(
        loc, adaptor.getSource(), emptyTensor, invertedTransposePerm);

    auto transposeResultType = RankedTensorType::get(
        transposeSourceDims, unsetEncodingOp.getSourceType().getElementType());
    std::optional<SmallVector<ReassociationIndices>> collapseReassoc =
        getReassociationIndicesForReshape(transposeResultType, unpackSrcType);
    assert(collapseReassoc.has_value());
    auto collapseShapeOp = rewriter.create<tensor::CollapseShapeOp>(
        loc, unpackSrcType, transposeOp->getResult(0), *collapseReassoc);

    auto unPackOp = lowerUnsetEncodingToUnpackOp(
        rewriter, unsetEncodingOp, collapseShapeOp, *converter,
        this->materializeEncodingValueFn);
    if (failed(unPackOp)) {
      Value result = adaptor.getSource();
      Type targetType =
          getTypeConverter()->convertType(unsetEncodingOp.getResultType());
      if (targetType != result.getType()) {
        result = rewriter.create<tensor::CastOp>(unsetEncodingOp.getLoc(),
                                                 targetType, result);
      }
      rewriter.replaceOp(unsetEncodingOp, result);
      return success();
    }
    rewriter.replaceOp(unsetEncodingOp, unPackOp->getResult());
    return success();
  }
};

} // namespace

void GPUMaterializeDeviceEncodingPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  FunctionOpInterface funcOp = getOperation();
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
  {
    RewritePatternSet patterns(ctx);
    MaterializeEncodingTypeConverter typeConverter(materializeEncodingForTarget,
                                                   targetAttr);
    MaterializeEncodingConversionTarget target(*funcOp.getContext());
    MaterializeEncodingValueFn materializeEncodingValueFn =
        [](RankedTensorType, OpBuilder,
           Location) -> FailureOr<MaterializeEncodingValueInfo> { return {}; };
    populateIREEMaterializeEncodingIntoPackUnPackPatterns(
        patterns, target, typeConverter, materializeEncodingValueFn);

    patterns.insert<GPUSetEncodingOpLoweringConversion,
                    GPUUnsetEncodingOpLoweringConversion>(
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
