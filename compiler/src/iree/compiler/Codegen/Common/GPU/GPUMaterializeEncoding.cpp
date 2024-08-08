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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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
std::optional<std::pair<int64_t, int64_t>>
getIntrinsicVectorSize(TypeRange elementTypes, int64_t roleIdx) {
  Type lhs = elementTypes[0];
  Type rhs = elementTypes[1];
  Type out = elementTypes[2];
  if (lhs.isF32() && rhs.isF32() && out.isF32()) {
    if (roleIdx == 0 || roleIdx == 1)
      return std::make_pair(1, 1);
    if (roleIdx == 2)
      return std::make_pair(4, 1);
  }
  return std::nullopt;
}

ArrayRef<int64_t> getTransposePermutation(int64_t roleIdx) {
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
    llvm_unreachable("unexpected roleIdx");
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
  return getEncodingInfoForMatmul(encoding, rank, enumeratedTileMxNxK[0]);
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

    // TODO(hanchung): Add a util to the encoding attribute, so we don't need
    // the map_to_vector method here.
    auto encoding = IREE::Encoding::getEncodingAttr(encodingOp.getResultType());
    int64_t roleIdx = encoding.getOperandIndex().getInt();
    auto elemTypes = llvm::map_to_vector(
        encoding.getElementTypes().getValue(),
        [](Attribute a) { return cast<TypeAttr>(a).getValue(); });
    auto loc = encodingOp.getLoc();
    std::optional<TileMxNxK> intrinsicShape = getIntrinsicSize(elemTypes);
    std::optional<std::pair<int64_t, int64_t>> intrinsicVectorShape =
        getIntrinsicVectorSize(elemTypes, roleIdx);
    if (!intrinsicShape || !intrinsicVectorShape) {
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

    // Check that the dimensions of the matrix can be divided by the tile shape,
    // if not then bail out.
    auto sourceType = encodingOp.getSourceType().getShape();
    assert(sourceType.size() == 2);
    if (sourceType[0] % innerTiles[0] == 0 ||
        sourceType[1] % innerTiles[1] == 0) {
      return failure();
    }

    // Create expand_shape
    llvm::SmallVector<int64_t> expandShapeShape;
    auto [iT1, iT2] = *intrinsicVectorShape;
    auto oT1 = sourceType[0] / iT1;
    auto oT2 = sourceType[1] / iT2;
    expandShapeShape = {oT1, iT1, oT2, iT2};
    assert(expandShapeShape.size() == 4);
    RankedTensorType expandShapeType =
        RankedTensorType::Builder(encodingOp.getSourceType())
            .setShape(expandShapeShape);
    Value expandShapeOp = rewriter.create<tensor::ExpandShapeOp>(
        loc, expandShapeType, packOp->getResult());

    // create linalg.transpose
    auto emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, expandShapeShape, encodingOp.getSourceType().getElementType());
    [[maybe_unused]] auto transposeOp = rewriter.create<linalg::TransposeOp>(
        loc, expandShapeOp, emptyTensor, getTransposePermutation(roleIdx));

    // TODO(hanchung): We want to make the shape consistent, so we need to
    // collpase and expand the shape. This is the shape we materialize for Flow
    // and HAL ops.
    // 1. Create tensor.collapse_shape.
    //    LHS: 4x16x1x1 -> 64
    //    ACC: 4x16x4x1 -> 256
    // 2. Create tensor.expand_shape to recover the shape (i.e., innerTiles).
    //    LHS: 64 -> 16x4 (innerTiles[0]xinnerTiles[1])
    //    ACC: 256 -> 16x16 (innerTiles[0]xinnerTiles[1])

    // TODO(lialan): Replace the op with the tensor.expand_shape op.
    rewriter.replaceOp(encodingOp, packOp->getResult());
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
