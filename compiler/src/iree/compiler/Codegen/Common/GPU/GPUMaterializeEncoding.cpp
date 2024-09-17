// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "iree/compiler/Codegen/Common/GPU/GPUTileSwizzleUtils.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-materialize-encoding"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUMATERIALIZEDEVICEENCODINGPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

// Returns the index of the dimension whose flattened size (flattening inner
// dimensions into it) matches the given `targetSize`. This is used to compute
// interleaving indices.
//
// Example:
//    Input shape = [16, 8, 4, 4]
//    Input targetSize = 16
// -> Return 2, because the tail of the shape starting at index 2 is [4, 4],
//    whose product equals targetSize.
static int64_t getDimIdxForTargetSize(const SmallVector<int64_t> &shape,
                                      int64_t targetSize) {
  int interleaveAt = 0;
  int size = 1;
  for (interleaveAt = shape.size() - 1; interleaveAt >= 0; --interleaveAt) {
    assert(size <= targetSize);
    assert((targetSize % size) == 0);
    if (size == targetSize) {
      break;
    }
    size *= shape[interleaveAt];
  }
  return interleaveAt;
}

// Generates the swizzle for the full data-tiled-mma tile, including all the
// relevant unrolling factors.
static TileSwizzle getSwizzle(IREE::GPU::DataTiledMMAAttr mma,
                              IREE::GPU::MMAFragment fragment) {
  auto [AType, BType, CType] = mma.getABCElementTypes();
  int ABits = AType.getIntOrFloatBitWidth();
  int BBits = BType.getIntOrFloatBitWidth();
  // TODO(bjacob): Should be looked up from GPU target, instead of hard-coded.
  const int targetPreferredLoadBitWidth = 128;
  auto swizzle = getIntrinsicSwizzle(mma.getIntrinsic().getValue(), fragment);
  switch (fragment) {
  case IREE::GPU::MMAFragment::Lhs:
    // A-matrix (LHS). Source dimensions are M (index 0) and K (index 1).
    // Unroll on K with interleaving, then on M.
    if (mma.getUnrollK() > 1) {
      unroll(swizzle, 1, mma.getUnrollK());
      int interleavingIdx = getDimIdxForTargetSize(
          swizzle.expandShape[1],
          targetPreferredLoadBitWidth / (mma.getUnrollK() * ABits));
      interleave(swizzle, 1, interleavingIdx);
    }
    if (mma.getUnrollM() > 1) {
      unroll(swizzle, 0, mma.getUnrollM());
    }
    break;
  case IREE::GPU::MMAFragment::Rhs:
    // B-matrix (RHS). Since the pack ops already took care of transposing B,
    // source dimensions are N (index 0) and K (index 1).
    // Unroll on K with interleaving, then on N.
    if (mma.getUnrollK() > 1) {
      unroll(swizzle, 1, mma.getUnrollK());
      int interleavingIdx = getDimIdxForTargetSize(
          swizzle.expandShape[1],
          targetPreferredLoadBitWidth / (mma.getUnrollK() * BBits));
      interleave(swizzle, 1, interleavingIdx);
    }
    if (mma.getUnrollN() > 1) {
      unroll(swizzle, 0, mma.getUnrollN());
    }
    break;
  case IREE::GPU::MMAFragment::Acc:
    // C-matrix (accumulator). Source dimensions are M (index 0) and N (index
    // 1). Unroll on N, then on M.
    if (mma.getUnrollN() > 1) {
      unroll(swizzle, 1, mma.getUnrollN());
    }
    if (mma.getUnrollM() > 1) {
      unroll(swizzle, 0, mma.getUnrollM());
    }
    break;
  }
  return swizzle;
}

static bool hasIntrinsic(IREE::GPU::TargetAttr target,
                         IREE::GPU::MMAIntrinsic intrinsic) {
  for (IREE::GPU::MMAAttr mma : target.getWgp().getMma()) {
    if (mma.getIntrinsic().getValue() == intrinsic) {
      return true;
    }
  }
  return false;
}

static std::optional<IREE::GPU::DataTiledMMAAttr>
chooseDataTiledMMAAttr(TypeRange elementTypes, IREE::GPU::TargetAttr target) {
  assert(elementTypes.size() == 3);
  using namespace IREE::GPU;
  MLIRContext *ctx = target.getContext();
  Type lhs = elementTypes[0];
  Type rhs = elementTypes[1];
  Type out = elementTypes[2];
  auto match = [=](MMAIntrinsic intrinsic, int unrollM, int unrollN,
                   int unrollK) -> std::optional<DataTiledMMAAttr> {
    if (!hasIntrinsic(target, intrinsic)) {
      return std::nullopt;
    }
    auto candidate = DataTiledMMAAttr::get(
        ctx, MMAIntrinsicAttr::get(ctx, intrinsic), unrollM, unrollN, unrollK);
    auto [candidateLhs, candidateRhs, candidateOut] =
        candidate.getABCElementTypes();
    if (candidateLhs != lhs || candidateRhs != rhs || candidateOut != out) {
      return std::nullopt;
    }
    return candidate;
  };
  if (auto m = match(MMAIntrinsic::MFMA_F32_16x16x4_F32, 8, 8, 4)) {
    return m;
  }
  if (auto m = match(MMAIntrinsic::MFMA_F32_16x16x16_F16, 8, 8, 2)) {
    return m;
  }
  if (auto m = match(MMAIntrinsic::MFMA_I32_16x16x32_I8, 8, 8, 2)) {
    return m;
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
  std::optional<IREE::GPU::DataTiledMMAAttr> mma =
      chooseDataTiledMMAAttr(elementTypes, gpuTargetAttr);
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
  auto fragment =
      static_cast<IREE::GPU::MMAFragment>(encoding.getOperandIndex().getInt());
  encodingInfo.swizzle = getSwizzle(*mma, fragment);
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

SmallVector<ReassociationIndices>
getReassociationIndices(int outerDims,
                        SmallVector<SmallVector<int64_t>> expandShape) {
  SmallVector<ReassociationIndices> result;
  int expandedIdx = 0;
  for (int i = 0; i < outerDims; ++i) {
    result.push_back({expandedIdx++});
  }
  for (auto expandShapeDim : expandShape) {
    result.push_back({});
    for (int64_t d : expandShapeDim) {
      (void)d;
      result.back().push_back(expandedIdx++);
    }
  }
  return result;
}

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
    if (!maybeEncodingInfo->swizzle) {
      rewriter.replaceOp(encodingOp, packOp->getResult());
      return success();
    }
    SmallVector<int64_t> innerTiles = maybeEncodingInfo->innerTileSizes;

    // TODO(hanchung): Add a util to the encoding attribute, so we don't need
    // the map_to_vector method here.
    auto loc = encodingOp.getLoc();

    // Create expand_shape op to tile the innermost two dimensions.
    int origRank = encodingOp.getSourceType().getRank();
    SmallVector<int64_t> expandShapeShape(packOp->getDestType().getShape());
    expandShapeShape.truncate(origRank);
    expandShapeShape.append(
        getExpandedTileShape(maybeEncodingInfo->swizzle->expandShape));

    auto expandShapeType = RankedTensorType::get(
        expandShapeShape, encodingOp.getSourceType().getElementType());

    SmallVector<ReassociationIndices> reassociation = getReassociationIndices(
        origRank, maybeEncodingInfo->swizzle->expandShape);
    auto expandShapeOp = rewriter.create<tensor::ExpandShapeOp>(
        loc, expandShapeType, packOp->getResult(), reassociation);

    // create linalg.transpose on expandShapeShape

    SmallVector<int64_t> transposePerm;
    for (int i = 0; i < origRank; ++i) {
      transposePerm.push_back(i);
    }
    for (auto perm : maybeEncodingInfo->swizzle->permutation) {
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
    Value unpackSrc = adaptor.getSource();
    if (maybeEncodingInfo->swizzle) {
      SmallVector<int64_t> innerTiles = maybeEncodingInfo->innerTileSizes;

      int targetRank = unsetEncodingOp.getResultType().getRank();
      auto srcConvertedType =
          cast<RankedTensorType>(adaptor.getSource().getType());
      SmallVector<int64_t> expandShapeShape(srcConvertedType.getShape());
      expandShapeShape.truncate(targetRank);
      expandShapeShape.append(
          getExpandedTileShape(maybeEncodingInfo->swizzle->expandShape));

      SmallVector<int64_t> transposePerm;
      for (int i = 0; i < targetRank; ++i) {
        transposePerm.push_back(i);
      }
      for (auto perm : maybeEncodingInfo->swizzle->permutation) {
        transposePerm.push_back(targetRank + perm);
      }
      SmallVector<int64_t> expandShapeResultDims = expandShapeShape;
      applyPermutationToVector(expandShapeResultDims, transposePerm);
      auto invertedTransposePerm = invertPermutationVector(transposePerm);

      auto emptyTensor = rewriter.create<tensor::EmptyOp>(
          loc, expandShapeShape,
          unsetEncodingOp.getSourceType().getElementType());
      auto transposeOp = rewriter.create<linalg::TransposeOp>(
          loc, adaptor.getSource(), emptyTensor, invertedTransposePerm);

      SmallVector<ReassociationIndices> reassociation = getReassociationIndices(
          targetRank, maybeEncodingInfo->swizzle->expandShape);
      SmallVector<int64_t> unpackSrcShape(
          srcConvertedType.getShape().take_front(targetRank));
      unpackSrcShape.append(maybeEncodingInfo->innerTileSizes.begin(),
                            maybeEncodingInfo->innerTileSizes.end());
      auto unpackSrcType = RankedTensorType::get(
          unpackSrcShape, unsetEncodingOp.getSourceType().getElementType());
      unpackSrc = rewriter.create<tensor::CollapseShapeOp>(
          loc, unpackSrcType, transposeOp->getResult(0), reassociation);
    }

    auto unPackOp = lowerUnsetEncodingToUnpackOp(
        rewriter, unsetEncodingOp, unpackSrc, *converter,
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

class GPUConvertToMultiMma final
    : public OpInterfaceConversionPattern<linalg::ContractionOpInterface> {
public:
  using OpInterfaceConversionPattern<
      linalg::ContractionOpInterface>::OpInterfaceConversionPattern;

  GPUConvertToMultiMma(
      MLIRContext *context,
      const MaterializeEncodingTypeConverter &typeConverter,
      MaterializeEncodingValueFn materializeEncodingValueFn = {},
      PatternBenefit benefit = 1)
      : OpInterfaceConversionPattern<mlir::linalg::ContractionOpInterface>(
            typeConverter, context, benefit),
        materializeEncodingValueFn(materializeEncodingValueFn) {}

  LogicalResult
  matchAndRewrite(linalg::ContractionOpInterface op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto linalgOp = cast<linalg::LinalgOp>(op.getOperation());
    auto inputs = linalgOp.getDpsInputOperands();
    auto outputs = linalgOp.getDpsInits();
    auto lhsType = cast<RankedTensorType>(inputs[0]->get().getType());
    auto rhsType = cast<RankedTensorType>(inputs[1]->get().getType());
    auto resultType = cast<RankedTensorType>(outputs[0].getType());
    auto lhsEncoding = IREE::Encoding::getEncodingAttr(lhsType);
    auto rhsEncoding = IREE::Encoding::getEncodingAttr(rhsType);
    auto resultEncoding = IREE::Encoding::getEncodingAttr(resultType);
    if (!lhsEncoding || !rhsEncoding || !resultEncoding) {
      LLVM_DEBUG(llvm::dbgs() << "expect encodings on operand types\n");
      return failure();
    }

    auto converter = static_cast<const MaterializeEncodingTypeConverter *>(
        getTypeConverter());

    // TODO(hanchung): Perhaps the MaterializedEncodingInfo should carry the
    // target intrinsic attribute, so we don't need to query it again.
    IREE::HAL::ExecutableTargetAttr targetAttr = converter->getTargetAttr();
    IREE::GPU::TargetAttr gpuTargetAttr;
    if (targetAttr) {
      gpuTargetAttr = getGPUTargetAttr(targetAttr);
    } else {
      gpuTargetAttr = getCLGPUTarget(op.getContext());
    }
    auto elementTypes = llvm::to_vector(llvm::map_range(
        resultEncoding.getElementTypes().getValue(),
        [](Attribute a) { return cast<TypeAttr>(a).getValue(); }));
    std::optional<IREE::GPU::DataTiledMMAAttr> mma =
        chooseDataTiledMMAAttr(elementTypes, gpuTargetAttr);
    if (!mma) {
      LLVM_DEBUG(llvm::dbgs() << "can't find supported Mma intrinsic\n");
      return failure();
    }
    LLVM_DEBUG(llvm::dbgs() << "Target MMA: " << mma.value() << "\n");

    FailureOr<linalg::ContractionDimensions> contractionDims =
        linalg::inferContractionDims(linalgOp);
    assert(
        succeeded(contractionDims) &&
        "should always be able to infer contraction dims for contraction ops");
    // TODO(hanchung): Support batch gemms.
    if (!contractionDims->batch.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "batch gemm is not yet implemented\n");
      return failure();
    }

    // TODO(hanchung): Support unrolling cases. We likely need to teach
    // multi_mma op about interleaving K dimension.
    MLIRContext *ctx = rewriter.getContext();
    AffineExpr mExpr = rewriter.getAffineDimExpr(0);
    AffineExpr nExpr = rewriter.getAffineDimExpr(1);
    AffineExpr kExpr = rewriter.getAffineDimExpr(2);

    // The outer dims are all in row-major fasion after relayout.
    auto lhsMap = AffineMap::get(3, 0, {mExpr, kExpr}, ctx);
    auto rhsMap = AffineMap::get(3, 0, {nExpr, kExpr}, ctx);
    auto accMap = AffineMap::get(3, 0, {mExpr, nExpr}, ctx);

    SmallVector<utils::IteratorType> iteratorTypes =
        linalgOp.getIteratorTypesArray();

    // TODO(hanchung): Support batch gemms.
    Location loc = op.getLoc();
    auto mmaOp = rewriter.create<IREE::GPU::MultiMmaOp>(
        loc, operands[0], operands[1], operands[2],
        ArrayRef<AffineMap>{lhsMap, rhsMap, accMap}, iteratorTypes,
        mma.value());
    rewriter.replaceOp(op, mmaOp);
    return success();
  }

protected:
  const MaterializeEncodingValueFn materializeEncodingValueFn;
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
                    GPUUnsetEncodingOpLoweringConversion, GPUConvertToMultiMma>(
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
