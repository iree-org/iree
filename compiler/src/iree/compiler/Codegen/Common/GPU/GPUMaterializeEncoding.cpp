// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPUTileSwizzleUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/HAL/Analysis/DeviceAnalysis.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Stream/Analysis/Affinity.h"
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
#define GEN_PASS_DEF_GPUMATERIALIZEHOSTENCODINGPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

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
  auto match = [=](MMAIntrinsic intrinsic, int unrollM, int unrollMToSubgroups,
                   int unrollN, int unrollNToSubgroups,
                   int unrollK) -> std::optional<DataTiledMMAAttr> {
    if (!hasIntrinsic(target, intrinsic)) {
      return std::nullopt;
    }
    auto candidate = DataTiledMMAAttr::get(
        ctx, MMAIntrinsicAttr::get(ctx, intrinsic), /*unroll_m=*/unrollM,
        /*unroll_m_to_subgroups=*/unrollMToSubgroups, /*unroll_n=*/unrollN,
        /*unroll_n_to_subgroups=*/unrollNToSubgroups, /*unroll_k=*/unrollK);
    auto [candidateLhs, candidateRhs, candidateOut] =
        candidate.getABCElementTypes();
    if (candidateLhs != lhs || candidateRhs != rhs || candidateOut != out) {
      return std::nullopt;
    }
    return candidate;
  };
  if (auto m = match(MMAIntrinsic::MFMA_F32_16x16x4_F32, 8, 1, 2, 4, 4)) {
    return m;
  }
  if (auto m = match(MMAIntrinsic::MFMA_F32_16x16x16_F16, 8, 1, 2, 4, 2)) {
    return m;
  }
  if (auto m = match(MMAIntrinsic::MFMA_I32_16x16x32_I8, 8, 1, 2, 4, 2)) {
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
  std::optional<IREE::GPU::DataTiledMMAAttr> mma =
      chooseDataTiledMMAAttr(encoding.getElementTypesArray(), gpuTargetAttr);
  if (!mma) {
    return failure();
  }

  // Map the matmul TileMxNxK to an actual tile shape for the tensor at hand,
  // based on its operand index in the matmul.
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

// TODO(hanchung): Delete this pass and rely on tensor-based analysis to
// materialize encodings based on where tensors are used. This pass is not able
// to handle that.
struct GPUMaterializeHostEncodingPass
    : public impl::GPUMaterializeHostEncodingPassBase<
          GPUMaterializeHostEncodingPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, tensor::TensorDialect,
                    linalg::LinalgDialect, IREE::Encoding::IREEEncodingDialect,
                    IREE::GPU::IREEGPUDialect>();
  }

  void runOnOperation() override;
};

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
                        const TileSwizzle::ExpandShapeType &expandShape) {
  SmallVector<ReassociationIndices> result;
  int expandedIdx = 0;
  for (int i = 0; i < outerDims; ++i) {
    result.push_back({expandedIdx++});
  }
  for (auto expandShapeDim : expandShape) {
    result.push_back({});
    for (int i = 0, e = expandShapeDim.size(); i < e; ++i) {
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
      Type targetType =
          getTypeConverter()->convertType(encodingOp.getResultType());
      Value result = rewriter.createOrFold<tensor::CastOp>(
          encodingOp.getLoc(), targetType, adaptor.getSource());
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

    Location loc = encodingOp.getLoc();

    // Create expand_shape op to tile the innermost two dimensions.
    int origRank = encodingOp.getSourceType().getRank();
    SmallVector<int64_t> expandShapeShape(
        packOp->getDestType().getShape().take_front(origRank));
    expandShapeShape.append(
        getExpandedTileShape(maybeEncodingInfo->swizzle->expandShape));
    RankedTensorType expandShapeType =
        encodingOp.getSourceType().clone(expandShapeShape);

    SmallVector<ReassociationIndices> reassociation = getReassociationIndices(
        origRank, maybeEncodingInfo->swizzle->expandShape);
    auto expandShapeOp = rewriter.create<tensor::ExpandShapeOp>(
        loc, expandShapeType, packOp->getResult(), reassociation);

    SmallVector<int64_t> transposePerm =
        llvm::to_vector(llvm::seq<int64_t>(0, origRank));
    for (auto perm : maybeEncodingInfo->swizzle->permutation) {
      transposePerm.push_back(origRank + perm);
    }
    SmallVector<OpFoldResult> transposeResultDims =
        tensor::getMixedSizes(rewriter, loc, expandShapeOp.getResult());
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

    FailureOr<MaterializeEncodingInfo> maybeEncodingInfo =
        converter->getEncodingInfo(unsetEncodingOp.getSource().getType());
    if (failed(maybeEncodingInfo)) {
      Type targetType =
          getTypeConverter()->convertType(unsetEncodingOp.getSourceType());
      Value result = rewriter.createOrFold<tensor::CastOp>(
          unsetEncodingOp.getLoc(), targetType, adaptor.getSource());
      rewriter.replaceOp(unsetEncodingOp, result);
      return success();
    }

    Location loc = unsetEncodingOp.getLoc();
    Value unpackSrc = adaptor.getSource();
    if (maybeEncodingInfo->swizzle) {
      int targetRank = unsetEncodingOp.getResultType().getRank();
      auto srcConvertedType =
          cast<RankedTensorType>(adaptor.getSource().getType());
      SmallVector<OpFoldResult> emptyShape =
          tensor::getMixedSizes(rewriter, loc, adaptor.getSource());
      emptyShape.resize(targetRank);
      for (auto i :
           getExpandedTileShape(maybeEncodingInfo->swizzle->expandShape)) {
        emptyShape.push_back(rewriter.getIndexAttr(i));
      }
      auto emptyTensor = rewriter.create<tensor::EmptyOp>(
          loc, emptyShape, unsetEncodingOp.getSourceType().getElementType());

      SmallVector<int64_t> transposePerm =
          llvm::to_vector(llvm::seq<int64_t>(0, targetRank));
      for (auto perm : maybeEncodingInfo->swizzle->permutation) {
        transposePerm.push_back(targetRank + perm);
      }
      auto invertedTransposePerm = invertPermutationVector(transposePerm);
      auto transposeOp = rewriter.create<linalg::TransposeOp>(
          loc, adaptor.getSource(), emptyTensor, invertedTransposePerm);

      SmallVector<ReassociationIndices> reassociation = getReassociationIndices(
          targetRank, maybeEncodingInfo->swizzle->expandShape);
      SmallVector<int64_t> unpackSrcShape(
          srcConvertedType.getShape().take_front(targetRank));
      unpackSrcShape.append(maybeEncodingInfo->innerTileSizes.begin(),
                            maybeEncodingInfo->innerTileSizes.end());
      RankedTensorType unpackSrcType =
          unsetEncodingOp.getResultType().clone(unpackSrcShape);
      unpackSrc = rewriter.create<tensor::CollapseShapeOp>(
          loc, unpackSrcType, transposeOp->getResult(0), reassociation);
    }

    auto unPackOp = lowerUnsetEncodingToUnpackOp(
        rewriter, unsetEncodingOp, unpackSrc, *converter,
        this->materializeEncodingValueFn);
    if (failed(unPackOp)) {
      Type targetType =
          getTypeConverter()->convertType(unsetEncodingOp.getResultType());
      Value result = rewriter.createOrFold<tensor::CastOp>(loc, targetType,
                                                           adaptor.getSource());
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
    std::optional<IREE::GPU::DataTiledMMAAttr> mma = chooseDataTiledMMAAttr(
        resultEncoding.getElementTypesArray(), gpuTargetAttr);
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

    MLIRContext *ctx = rewriter.getContext();
    SmallVector<AffineExpr> lhsExprs, rhsExprs, accExprs;
    int baseIdx = contractionDims->batch.empty() ? 0 : 1;
    if (baseIdx) {
      AffineExpr bExpr = rewriter.getAffineDimExpr(0);
      lhsExprs.push_back(bExpr);
      rhsExprs.push_back(bExpr);
      accExprs.push_back(bExpr);
    }
    AffineExpr mExpr = rewriter.getAffineDimExpr(baseIdx + 0);
    AffineExpr nExpr = rewriter.getAffineDimExpr(baseIdx + 1);
    AffineExpr kExpr = rewriter.getAffineDimExpr(baseIdx + 2);

    // The outer dims are all in row-major order after relayout.
    lhsExprs.append({mExpr, kExpr});
    rhsExprs.append({nExpr, kExpr});
    accExprs.append({mExpr, nExpr});
    int64_t numDims = baseIdx + 3;
    auto lhsMap = AffineMap::get(numDims, 0, lhsExprs, ctx);
    auto rhsMap = AffineMap::get(numDims, 0, rhsExprs, ctx);
    auto accMap = AffineMap::get(numDims, 0, accExprs, ctx);

    SmallVector<utils::IteratorType> iteratorTypes =
        linalgOp.getIteratorTypesArray();

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

static LogicalResult
materializeFuncOpEncodings(FunctionOpInterface funcOp,
                           IREE::HAL::ExecutableTargetAttr targetAttr) {
  MLIRContext *ctx = funcOp.getContext();
  {
    RewritePatternSet patterns(ctx);
    // On GPU, we use transposeNarrowN=false for a combination of reasons:
    // 1. As linalg.matmul materializes into iree_gpu.multi_mma, which inherits
    //    its semantics from the wrapped intrinsic, we can't rely on any kind of
    //    LHS<->RHS symmetry.
    // 2. We do not currently use ukernels, which would be one of the main areas
    //    to benefit from transposeNarrowN.
    // 3. Heuristics for cache-friendly dispatch tiling are internal to the GPU
    //    runtime, so we don't need a simplification at that level either.
    MaterializeEncodingTypeConverter typeConverter(
        materializeEncodingForTarget, targetAttr, /*transposeNarrowN=*/false);
    MaterializeEncodingConversionTarget target(*ctx);
    MaterializeEncodingValueFn materializeEncodingValueFn =
        [](RankedTensorType, OpBuilder,
           Location) -> FailureOr<MaterializeEncodingValueInfo> { return {}; };
    populateShapeIndependentMaterializeEncodingPatterns(
        patterns, target, typeConverter, materializeEncodingValueFn);

    patterns.insert<GPUSetEncodingOpLoweringConversion,
                    GPUUnsetEncodingOpLoweringConversion, GPUConvertToMultiMma>(
        ctx, typeConverter, materializeEncodingValueFn);

    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
      funcOp.emitOpError("materialization failed");
      return failure();
    }
  }

  // Add patterns to fold pack/unpack ops with pad/extract_slice ops and
  // resolve dims ops.
  {
    RewritePatternSet patterns(ctx);
    tensor::CastOp::getCanonicalizationPatterns(patterns, ctx);
    tensor::populateFoldIntoPackAndUnpackPatterns(patterns);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitOpError("folding patterns failed");
      return failure();
    }
  }

  return success();
}

static std::optional<SetVector<IREE::HAL::ExecutableTargetAttr>>
getFuncExecutableTargetAttrs(FunctionOpInterface funcOp,
                             IREE::Stream::AffinityAnalysis &affinityAnalysis,
                             IREE::HAL::DeviceAnalysis &deviceAnalysis) {
  // Get a set of all unique affinities used by resources within the function.
  SetVector<IREE::Stream::AffinityAttr> uniqueAffinityAttrs;
  SmallVector<IREE::Stream::AffinityAttr> lookupAffinityAttrs;
  funcOp.walk([&](Operation *op) {
    if (affinityAnalysis.tryLookupExecutionAffinity(op, lookupAffinityAttrs)) {
      uniqueAffinityAttrs.insert(lookupAffinityAttrs.begin(),
                                 lookupAffinityAttrs.end());
    }
    lookupAffinityAttrs.clear();
  });

  // Resolve affinities to executable targets.
  SetVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
  for (auto affinityAttr : uniqueAffinityAttrs) {
    deviceAnalysis.gatherRequiredExecutableTargets(affinityAttr, funcOp,
                                                   executableTargetAttrs);
  }
  return executableTargetAttrs;
}

} // namespace

void GPUMaterializeHostEncodingPass::runOnOperation() {
  auto moduleOp = getOperation();

  // Run required analysis passes.
  IREE::Stream::AffinityAnalysis affinityAnalysis(moduleOp);
  if (failed(affinityAnalysis.run())) {
    return signalPassFailure();
  }
  IREE::HAL::DeviceAnalysis deviceAnalysis(moduleOp);
  if (failed(deviceAnalysis.run())) {
    return signalPassFailure();
  }

  for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
    // Gather the required executable targets for the function. Note that it's
    // possible there are more required for ops nested within the function but
    // this pass is a hack and can't handle that :shrug:.
    auto executableTargets =
        getFuncExecutableTargetAttrs(funcOp, affinityAnalysis, deviceAnalysis);
    if (!executableTargets) {
      funcOp.emitOpError()
          << "could not determine executable targets for the function";
      return signalPassFailure();
    } else if (executableTargets->empty()) {
      // Probably no tensors.
      continue;
    }

    // HACK: this pass is run on the host _but shouldn't be_. Because it's
    // run on the host and IREE is a compiler capable of multi-targeting there
    // may be multiple executable targets at any point in the host program.
    // This pass can't handle that and assumes it's been checked earlier by
    // spooky action at a distance. This needs to be fixed.
    if (executableTargets->size() != 1) {
      funcOp.emitOpError() << "has multiple executable targets and CPU data "
                              "tiling isn't built to support that";
      return signalPassFailure();
    }

    // Materialize encodings within the function.
    if (failed(
            materializeFuncOpEncodings(funcOp, executableTargets->front()))) {
      return signalPassFailure();
    }
  }
}

void GPUMaterializeDeviceEncodingPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
  if (failed(materializeFuncOpEncodings(funcOp, targetAttr))) {
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler
