// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include "iree/compiler/Codegen/Common/EncodingUtils.h"
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
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-materialize-encoding"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUMATERIALIZEDEVICEENCODINGPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"


static llvm::cl::opt<std::string> clUnrollingFactor(
    "iree-data-tiling-unrolling-factor",
    llvm::cl::desc(
        "custom unrolling factor for data tiling"),
    llvm::cl::value_desc("unrolling factor"));

using namespace IREE::GPU;
using SmallVectorType = SmallVector<int64_t>;
// array component:
// {intrinsicVectorSize(roleIdx == 0 | 1),
//  intrinsicVectorSize(roleIdx == 2),
//  encodingTransposePerm(roleIdx == 1),
//  encodingTransposePerm(roleIdx == 2)}
static const std::unordered_map<MMAIntrinsic, std::array<SmallVectorType, 4>>
    mmaIntrinsicSizes = {
        {MMAIntrinsic::MFMA_F32_16x16x4_F32,
         {SmallVectorType{1, 1}, {4, 1}, {3, 0, 1, 4, 2}, {0, 3, 1, 2, 4}}},
        {MMAIntrinsic::MFMA_I32_32x32x16_I8,
         {SmallVectorType{1, 1}, {4, 1}, {3, 0, 1, 4, 2}, {0, 3, 1, 2, 4}}},
};

/// Returns the corresponding native tensor sizes defined by the `mma`
/// intrinsic.
static SmallVector<int64_t> getIntrinsicVectorSize(IREE::GPU::MMAIntrinsic mma,
                                                   int64_t roleIdx) {
  auto it = mmaIntrinsicSizes.find(mma);
  if (it == mmaIntrinsicSizes.end()) return {};
  return roleIdx == 0 || roleIdx == 1 ? it->second[0] : it->second[1];
  return {};
}

// Given encoding's role index and element types, return the transpose
// permutation used in GPU materialization.
static SmallVector<int64_t>
getEncodingTransposePerm(IREE::GPU::MMAIntrinsic mma, int64_t roleIdx) {
  auto it = mmaIntrinsicSizes.find(mma);
  if (it == mmaIntrinsicSizes.end()) return {};
  return roleIdx == 0 || roleIdx == 1 ? it->second[2] : it->second[3];
}

static std::optional<IREE::GPU::DataTiledMMAAttr>
enumerateMmaIntrinsic(TypeRange elementTypes, IREE::GPU::TargetAttr target) {
  assert(elementTypes.size() == 3);
  MLIRContext *ctx = target.getContext();

  for (IREE::GPU::MMAAttr mma : target.getWgp().getMma()) {
    IREE::GPU::MMAIntrinsic type = mma.getIntrinsic().getValue();
    
    // TODO: Drop this once all intrinsics are supported.
    if (type != IREE::GPU::MMAIntrinsic::MFMA_F32_16x16x4_F32 &&
        type != IREE::GPU::MMAIntrinsic::MFMA_I32_32x32x16_I8) {
      continue;
    }

    // make sure element types can match
    if (mma.getABCElementTypes() != std::tuple<Type, Type, Type>(
      elementTypes[0], elementTypes[1], elementTypes[2])) {
      continue;
    }
    return IREE::GPU::DataTiledMMAAttr::get(ctx, mma.getIntrinsic().getValue());
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
  auto intrinsicVectorSizes =
      getIntrinsicVectorSize(mma->getIntrinsic().getValue(), roleIdx);
  assert(!intrinsicVectorSizes.empty());
  auto permutation =
      getEncodingTransposePerm(mma->getIntrinsic().getValue(), roleIdx);
  assert(!permutation.empty());

  encodingInfo.innerTileShapes = intrinsicVectorSizes;
  auto mnkShape = mma->getMNKShape();
  encodingInfo.intrinsicSize = {std::get<0>(mnkShape), std::get<1>(mnkShape),
                                std::get<2>(mnkShape)};
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
    SmallVector<int64_t> expandShapeShape =
        getDataTilingTransposeDimensions<int64_t>(
            sourceShape, maybeEncodingInfo->innerTileSizes,
            intrinsicVectorShape);
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
    transposePerm.append({0, 1});
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

    SmallVector<int64_t> transposeSourceDims =
        getDataTilingTransposeDimensions<int64_t>(
            unpackSrcShape, maybeEncodingInfo->innerTileSizes,
            intrinsicVectorShape);

    size_t targetRank = unsetEncodingOp.getResultType().getRank();

    SmallVector<int64_t> transposePerm;
    transposePerm.append({0, 1});
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
        enumerateMmaIntrinsic(elementTypes, gpuTargetAttr);
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
