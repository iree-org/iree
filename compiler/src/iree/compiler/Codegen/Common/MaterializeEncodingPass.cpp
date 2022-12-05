// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===---------------------------------------------------------------------===//
// Pass to materialize the encoding of tensor based on target information.
//===---------------------------------------------------------------------===//

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

using IREE::HAL::ExecutableTargetAttr;
using IREE::LinalgExt::MaterializeEncodingFn;
using IREE::LinalgExt::MaterializeEncodingInfo;
using IREE::LinalgExt::RuntimeTileSizeFn;
using IREE::LinalgExt::TensorEncoding;

/// For `dispatchTensorType` that bind a `RankedTensorType` with encoding,
/// returns the materialized shape of the `dispatchTensorType`. The
/// dynamic dimensions of the `dispatchTensorType` are provided in
/// `dynamicDims`.
static FailureOr<SmallVector<OpFoldResult>> getPackedDimsForDispatchTensor(
    OpBuilder &builder, Location loc,
    IREE::LinalgExt::MaterializeEncodingTypeConverter &typeConverter,
    IREE::Flow::DispatchTensorType dispatchTensorType, ValueRange dynamicDims,
    RuntimeTileSizeFn runtimeTileSizeFn) {
  auto boundTensorType =
      dispatchTensorType.getBoundType().dyn_cast<RankedTensorType>();
  if (!boundTensorType) {
    return failure();
  }

  auto encoding =
      boundTensorType.getEncoding().dyn_cast<IREE::LinalgExt::EncodingAttr>();
  if (!encoding) {
    return failure();
  }

  IREE::LinalgExt::MaterializeEncodingFn materializeEncodingFn =
      typeConverter.getEncodingFn();
  FailureOr<MaterializeEncodingInfo> materializeEncodingInfo =
      materializeEncodingFn(boundTensorType);
  if (failed(materializeEncodingInfo)) {
    return failure();
  }

  SmallVector<OpFoldResult, 4> targetShape =
      getMixedValues(dispatchTensorType.getShape(), dynamicDims, builder);
  auto innerTileSizes =
      getInnerTileSizesOfr(builder, loc, boundTensorType,
                           *materializeEncodingInfo, runtimeTileSizeFn);
  if (failed(innerTileSizes)) return failure();
  SmallVector<OpFoldResult> convertedTargetShape =
      IREE::LinalgExt::PackOp::getResultShape(
          builder, loc, targetShape, *innerTileSizes,
          materializeEncodingInfo->innerDimsPos,
          materializeEncodingInfo->outerDimsPerm);
  return convertedTargetShape;
}

/// For `dispatchTensorType` that bind a `RankedTensorType` with encoding,
/// returns the dynamic dimensions of the materialized shape of the
/// `dispatchTensorType`. The dynamic dimensions of the `dispatchTensorType` are
/// provided in `dynamicDims`.
static FailureOr<SmallVector<Value>> getPackedDynamicDimsForDispatchTensor(
    OpBuilder &builder, Location loc,
    IREE::LinalgExt::MaterializeEncodingTypeConverter &typeConverter,
    IREE::Flow::DispatchTensorType dispatchTensorType, ValueRange dynamicDims,
    RuntimeTileSizeFn runtimeTileSizeFn) {
  FailureOr<SmallVector<OpFoldResult>> convertedTargetShape =
      getPackedDimsForDispatchTensor(builder, loc, typeConverter,
                                     dispatchTensorType, dynamicDims,
                                     runtimeTileSizeFn);
  if (failed(convertedTargetShape)) {
    return failure();
  }
  SmallVector<int64_t> convertedStaticTargetShape;
  SmallVector<Value> convertedDynamicTargetShape;
  dispatchIndexOpFoldResults(convertedTargetShape.value(),
                             convertedDynamicTargetShape,
                             convertedStaticTargetShape, ShapedType::kDynamic);
  return convertedDynamicTargetShape;
}

namespace {
/// Extract encoding from the `tensorType` if specified.
static Optional<TensorEncoding> getEncoding(RankedTensorType tensorType) {
  auto encodingAttr = tensorType.getEncoding()
                          .dyn_cast_or_null<IREE::LinalgExt::EncodingAttr>();
  if (!encodingAttr) return llvm::None;
  return encodingAttr.getEncoding().getValue();
}

enum class MatmulType {
  F32F32F32,
  I8I8I32,
};

enum class MatmulOperandRole {
  LHS,
  RHS,
  RHS_TRANSPOSE,
  RESULT,
};

static Optional<MatmulType> getMatmulType(TensorEncoding encoding) {
  switch (encoding) {
    case TensorEncoding::MATMUL_F32F32F32_LHS:
    case TensorEncoding::MATMUL_F32F32F32_RHS:
    case TensorEncoding::MATMUL_F32F32F32_RHS_TRANSPOSE:
    case TensorEncoding::MATMUL_F32F32F32_RESULT:
      return MatmulType::F32F32F32;
    case TensorEncoding::MATMUL_I8I8I32_LHS:
    case TensorEncoding::MATMUL_I8I8I32_RHS:
    case TensorEncoding::MATMUL_I8I8I32_RHS_TRANSPOSE:
    case TensorEncoding::MATMUL_I8I8I32_RESULT:
      return MatmulType::I8I8I32;
    default:
      return llvm::None;
  }
}

static Optional<MatmulOperandRole> getMatmulOperandRole(
    TensorEncoding encoding) {
  switch (encoding) {
    case TensorEncoding::MATMUL_F32F32F32_LHS:
    case TensorEncoding::MATMUL_I8I8I32_LHS:
      return MatmulOperandRole::LHS;
    case TensorEncoding::MATMUL_F32F32F32_RHS:
    case TensorEncoding::MATMUL_I8I8I32_RHS:
      return MatmulOperandRole::RHS;
    case TensorEncoding::MATMUL_F32F32F32_RHS_TRANSPOSE:
    case TensorEncoding::MATMUL_I8I8I32_RHS_TRANSPOSE:
      return MatmulOperandRole::RHS_TRANSPOSE;
    case TensorEncoding::MATMUL_F32F32F32_RESULT:
    case TensorEncoding::MATMUL_I8I8I32_RESULT:
      return MatmulOperandRole::RESULT;
    default:
      return llvm::None;
  }
}

struct MatmulTileParams {
  int64_t M = 1;
  int64_t K = 1;
  int64_t N = 1;
};

// Generic fallback path, used outside of target architectures for which we have
// specific optimized tile shapes. This function might as well return only 1x1
// tiles as far as performance is concerned, but the current values are the ones
// that we had just before the actual target-aware logic was implemented, kept
// for compatibility with existing tests. These values are historically what we
// were using on AArch64+dotprod.
static MatmulTileParams chooseMatmulTileParamsGeneric() { return {8, 4, 8}; }

static MatmulTileParams chooseMatmulTileParamsAArch64(
    MatmulType type, ExecutableTargetAttr target) {
  switch (type) {
    case MatmulType::F32F32F32:
      return {8, 1, 8};
    case MatmulType::I8I8I32:
      if (hasFeature(target, "+i8mm")) return {8, 8, 8};
      if (hasFeature(target, "+dotprod")) return {8, 4, 8};
      return {8, 1, 8};
    default:
      assert(false);
      return {};
  }
}

static MatmulTileParams chooseMatmulTileParamsVMVXMicrokernels() {
  return {ShapedType::kDynamic, ShapedType::kDynamic, ShapedType::kDynamic};
}

static MatmulTileParams chooseMatmulTileParams(MatmulType type,
                                               ExecutableTargetAttr target) {
  if (isVMVXBackend(target) && hasMicrokernels(target)) {
    return chooseMatmulTileParamsVMVXMicrokernels();
  }
  if (isAArch64(target)) {
    return chooseMatmulTileParamsAArch64(type, target);
  }
  return chooseMatmulTileParamsGeneric();
}

static MaterializeEncodingInfo chooseEncodingInfoForMatmul(
    MatmulType type, MatmulOperandRole operandRole,
    ExecutableTargetAttr target) {
  MatmulTileParams tileParams = chooseMatmulTileParams(type, target);
  MaterializeEncodingInfo materializeEncodingInfo;
  materializeEncodingInfo.innerDimsPos = {0, 1};
  if (operandRole == MatmulOperandRole::LHS) {
    materializeEncodingInfo.innerTileSizes = {tileParams.M, tileParams.K};
  } else if (operandRole == MatmulOperandRole::RHS) {
    materializeEncodingInfo.innerTileSizes = {tileParams.K, tileParams.N};
  } else if (operandRole == MatmulOperandRole::RHS_TRANSPOSE) {
    materializeEncodingInfo.innerTileSizes = {tileParams.N, tileParams.K};
    materializeEncodingInfo.innerDimsPos = {1, 0};
    materializeEncodingInfo.outerDimsPerm = {1, 0};
  } else if (operandRole == MatmulOperandRole::RESULT) {
    materializeEncodingInfo.innerTileSizes = {tileParams.M, tileParams.N};
  } else {
    assert(false);  // already validated.
    return {};
  }
  return materializeEncodingInfo;
}

static void AdjustTileSizesToNarrowStaticShape(
    MaterializeEncodingInfo &materializeEncodingInfo, ArrayRef<int64_t> shape) {
  for (size_t i = 0; i < shape.size(); i++) {
    int64_t size = shape[materializeEncodingInfo.innerDimsPos[i]];
    // Dynamic sizes are assumed to be large enough, not to be candidates for
    // narrow kernels.
    if (ShapedType::isDynamic(size)) continue;
    int64_t &tileSize = materializeEncodingInfo.innerTileSizes[i];
    // TODO: find a way to do narrow tile size selection for dynamic tile sizes.
    if (ShapedType::isDynamic(tileSize)) continue;
    auto generateNarrowTileSize = [&](int64_t n) {
      if (size <= n && tileSize >= n) tileSize = n;
    };
    generateNarrowTileSize(1);
    generateNarrowTileSize(2);
    generateNarrowTileSize(4);
  }
}

static FailureOr<MaterializeEncodingInfo> chooseEncodingInfo(
    RankedTensorType tensorType, ExecutableTargetAttr target) {
  Optional<TensorEncoding> encoding = getEncoding(tensorType);
  if (!encoding) return failure();
  auto matmulType = getMatmulType(*encoding);
  auto matmulOperandRole = getMatmulOperandRole(*encoding);
  MaterializeEncodingInfo materializeEncodingInfo;
  if (matmulType && matmulOperandRole) {
    materializeEncodingInfo =
        chooseEncodingInfoForMatmul(*matmulType, *matmulOperandRole, target);
  } else {
    return failure();
  }
  AdjustTileSizesToNarrowStaticShape(materializeEncodingInfo,
                                     tensorType.getShape());
  return materializeEncodingInfo;
}

static FailureOr<SmallVector<Value>> chooseDynamicEncodingInfoVMVXMicrokernels(
    RankedTensorType tensorType, OpBuilder &builder, Location loc) {
  // For now just create Values equal to 1.
  // TODO: create a new vmvx.get_tile_sizes op here.
  Value v = builder.create<arith::ConstantIndexOp>(loc, 1);
  return SmallVector<Value>(tensorType.getRank(), v);
}

/// Pattern to materialize the encoding for `hal.interface.binding.subspan`
/// operations.
struct MaterializeInterfaceBindingEncoding
    : public IREE::LinalgExt::OpMaterializeEncodingPattern<
          IREE::HAL::InterfaceBindingSubspanOp> {
  using IREE::LinalgExt::OpMaterializeEncodingPattern<
      IREE::HAL::InterfaceBindingSubspanOp>::OpMaterializeEncodingPattern;

  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceBindingSubspanOp subspanOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto resultType = subspanOp.getResult()
                          .getType()
                          .dyn_cast<IREE::Flow::DispatchTensorType>();
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          subspanOp, "expected result type to be !flow.dispatch.tensor");
    }
    auto boundTensorType =
        resultType.getBoundType().dyn_cast<RankedTensorType>();
    if (!boundTensorType) {
      return rewriter.notifyMatchFailure(
          subspanOp, "bound type is not a RankedTensorType");
    }

    auto convertedBoundType = getTypeConverter()->convertType(boundTensorType);
    if (convertedBoundType == boundTensorType) {
      return rewriter.notifyMatchFailure(subspanOp, "bound type already valid");
    }

    auto *typeConverter =
        static_cast<IREE::LinalgExt::MaterializeEncodingTypeConverter *>(
            getTypeConverter());
    // Get the dynamic dims of the target.
    Location loc = subspanOp.getLoc();
    FailureOr<SmallVector<Value>> convertedDynamicDims =
        getPackedDynamicDimsForDispatchTensor(
            rewriter, loc, *typeConverter, resultType,
            subspanOp.getDynamicDims(), this->runtimeTileSizeFn);
    if (failed(convertedDynamicDims)) {
      return rewriter.notifyMatchFailure(
          subspanOp, "failed to get converted dynamic dims");
    }

    auto newResultType = IREE::Flow::DispatchTensorType::get(
        resultType.getAccess(), convertedBoundType);
    rewriter.replaceOpWithNewOp<IREE::HAL::InterfaceBindingSubspanOp>(
        subspanOp, newResultType, subspanOp.getSet(), subspanOp.getBinding(),
        subspanOp.getDescriptorType(), subspanOp.getByteOffset(),
        convertedDynamicDims.value(), subspanOp.getAlignmentAttr());
    return success();
  }
};

/// Pattern to convert `flow.dispatch.tensor.store` operation when
/// materializing the encoding.
struct MaterializeFlowDispatchTensorLoadOp
    : public IREE::LinalgExt::OpMaterializeEncodingPattern<
          IREE::Flow::DispatchTensorLoadOp> {
  using IREE::LinalgExt::OpMaterializeEncodingPattern<
      IREE::Flow::DispatchTensorLoadOp>::OpMaterializeEncodingPattern;

  LogicalResult matchAndRewrite(
      IREE::Flow::DispatchTensorLoadOp loadOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Only handle operations where the load covers the entire
    // `!flow.dispatch.tensor` type.
    // TODO(ravishankarm): Relax this for partial loads.
    if (!loadOp.isLoadOfWholeSource()) {
      return rewriter.notifyMatchFailure(loadOp, "unhandled partial loads");
    }

    auto sourceType = loadOp.getSourceType();
    auto boundTensorType = sourceType.getBoundType();
    auto *typeConverter =
        static_cast<IREE::LinalgExt::MaterializeEncodingTypeConverter *>(
            getTypeConverter());
    if (typeConverter->convertType(boundTensorType) == boundTensorType) {
      return rewriter.notifyMatchFailure(loadOp, "bound type already valid");
    }

    Location loc = loadOp.getLoc();
    FailureOr<SmallVector<OpFoldResult>> convertedMixedSizes =
        getPackedDimsForDispatchTensor(rewriter, loc, *typeConverter,
                                       sourceType, loadOp.getSourceDims(),
                                       this->runtimeTileSizeFn);
    if (failed(convertedMixedSizes)) {
      return rewriter.notifyMatchFailure(
          loadOp, "failed to get converted dynamic dims for result");
    }
    SmallVector<OpFoldResult> convertedOffsets(convertedMixedSizes->size(),
                                               rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> convertedStrides(convertedMixedSizes->size(),
                                               rewriter.getIndexAttr(1));
    SmallVector<int64_t> convertedStaticDims;
    SmallVector<Value> convertedDynamicDims;
    dispatchIndexOpFoldResults(convertedMixedSizes.value(),
                               convertedDynamicDims, convertedStaticDims,
                               ShapedType::kDynamic);
    rewriter.replaceOpWithNewOp<IREE::Flow::DispatchTensorLoadOp>(
        loadOp, adaptor.getSource(), convertedDynamicDims, convertedOffsets,
        convertedMixedSizes.value(), convertedStrides);

    return success();
  }
};

/// Pattern to convert `flow.dispatch.tensor.store` operation when
/// materializing the encoding.
struct MaterializeFlowDispatchTensorStoreOp
    : public IREE::LinalgExt::OpMaterializeEncodingPattern<
          IREE::Flow::DispatchTensorStoreOp> {
  using IREE::LinalgExt::OpMaterializeEncodingPattern<
      IREE::Flow::DispatchTensorStoreOp>::OpMaterializeEncodingPattern;

  LogicalResult matchAndRewrite(
      IREE::Flow::DispatchTensorStoreOp storeOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Only handle operations where the store covers the entire
    // `!flow.dispatch.tensor` type.
    // TODO(ravishankarm): Relax this for partial stores.
    if (!storeOp.isStoreToWholeTarget()) {
      return rewriter.notifyMatchFailure(storeOp, "unhandled partial stores");
    }

    auto targetType = storeOp.getTargetType();
    auto boundTensorType = targetType.getBoundType();
    auto *typeConverter =
        static_cast<IREE::LinalgExt::MaterializeEncodingTypeConverter *>(
            getTypeConverter());
    if (typeConverter->convertType(boundTensorType) == boundTensorType) {
      return rewriter.notifyMatchFailure(storeOp, "bound type already valid");
    }

    Location loc = storeOp.getLoc();
    FailureOr<SmallVector<OpFoldResult>> convertedMixedSizes =
        getPackedDimsForDispatchTensor(rewriter, loc, *typeConverter,
                                       targetType, storeOp.getTargetDims(),
                                       this->runtimeTileSizeFn);
    if (failed(convertedMixedSizes)) {
      return rewriter.notifyMatchFailure(
          storeOp, "failed to get converted dynamic dims for result");
    }
    SmallVector<OpFoldResult> convertedOffsets(convertedMixedSizes->size(),
                                               rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> convertedStrides(convertedMixedSizes->size(),
                                               rewriter.getIndexAttr(1));
    SmallVector<int64_t> convertedStaticDims;
    SmallVector<Value> convertedDynamicDims;
    dispatchIndexOpFoldResults(convertedMixedSizes.value(),
                               convertedDynamicDims, convertedStaticDims,
                               ShapedType::kDynamic);
    rewriter.replaceOpWithNewOp<IREE::Flow::DispatchTensorStoreOp>(
        storeOp, adaptor.getValue(), adaptor.getTarget(), convertedDynamicDims,
        convertedOffsets, convertedMixedSizes.value(), convertedStrides);
    return success();
  }
};

struct IREEMaterializeEncodingPass
    : public IREEMaterializeEncodingBase<IREEMaterializeEncodingPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, AffineDialect, IREE::Flow::FlowDialect,
                    IREE::LinalgExt::IREELinalgExtDialect>();
  }
  void runOnOperation() override;
};

}  // namespace

void IREEMaterializeEncodingPass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto operation = getOperation();

  {
    RewritePatternSet materializeEncodingPattern(context);
    auto targetAttr = ExecutableTargetAttr::lookup(operation);
    IREE::LinalgExt::MaterializeEncodingTypeConverter typeConverter(
        [targetAttr](RankedTensorType tensorType) {
          return chooseEncodingInfo(tensorType, targetAttr);
        });
    // Add type conversion for `!flow.dispatch.tensor` type.
    typeConverter.addConversion(
        [&typeConverter](IREE::Flow::DispatchTensorType dispatchTensorType) {
          Type boundType = dispatchTensorType.getBoundType();
          Type convertedBoundType = typeConverter.convertType(boundType);
          if (convertedBoundType == boundType) {
            return dispatchTensorType;
          }
          return IREE::Flow::DispatchTensorType::get(
              dispatchTensorType.getAccess(), convertedBoundType);
        });

    IREE::LinalgExt::MaterializeEncodingConversionTarget target(*context);
    target.addDynamicallyLegalOp<IREE::HAL::InterfaceBindingSubspanOp>(
        [&typeConverter](IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
          auto resultType =
              subspanOp.getResult()
                  .getType()
                  .template dyn_cast<IREE::Flow::DispatchTensorType>();
          // For types that are not `Flow::DispatchTensorType` mark as legal.
          if (!resultType) return true;
          return resultType == typeConverter.convertType(resultType);
        });

    IREE::LinalgExt::RuntimeTileSizeFn runtimeTileSizeFn;
    if (isVMVXBackend(targetAttr) && hasMicrokernels(targetAttr)) {
      runtimeTileSizeFn = chooseDynamicEncodingInfoVMVXMicrokernels;
    }
    IREE::LinalgExt::populateMaterializeEncodingPatterns(
        materializeEncodingPattern, target, typeConverter, runtimeTileSizeFn);
    materializeEncodingPattern.insert<MaterializeFlowDispatchTensorLoadOp,
                                      MaterializeFlowDispatchTensorStoreOp,
                                      MaterializeInterfaceBindingEncoding>(
        context, typeConverter, runtimeTileSizeFn);
    if (failed(applyPartialConversion(operation, target,
                                      std::move(materializeEncodingPattern)))) {
      operation.emitOpError("materialization failed");
      return signalPassFailure();
    }
  }

  // Add patterns to fold pack/unpack ops with pad/extract_slice ops and resolve
  // dims ops.
  {
    RewritePatternSet patterns(context);
    IREE::LinalgExt::populateFoldIntoPackAndUnpackOpsPatterns(patterns);
    memref::populateResolveRankedShapeTypeResultDimsPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(operation, std::move(patterns)))) {
      operation.emitOpError("folding patterns failed");
      return signalPassFailure();
    }
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
createIREEMaterializeEncodingPass() {
  return std::make_unique<IREEMaterializeEncodingPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
