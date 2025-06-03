// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cassert>
#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "iree/compiler/Codegen/Common/PassUtils.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/TensorExt/Transforms/Transforms.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-codegen-materialize-encoding-into-padding"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_MATERIALIZEENCODINGINTOPADDINGPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

using namespace IREE::Encoding;

namespace {

// Returns the pad encoding layout, or nullptr if this is not the only layout or
// if there's no encoding at all.
static PadEncodingLayoutAttr getPadLayout(Attribute layoutAttr,
                                          RankedTensorType type) {
  if (!type.getEncoding()) {
    return nullptr;
  }
  auto encoding =
      dyn_cast_or_null<IREE::Encoding::LayoutAttr>(type.getEncoding());
  if (encoding) {
    ArrayAttr layouts = encoding.getLayouts();
    if (layouts.size() != 1) {
      return nullptr;
    }
    return dyn_cast<PadEncodingLayoutAttr>(*layouts.begin());
  }
  Attribute resolvedEncoding =
      cast<IREE::Encoding::LayoutResolverAttr>(layoutAttr).getLayout(type);
  LLVM_DEBUG({
    llvm::dbgs() << "Unresolved type: " << type << "\n";
    llvm::dbgs() << "layoutAttr: " << layoutAttr << "\n";
    llvm::dbgs() << "Resolved into: " << resolvedEncoding << "\n";
  });
  return dyn_cast<PadEncodingLayoutAttr>(resolvedEncoding);
}

// Returns a padded tensor type (without encoding) for tensor types with the pad
// encoding layout, or the same type for all other tensors.
static RankedTensorType getPaddedType(Attribute layoutAttr,
                                      RankedTensorType type) {
  PadEncodingLayoutAttr layout = getPadLayout(layoutAttr, type);
  if (layout.isIdentityLayout()) {
    return type.dropEncoding();
  }

  ArrayRef<int64_t> padding = layout.getPadding().asArrayRef();
  auto newShape = llvm::to_vector_of<int64_t>(type.getShape());
  for (auto [newDim, padValue] : llvm::zip_equal(newShape, padding)) {
    assert((padValue == 0 || !ShapedType::isDynamic(newDim)) &&
           "Padding dynamic dims not supported");
    newDim += padValue;
  }

  return RankedTensorType::get(newShape, type.getElementType());
}

struct MaterializePadEncodingTypeConverter final
    : MaterializeEncodingTypeConverter {
  MaterializePadEncodingTypeConverter(
      IREE::Encoding::LayoutMaterializerAttr layoutAttr)
      : MaterializeEncodingTypeConverter(layoutAttr) {
    addConversion([](RankedTensorType type) -> std::optional<RankedTensorType> {
      // The type converter is designed for `pad_encoding_layout` encoding
      // attribute. By the definition, the final converted type is the same
      // tensor type without encodings.
      return type.dropEncoding();
    });
    addConversion([&](IREE::TensorExt::DispatchTensorType dispatchTensorType)
                      -> IREE::TensorExt::DispatchTensorType {
      auto type = dyn_cast<RankedTensorType>(dispatchTensorType.getBoundType());
      if (!type || !type.getEncoding()) {
        return dispatchTensorType;
      }
      // The incoming bindings have the padded type, if `pad_encoding_layout` is
      // present.
      if (getPadLayout(getLayoutAttr(), type)) {
        type = getPaddedType(getLayoutAttr(), type);
      }
      return IREE::TensorExt::DispatchTensorType::get(
          dispatchTensorType.getAccess(), type);
    });
  }

  bool hasNonZeroPadding(RankedTensorType type) const {
    PadEncodingLayoutAttr layout = getPadLayout(getLayoutAttr(), type);
    return layout && !layout.isIdentityLayout();
  }
};

/// Pattern to convert `iree_tensor_ext.dispatch.tensor.load` operation when
/// materializing the encoding. We extract a smaller tensor for the padded
/// source. This way we do not create partial loads prematurely, which would be
/// difficult to undo later on.
struct MaterializeFlowDispatchTensorLoadOp final
    : OpConversionPattern<IREE::TensorExt::DispatchTensorLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::TensorExt::DispatchTensorLoadOp loadOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle operations where the load covers the entire
    // `!iree_tensor_ext.dispatch.tensor` type.
    if (!loadOp.isLoadOfWholeSource()) {
      return rewriter.notifyMatchFailure(loadOp, "unhandled partial loads");
    }

    auto &typeConverter =
        *getTypeConverter<MaterializePadEncodingTypeConverter>();
    IREE::TensorExt::DispatchTensorType sourceType = loadOp.getSourceType();
    auto boundTensorType = cast<RankedTensorType>(sourceType.getBoundType());
    if (!typeConverter.hasNonZeroPadding(boundTensorType)) {
      // Let the Nop pattern handle this.
      return rewriter.notifyMatchFailure(loadOp, "no padding applied");
    }

    auto paddedType =
        typeConverter.convertType<RankedTensorType>(boundTensorType);
    assert(paddedType != boundTensorType && "Expected conversion with padding");

    SmallVector<OpFoldResult> newMixedSizes =
        getMixedValues(paddedType.getShape(), loadOp.getSourceDims(), rewriter);

    SmallVector<OpFoldResult> newOffsets(newMixedSizes.size(),
                                         rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> newStrides(newMixedSizes.size(),
                                         rewriter.getIndexAttr(1));
    SmallVector<int64_t> newStaticDims;
    SmallVector<Value> newDynamicDims;
    dispatchIndexOpFoldResults(newMixedSizes, newDynamicDims, newStaticDims);

    Location loc = loadOp.getLoc();
    Value newLoad = rewriter.create<IREE::TensorExt::DispatchTensorLoadOp>(
        loc, adaptor.getSource(), newDynamicDims, newOffsets, newMixedSizes,
        newStrides);
    auto extractType = RankedTensorType::get(boundTensorType.getShape(),
                                             boundTensorType.getElementType());
    SmallVector<OpFoldResult> extractSizes = getMixedValues(
        boundTensorType.getShape(), loadOp.getSourceDims(), rewriter);
    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        loadOp, extractType, newLoad, newOffsets, extractSizes, newStrides);
    return success();
  }
};

/// Pattern to convert `iree_tensor_ext.dispatch.tensor.store` operation when
/// materializing the encoding. We create a larger empty tensor for the
/// destination and insert the value into it. This way we do not create partial
/// stores prematurely, which would be difficult to undo later on.
struct MaterializeFlowDispatchTensorStoreOp final
    : OpConversionPattern<IREE::TensorExt::DispatchTensorStoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::TensorExt::DispatchTensorStoreOp storeOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle operations where the store covers the entire
    // `!iree_tensor_ext.dispatch.tensor` type.
    if (!storeOp.isStoreToWholeTarget()) {
      return rewriter.notifyMatchFailure(storeOp, "unhandled partial stores");
    }

    auto &typeConverter =
        *getTypeConverter<MaterializePadEncodingTypeConverter>();
    IREE::TensorExt::DispatchTensorType targetType = storeOp.getTargetType();
    auto boundTensorType = cast<RankedTensorType>(targetType.getBoundType());
    if (!typeConverter.hasNonZeroPadding(boundTensorType)) {
      // Let the Nop pattern handle this.
      return rewriter.notifyMatchFailure(storeOp, "no padding applied");
    }

    IREE::TensorExt::DispatchTensorType newTargetType =
        typeConverter.convertType<IREE::TensorExt::DispatchTensorType>(
            targetType);
    RankedTensorType paddedType = newTargetType.asRankedTensorType();

    Location loc = storeOp.getLoc();
    SmallVector<Value> dynamicResultSizes{adaptor.getOperands()};
    Value empty =
        rewriter.create<tensor::EmptyOp>(loc, paddedType, dynamicResultSizes);

    SmallVector<OpFoldResult> offsets(paddedType.getRank(),
                                      rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(paddedType.getRank(),
                                      rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes =
        tensor::getMixedSizes(rewriter, loc, adaptor.getValue());
    Value insertOp = rewriter.create<tensor::InsertSliceOp>(
        loc, adaptor.getValue(), empty, offsets, sizes, strides);

    SmallVector<OpFoldResult> newMixedSizes = getMixedValues(
        paddedType.getShape(), storeOp.getTargetDims(), rewriter);
    SmallVector<int64_t> newStaticDims;
    SmallVector<Value> newDynamicDims;
    dispatchIndexOpFoldResults(newMixedSizes, newDynamicDims, newStaticDims);

    rewriter.replaceOpWithNewOp<IREE::TensorExt::DispatchTensorStoreOp>(
        storeOp, insertOp, adaptor.getTarget(), newDynamicDims, offsets,
        newMixedSizes, strides);
    return success();
  }
};

/// Pattern to convert `iree_tensor_ext.dispatch.tensor.store` operation when
/// materializing the encoding. We can not reuse the existing one because it
/// does not transform new dynamic dimension through interface. The other
/// difference is that the converted type of the padding attribute is not as the
/// same as the tensor type that drops encoding.
/// TODO(#20160): Abstract new interface methods and collapse two patterns.
struct MaterializeInterfaceBindingEncoding final
    : OpConversionPattern<IREE::HAL::InterfaceBindingSubspanOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceBindingSubspanOp subspanOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<IREE::TensorExt::DispatchTensorType>(
        subspanOp.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          subspanOp,
          "expected result type to be !iree_tensor_ext.dispatch.tensor");
    }
    auto newResultType = getTypeConverter()->convertType(resultType);
    SmallVector<Value> newDynamicDims = subspanOp.getDynamicDims();
    rewriter.replaceOpWithNewOp<IREE::HAL::InterfaceBindingSubspanOp>(
        subspanOp, newResultType, subspanOp.getLayout(), subspanOp.getBinding(),
        subspanOp.getByteOffset(), newDynamicDims, subspanOp.getAlignmentAttr(),
        subspanOp.getDescriptorFlagsAttr());
    return success();
  }
};

struct MaterializeEncodingIntoPaddingPass final
    : impl::MaterializeEncodingIntoPaddingPassBase<
          MaterializeEncodingIntoPaddingPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect, IREE::Codegen::IREECodegenDialect,
                    IREE::GPU::IREEGPUDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    FunctionOpInterface operation = getOperation();

    // Retrieve the config from executable target attribute, if any. Otherwise,
    // retrieve the config from CLI GPU target and construct a virtual
    // configuration.
    auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(operation);
    DictionaryAttr targetConfig;
    if (targetAttr) {
      targetConfig = targetAttr.getConfiguration();
    } else {
      IREE::GPU::TargetAttr gpuTargetAttr = getCLGPUTarget(context);
      SmallVector<NamedAttribute> items;
      items.emplace_back(
          IREE::Encoding::kEncodingResolverAttrName,
          IREE::GPU::getHIPTargetEncodingLayoutAttr(gpuTargetAttr, "pad"));
      targetConfig = DictionaryAttr::get(context, items);
    }

    // The layoutAttr should come in without any target info attached to it,
    // so we need to clone the layout attrs with the configuration so it can
    // access the target info during materialization.
    //
    // Otherwise, fall back to the nop layout.
    IREE::Encoding::LayoutMaterializerAttr layoutAttr;
    if (targetConfig &&
        targetConfig.contains(IREE::Encoding::kEncodingResolverAttrName)) {
      layoutAttr = targetConfig.getAs<IREE::Encoding::LayoutMaterializerAttr>(
          IREE::Encoding::kEncodingResolverAttrName);
      auto resolverAttr = cast<IREE::Encoding::LayoutResolverAttr>(layoutAttr);
      layoutAttr = cast<IREE::Encoding::LayoutMaterializerAttr>(
          resolverAttr.cloneWithSimplifiedConfig(targetConfig));
    } else {
      layoutAttr = cast<IREE::Encoding::LayoutMaterializerAttr>(
          IREE::Codegen::EncodingNopLayoutAttr::get(context));
    }

    RewritePatternSet materializeEncodingPattern(context);
    MaterializePadEncodingTypeConverter typeConverter(layoutAttr);
    MaterializeEncodingConversionTarget target(*context);
    populateMaterializeEncodingPatterns(materializeEncodingPattern, target,
                                        typeConverter);

    // The majority of this conversion is based on the 'Nop' materialization,
    // with the exception of a few ops that have to account for padding.
    // We add custom patterns with much higher priority to run before the
    // equivalent 'Nop' patterns.
    materializeEncodingPattern.add<MaterializeFlowDispatchTensorLoadOp,
                                   MaterializeFlowDispatchTensorStoreOp,
                                   MaterializeInterfaceBindingEncoding>(
        typeConverter, context, PatternBenefit{100});

    if (failed(applyPartialConversion(operation, target,
                                      std::move(materializeEncodingPattern)))) {
      operation.emitOpError("materialization failed");
      return signalPassFailure();
    }

    // Add patterns to resolve dims ops and cleanups.
    {
      RewritePatternSet patterns(context);
      memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
      context->getOrLoadDialect<tensor::TensorDialect>()
          ->getCanonicalizationPatterns(patterns);
      // TODO: Drop these when we deprecate partial loads/stores.
      IREE::TensorExt::populateTensorSliceOpWithDispatchTensorOpFoldingPatterns(
          patterns, context);
      if (failed(applyPatternsGreedily(operation, std::move(patterns)))) {
        operation.emitOpError("folding patterns failed");
        return signalPassFailure();
      }
    }
  }
};
} // namespace

void addEncodingToPaddingPasses(FunctionLikeNest &passManager) {
  passManager.addPass(createMaterializeEncodingIntoPaddingPass)
      .addPass(createBufferizeCopyOnlyDispatchesPass)
      .addPass(createCanonicalizerPass);
}

} // namespace mlir::iree_compiler
