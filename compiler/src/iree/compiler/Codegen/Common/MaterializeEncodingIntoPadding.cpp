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
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_MATERIALIZEENCODINGINTOPADDINGPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

using namespace IREE::Encoding;

namespace {

// Returns the pad encoding layout, or nullptr if this is not the only layout or
// if there's no encoding at all.
static PadEncodingLayoutAttr getPadLayout(RankedTensorType type) {
  auto encoding =
      dyn_cast_or_null<IREE::Encoding::LayoutAttr>(type.getEncoding());
  if (!encoding) {
    return nullptr;
  }
  ArrayAttr layouts = encoding.getLayouts();
  if (!layouts || layouts.size() != 1) {
    return nullptr;
  }

  return dyn_cast<PadEncodingLayoutAttr>(*layouts.begin());
}

// Returns a padded tensor type (without encoding) for tensor types with the pad
// encoding layout, or the same type for all other tensors.
static RankedTensorType getPaddedType(RankedTensorType type) {
  PadEncodingLayoutAttr layout = getPadLayout(type);
  if (!isNonZeroPadding(layout)) {
    return type.dropEncoding();
  }

  ArrayRef<int32_t> padding = layout.getPadding().asArrayRef();
  auto newShape = llvm::to_vector_of<int64_t>(type.getShape());
  for (auto [newDim, padValue] : llvm::zip_equal(newShape, padding)) {
    assert((padValue == 0 || !ShapedType::isDynamic(newDim)) &&
           "Padding dynamic dims not supported");
    newDim += padValue;
  }

  return RankedTensorType::get(newShape, type.getElementType());
}

static bool hasNonZeroPadding(RankedTensorType type) {
  return isNonZeroPadding(getPadLayout(type));
}

struct MaterializePadEncodingTypeConverter final
    : MaterializeEncodingTypeConverter {
  MaterializePadEncodingTypeConverter(MLIRContext *ctx)
      : MaterializeEncodingTypeConverter(
            IREE::Codegen::EncodingNopLayoutAttr::get(ctx)) {
    addConversion([](RankedTensorType type) -> std::optional<RankedTensorType> {
      // The type converter is designed for `pad_encoding_layout` encoding
      // attribute. By the definition, the final converted type is the same
      // tensor type without encodings.
      return type.dropEncoding();
    });
    addConversion([&](IREE::Flow::DispatchTensorType dispatchTensorType)
                      -> IREE::Flow::DispatchTensorType {
      auto type = dyn_cast<RankedTensorType>(dispatchTensorType.getBoundType());
      if (!type) {
        return dispatchTensorType;
      }
      // The incoming bindings have the padded type, if `pad_encoding_layout` is
      // present.
      if (getPadLayout(type)) {
        type = getPaddedType(type);
      }
      return IREE::Flow::DispatchTensorType::get(dispatchTensorType.getAccess(),
                                                 type);
    });
  }
};

/// Pattern to convert `flow.dispatch.tensor.load` operation when
/// materializing the encoding. We extract a smaller tensor for the padded
/// source. This way we do not create partial loads prematurely, which would be
/// difficult to undo later on.
struct MaterializeFlowDispatchTensorLoadOp final
    : OpMaterializeEncodingPattern<IREE::Flow::DispatchTensorLoadOp> {
  using OpMaterializeEncodingPattern::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(IREE::Flow::DispatchTensorLoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle operations where the load covers the entire
    // `!flow.dispatch.tensor` type.
    if (!loadOp.isLoadOfWholeSource()) {
      return rewriter.notifyMatchFailure(loadOp, "unhandled partial loads");
    }

    IREE::Flow::DispatchTensorType sourceType = loadOp.getSourceType();
    auto boundTensorType = cast<RankedTensorType>(sourceType.getBoundType());
    if (!hasNonZeroPadding(boundTensorType)) {
      // Let the Nop pattern handle this.
      return rewriter.notifyMatchFailure(loadOp, "no padding applied");
    }

    auto &typeConverter =
        *getTypeConverter<MaterializePadEncodingTypeConverter>();
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
    Value newLoad = rewriter.create<IREE::Flow::DispatchTensorLoadOp>(
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

/// Pattern to convert `flow.dispatch.tensor.store` operation when
/// materializing the encoding. We create a larger empty tensor for the
/// destination and insert the value into it. This way we do not create partial
/// stores prematurely, which would be difficult to undo later on.
struct MaterializeFlowDispatchTensorStoreOp final
    : OpMaterializeEncodingPattern<IREE::Flow::DispatchTensorStoreOp> {
  using OpMaterializeEncodingPattern::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(IREE::Flow::DispatchTensorStoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle operations where the store covers the entire
    // `!flow.dispatch.tensor` type.
    if (!storeOp.isStoreToWholeTarget()) {
      return rewriter.notifyMatchFailure(storeOp, "unhandled partial stores");
    }

    IREE::Flow::DispatchTensorType targetType = storeOp.getTargetType();
    auto boundTensorType = cast<RankedTensorType>(targetType.getBoundType());
    if (!hasNonZeroPadding(boundTensorType)) {
      // Let the Nop pattern handle this.
      return rewriter.notifyMatchFailure(storeOp, "no padding applied");
    }

    auto &typeConverter =
        *getTypeConverter<MaterializePadEncodingTypeConverter>();
    IREE::Flow::DispatchTensorType newTargetType =
        typeConverter.convertType<IREE::Flow::DispatchTensorType>(targetType);
    RankedTensorType paddedType = newTargetType.asRankedTensorType();

    Location loc = storeOp.getLoc();
    SmallVector<Value> dynamicResultSizes{storeOp->getOperands()};
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

    rewriter.replaceOpWithNewOp<IREE::Flow::DispatchTensorStoreOp>(
        storeOp, insertOp, adaptor.getTarget(), newDynamicDims, offsets,
        newMixedSizes, strides);
    return success();
  }
};

/// Pattern to convert `flow.dispatch.tensor.store` operation when
/// materializing the encoding. We can not reuse the existing one because it
/// does not transform new dynamic dimension through interface. The other
/// difference is that the converted type of the padding attribute is not as the
/// same as the tensor type that drops encoding.
/// TODO(#20160): Abstract new interface methods and collapse two patterns.
struct MaterializeInterfaceBindingEncoding final
    : OpMaterializeEncodingPattern<IREE::HAL::InterfaceBindingSubspanOp> {
  using OpMaterializeEncodingPattern::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceBindingSubspanOp subspanOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<IREE::Flow::DispatchTensorType>(
        subspanOp.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          subspanOp, "expected result type to be !flow.dispatch.tensor");
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
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect,
                    IREE::Codegen::IREECodegenDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    FunctionOpInterface operation = getOperation();

    auto materializeEncodingValueFn =
        [](RankedTensorType, OpBuilder &,
           Location) -> FailureOr<MaterializeEncodingValueInfo> {
      return failure();
    };

    RewritePatternSet materializeEncodingPattern(context);
    MaterializePadEncodingTypeConverter typeConverter(context);
    MaterializeEncodingConversionTarget target(*context);
    populateMaterializeEncodingPatterns(materializeEncodingPattern, target,
                                        typeConverter,
                                        materializeEncodingValueFn);

    // The majority of this conversion is based on the 'Nop' materialization,
    // with the exception of a few ops that have to account for padding.
    // We add custom patterns with much higher priority to run before the
    // equivalent 'Nop' patterns.
    materializeEncodingPattern.add<MaterializeFlowDispatchTensorLoadOp,
                                   MaterializeFlowDispatchTensorStoreOp,
                                   MaterializeInterfaceBindingEncoding>(
        context, typeConverter, materializeEncodingValueFn,
        PatternBenefit{100});

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
      IREE::Flow::populateTensorSliceOpWithDispatchTensorOpFoldingPatterns(
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
