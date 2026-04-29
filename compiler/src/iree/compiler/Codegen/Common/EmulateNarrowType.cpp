// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/EmulateNarrowType.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/NarrowTypeEmulationConverter.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_EMULATENARROWTYPEPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

struct ConvertHalInterfaceBindingSubspan final
    : OpConversionPattern<IREE::HAL::InterfaceBindingSubspanOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceBindingSubspanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto currentType = dyn_cast<MemRefType>(op.getType());
    if (!currentType) {
      return rewriter.notifyMatchFailure(op->getLoc(),
                                         "unhandled non-memref types");
    }
    auto newResultType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(currentType));
    if (!newResultType) {
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          llvm::formatv("failed to legalize memref type: {}", op.getType()));
    }
    Location loc = op.getLoc();
    OpFoldResult zero = rewriter.getIndexAttr(0);
    SmallVector<OpFoldResult> indices(currentType.getRank(), zero);

    // Get linearized type.
    int srcBits = currentType.getElementType().getIntOrFloatBitWidth();
    int dstBits = newResultType.getElementType().getIntOrFloatBitWidth();
    OpFoldResult elementOffset;
    Value byteOffset = adaptor.getByteOffset();
    if (byteOffset && !matchPattern(byteOffset, m_Zero())) {
      elementOffset = convertByteOffsetToElementOffset(
          rewriter, loc, byteOffset, currentType.getElementType());
    } else {
      elementOffset = rewriter.getIndexAttr(0);
    }
    SmallVector<OpFoldResult> sizes = getMixedValues(
        currentType.getShape(), adaptor.getDynamicDims(), rewriter);
    memref::LinearizedMemRefInfo linearizedMemRefInfo =
        memref::getLinearizedMemRefOffsetAndSize(rewriter, loc, srcBits,
                                                 dstBits, elementOffset, sizes);

    SmallVector<Value> dynamicLinearizedSize;
    if (newResultType.getRank() > 0 && !newResultType.hasStaticShape()) {
      dynamicLinearizedSize.push_back(getValueOrCreateConstantIndexOp(
          rewriter, loc, linearizedMemRefInfo.linearizedSize));
    }

    rewriter.replaceOpWithNewOp<IREE::HAL::InterfaceBindingSubspanOp>(
        op, newResultType, adaptor.getLayout(), adaptor.getBinding(),
        byteOffset, dynamicLinearizedSize, adaptor.getAlignmentAttr(),
        adaptor.getDescriptorFlagsAttr());
    return success();
  }
};

/// Conversion pattern for memref.extract_strided_metadata on a sub-byte
/// element-type source that was remapped by the type converter to a wider
/// (i8) container type.
///
/// When FunctionOpInterfaceAllBlocksSignatureConversion converts a block-arg of
/// type memref<...xfN, strided<..., offset:?>> to the i8 container type, it
/// inserts an unrealized_conversion_cast so that existing users of the old
/// block arg still see the emulated element type. ConvertVectorLoad then
/// creates an extract_strided_metadata on that cast result to obtain the
/// runtime offset for index linearization. Without this pattern,
/// extract_strided_metadata on the sub-byte source is illegal (its base-buffer
/// result has sub-byte element type).
///
/// This pattern:
///   1. Calls extract_strided_metadata on adaptor.getSource() (the i8 memref).
///   2. Scales the returned offset from i8 units to emulated-element units
///      (multiply by containerBits/emulatedBits).
///   3. Returns the emulated-element strides/sizes as constants derived
///      directly from the source MemRefType (they are always static in the
///      strided-memref patterns generated by narrow-type emulation).
///   4. Replaces the base-buffer result with the i8 base from step 1 cast
///      back to the emulated element base type, so downstream use of
///      base_buffer (if any) still type-checks.
struct ConvertExtractStridedMetadata final
    : OpConversionPattern<memref::ExtractStridedMetadataOp> {
  ConvertExtractStridedMetadata(const TypeConverter &converter,
                                MLIRContext *ctx, unsigned loadStoreBitwidth,
                                PatternBenefit benefit = 1)
      : OpConversionPattern(converter, ctx, benefit),
        loadStoreBitwidth(loadStoreBitwidth) {}

  LogicalResult
  matchAndRewrite(memref::ExtractStridedMetadataOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<MemRefType>(op.getSource().getType());
    if (!srcType) {
      return rewriter.notifyMatchFailure(op, "source is not a MemRefType");
    }

    // Only handle sub-byte element types that the type converter is converting.
    Type elemTy = srcType.getElementType();
    if (!elemTy.isIntOrFloat() ||
        elemTy.getIntOrFloatBitWidth() >= loadStoreBitwidth) {
      return rewriter.notifyMatchFailure(op, "source element not sub-byte");
    }

    unsigned emulatedBits = elemTy.getIntOrFloatBitWidth();
    unsigned containerBits = loadStoreBitwidth;

    Location loc = op.getLoc();
    Value convertedSrc = adaptor.getSource(); // i8 memref

    // Create extract_strided_metadata on the legal i8 source.
    auto i8Meta =
        memref::ExtractStridedMetadataOp::create(rewriter, loc, convertedSrc);

    // The offset returned by i8Meta is in i8 units. Scale it back to
    // emulated-element units: emulatedOffset = i8Offset * (containerBits /
    // emulatedBits).
    unsigned scale = containerBits / emulatedBits;
    Value emulatedOffset;
    int64_t srcStaticOffset;
    SmallVector<int64_t> srcStaticStrides;
    if (failed(
            srcType.getStridesAndOffset(srcStaticStrides, srcStaticOffset))) {
      return rewriter.notifyMatchFailure(op, "failed to get strides from type");
    }

    if (srcStaticOffset == ShapedType::kDynamic) {
      // Dynamic offset: scale the runtime value from i8Meta.
      Value scaleCst = arith::ConstantIndexOp::create(rewriter, loc, scale);
      emulatedOffset =
          arith::MulIOp::create(rewriter, loc, i8Meta.getOffset(), scaleCst);
    } else {
      emulatedOffset = arith::ConstantIndexOp::create(rewriter, loc,
                                                      srcStaticOffset * scale);
    }

    // Sizes from the original type (always static in our emulation patterns).
    SmallVector<Value> emulatedSizes;
    for (int64_t dim : srcType.getShape()) {
      emulatedSizes.push_back(
          arith::ConstantIndexOp::create(rewriter, loc, dim));
    }

    // Strides from the original type (always static in our emulation patterns).
    SmallVector<Value> emulatedStrides;
    for (int64_t stride : srcStaticStrides) {
      if (stride == ShapedType::kDynamic) {
        return rewriter.notifyMatchFailure(op, "dynamic stride not supported");
      }
      emulatedStrides.push_back(
          arith::ConstantIndexOp::create(rewriter, loc, stride));
    }

    // The base-buffer result: ConvertVectorLoad never uses the base-buffer
    // result of extract_strided_metadata (it only uses offset/sizes/strides),
    // so provide the i8 base directly. If any unexpected downstream user needs
    // the original emulated-element type, the conversion framework will insert
    // an unrealized_conversion_cast; since no actual user exists, the cast will
    // be absent and no unresolved-materialization error will occur.
    SmallVector<Value> results = {i8Meta.getBaseBuffer(), emulatedOffset};
    results.append(emulatedSizes);
    results.append(emulatedStrides);
    rewriter.replaceOp(op, results);
    return success();
  }

private:
  unsigned loadStoreBitwidth;
};

static void populateIreeNarrowTypeEmulationPatterns(
    arith::NarrowTypeEmulationConverter &converter,
    RewritePatternSet &patterns) {
  patterns.add<ConvertHalInterfaceBindingSubspan>(converter,
                                                  patterns.getContext());
  patterns.add<ConvertExtractStridedMetadata>(converter, patterns.getContext(),
                                              converter.getLoadStoreBitwidth());
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct EmulateNarrowTypePass final
    : impl::EmulateNarrowTypePassBase<EmulateNarrowTypePass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, cf::ControlFlowDialect,
                    func::FuncDialect, memref::MemRefDialect,
                    vector::VectorDialect, affine::AffineDialect,
                    IREE::Codegen::IREECodegenDialect, IREE::HAL::HALDialect>();
  }

  void runOnOperation() override {
    if (failed(emulateNarrowType(getOperation(), disableAtomicRMW))) {
      return signalPassFailure();
    }
  }
};
} // namespace

LogicalResult emulateNarrowType(
    Operation *root, bool disableAtomicRMW,
    std::optional<NarrowTypeConversionPopulationFn> populateCallback) {
  // The number of bits used in a load/store op.
  constexpr unsigned kLoadStoreEmulateBitwidth = 8;
  static_assert(
      llvm::isPowerOf2_32(kLoadStoreEmulateBitwidth) &&
      "only power of 2 is supported for narrow type load/store emulation");

  MLIRContext *ctx = root->getContext();

  // Resolve memref.dim ops before emulation. This is needed because the
  // emulation linearizes memrefs, changing their rank and shape semantics.
  // Any memref.dim on a narrow-type memref must be traced back to its source
  // dynamic dimensions before we can safely emulate.
  {
    RewritePatternSet dimPatterns(ctx);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(dimPatterns);
    GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    if (failed(applyPatternsGreedily(root, std::move(dimPatterns), config))) {
      return root->emitOpError("failed to resolve shaped type result dims");
    }
  }

  arith::NarrowTypeEmulationConverter typeConverter(kLoadStoreEmulateBitwidth);
  memref::populateMemRefNarrowTypeEmulationConversions(typeConverter);

  ConversionTarget target(*ctx);
  target.addDynamicallyLegalOp<func::FuncOp>([&typeConverter](Operation *op) {
    auto funcOp = cast<func::FuncOp>(op);
    if (!typeConverter.isLegal(funcOp.getFunctionType())) {
      return false;
    }
    // Also check that all block arguments in non-entry blocks are legal, so
    // that FunctionOpInterfaceAllBlocksSignatureConversion fires when needed.
    for (Block &block : funcOp.getFunctionBody()) {
      if (!typeConverter.isLegal(block.getArgumentTypes())) {
        return false;
      }
    }
    return true;
  });
  auto opLegalCallback = [&typeConverter](Operation *op) {
    return typeConverter.isLegal(op);
  };
  target.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>(opLegalCallback);
  target.addDynamicallyLegalDialect<
      arith::ArithDialect, vector::VectorDialect, memref::MemRefDialect,
      affine::AffineDialect, IREE::HAL::HALDialect>(opLegalCallback);

  RewritePatternSet patterns(ctx);

  // Try to flatten memrefs as a prerequisite for narrow type emulation,
  // so we can have simplified checks in the emulation patterns.
  memref::populateFlattenMemrefsPatterns(patterns);

  arith::populateArithNarrowTypeEmulationPatterns(typeConverter, patterns);
  memref::populateMemRefNarrowTypeEmulationPatterns(typeConverter, patterns,
                                                    disableAtomicRMW);
  populateIREEResolveExtractStridedMetadataPatterns(patterns);
  vector::populateVectorNarrowTypeEmulationPatterns(typeConverter, patterns,
                                                    disableAtomicRMW,
                                                    /*assumeAligned=*/true);
  populateIreeNarrowTypeEmulationPatterns(typeConverter, patterns);
  populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
  populateFunctionOpInterfaceAllBlocksTypeConversionPattern<func::FuncOp>(
      patterns, typeConverter);
  target.addDynamicallyLegalDialect<cf::ControlFlowDialect>([&typeConverter](
                                                                Operation *op) {
    return isLegalForBranchOpInterfaceTypeConversionPattern(op, typeConverter);
  });
  if (populateCallback) {
    populateCallback.value()(typeConverter, patterns, target);
  }

  if (failed(applyPartialConversion(root, target, std::move(patterns)))) {
    return root->emitOpError("failed to emulate bit width");
  }

  GreedyRewriteConfig config;
  config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Normal);

  RewritePatternSet sinkBroadcast(ctx);
  vector::populateSinkVectorOpsPatterns(sinkBroadcast);
  if (failed(applyPatternsGreedily(root, std::move(sinkBroadcast), config))) {
    return root->emitOpError("failed in sinking of broadcasts");
  }

  // Also do the `bitcast -> extui/extsi` rewrite.
  RewritePatternSet foldExtPatterns(ctx);
  vector::populateVectorNarrowTypeRewritePatterns(foldExtPatterns);
  if (failed(applyPatternsGreedily(root, std::move(foldExtPatterns), config))) {
    return failure();
  }
  return success();
}

} // namespace mlir::iree_compiler
