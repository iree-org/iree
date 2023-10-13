// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/NarrowTypeEmulationConverter.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

struct ConvertHalInterfaceBindingSubspan final
    : OpConversionPattern<IREE::HAL::InterfaceBindingSubspanOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceBindingSubspanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto currentType = op.getType().dyn_cast<MemRefType>();
    if (!currentType) {
      return rewriter.notifyMatchFailure(op->getLoc(),
                                         "unhandled non-memref types");
    }
    auto newResultType =
        getTypeConverter()->convertType(currentType).dyn_cast<MemRefType>();
    if (!newResultType) {
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          llvm::formatv("failed to legalize memref type: {0}", op.getType()));
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
        op, newResultType, adaptor.getSet(), adaptor.getBinding(),
        adaptor.getDescriptorType(), byteOffset, dynamicLinearizedSize,
        adaptor.getAlignmentAttr(), adaptor.getDescriptorFlagsAttr());
    return success();
  }
};

static void populateIreeNarrowTypeEmulationPatterns(
    arith::NarrowTypeEmulationConverter &converter,
    RewritePatternSet &patterns) {
  patterns.add<ConvertHalInterfaceBindingSubspan>(converter,
                                                  patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct EmulateNarrowTypePass
    : public EmulateNarrowTypeBase<EmulateNarrowTypePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, vector::VectorDialect,
                    affine::AffineDialect, IREE::HAL::HALDialect>();
  }

  void runOnOperation() override {
    // The number of bits used in a load/store op.
    constexpr unsigned kLoadStoreEmulateBitwidth = 8;
    static_assert(
        llvm::isPowerOf2_32(kLoadStoreEmulateBitwidth) &&
        "only power of 2 is supported for narrow type load/store emulation");

    MLIRContext *ctx = &getContext();

    arith::NarrowTypeEmulationConverter typeConverter(
        kLoadStoreEmulateBitwidth);
    memref::populateMemRefNarrowTypeEmulationConversions(typeConverter);

    ConversionTarget target(*ctx);
    target.addDynamicallyLegalOp<func::FuncOp>([&typeConverter](Operation *op) {
      return typeConverter.isLegal(cast<func::FuncOp>(op).getFunctionType());
    });
    auto opLegalCallback = [&typeConverter](Operation *op) {
      return typeConverter.isLegal(op);
    };
    target.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>(opLegalCallback);
    target.addDynamicallyLegalDialect<
        arith::ArithDialect, vector::VectorDialect, memref::MemRefDialect,
        affine::AffineDialect, IREE::HAL::HALDialect>(opLegalCallback);

    RewritePatternSet patterns(ctx);
    arith::populateArithNarrowTypeEmulationPatterns(typeConverter, patterns);
    memref::populateMemRefNarrowTypeEmulationPatterns(typeConverter, patterns);
    populateIREEResolveExtractStridedMetadataPatterns(ctx, patterns);
    vector::populateVectorNarrowTypeEmulationPatterns(typeConverter, patterns);
    populateIreeNarrowTypeEmulationPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      getOperation()->emitOpError("failed to emulate bit width");
      return signalPassFailure();
    }

    RewritePatternSet sinkBroadcast(ctx);
    vector::populateSinkVectorBroadcastPatterns(sinkBroadcast);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(sinkBroadcast)))) {
      getOperation()->emitOpError("failed in sinking of broadcasts");
      return signalPassFailure();
    }

    // Also do the `bitcast -> extui/extsi` rewrite.
    RewritePatternSet foldExtPatterns(ctx);
    vector::populateVectorNarrowTypeRewritePatterns(foldExtPatterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(foldExtPatterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Public interface
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<ModuleOp>> createEmulateNarrowTypePass() {
  return std::make_unique<EmulateNarrowTypePass>();
}

} // namespace iree_compiler
} // namespace mlir
