// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
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
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

unsigned loadStoreEmulateBitwidth = 8;
unsigned arithComputeBitwidth = 4;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

struct ConvertHalInterfaceBindingSubspan final
    : OpConversionPattern<IREE::HAL::InterfaceBindingSubspanOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceBindingSubspanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newResultTy = getTypeConverter()->convertType(op.getType());
    if (!newResultTy)
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          llvm::formatv("failed to legalize memref type: {0}", op.getType()));

    rewriter.replaceOpWithNewOp<IREE::HAL::InterfaceBindingSubspanOp>(
        op, newResultTy, adaptor.getSet(), adaptor.getBinding(),
        adaptor.getDescriptorType(), adaptor.getByteOffset(),
        adaptor.getDynamicDims(), adaptor.getAlignmentAttr(),
        adaptor.getDescriptorFlagsAttr());
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
    if (!llvm::isPowerOf2_32(loadStoreEmulateBitwidth) ||
        loadStoreEmulateBitwidth < 8) {
      signalPassFailure();
      return;
    }

    ModuleOp op = getOperation();
    MLIRContext *ctx = &getContext();

    arith::NarrowTypeEmulationConverter typeConverter(loadStoreEmulateBitwidth);

    // Convert scalar type.
    typeConverter.addConversion([](IntegerType ty) -> std::optional<Type> {
      unsigned width = ty.getWidth();
      if (width < 2 || !llvm::isPowerOf2_32(width) ||
          width >= arithComputeBitwidth)
        return ty;

      return IntegerType::get(ty.getContext(), arithComputeBitwidth);
    });

    // Convert vector type.
    typeConverter.addConversion([](VectorType ty) -> std::optional<Type> {
      auto intTy = dyn_cast<IntegerType>(ty.getElementType());
      if (!intTy)
        return ty;

      unsigned width = intTy.getWidth();
      if (width < 2 || !llvm::isPowerOf2_32(width) ||
          width >= arithComputeBitwidth)
        return ty;

      return VectorType::get(
          to_vector(ty.getShape()),
          IntegerType::get(ty.getContext(), arithComputeBitwidth));
    });

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
        affine::AffineDialect, IREE::HAL::HALDialect>(
        [&typeConverter](Operation *op) { return typeConverter.isLegal(op); });

    RewritePatternSet patterns(ctx);

    arith::populateArithNarrowTypeEmulationPatterns(typeConverter, patterns);
    memref::populateMemRefNarrowTypeEmulationPatterns(typeConverter, patterns);
    vector::populateVectorNarrowTypeEmulationPatterns(typeConverter, patterns);
    populateIreeNarrowTypeEmulationPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
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
