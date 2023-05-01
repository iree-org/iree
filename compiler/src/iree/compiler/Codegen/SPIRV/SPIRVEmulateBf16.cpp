// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file implements a pass to emulate 16-bit brain float operations with
// 32-bit ones.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-spirv-emulate-bf16"

namespace mlir {
namespace iree_compiler {
namespace {

class Bf16EmulationConverter : public TypeConverter {
 public:
  explicit Bf16EmulationConverter() {
    // Allow unknown types.
    addConversion([](Type ty) -> std::optional<Type> { return ty; });

    // Scalar case.
    addConversion([](FloatType ty) -> std::optional<Type> {
      if (ty.isBF16()) return IntegerType::get(ty.getContext(), 16);
      return ty;
    });

    addConversion([this](ShapedType ty) -> std::optional<Type> {
      return ty.clone(convertType(ty.getElementType()));
    });
  }
};

//===----------------------------------------------------------------------===//
// Rewrite patterns
//===----------------------------------------------------------------------===//
struct ConvertHalInterfaceBindingSubspan final
    : OpConversionPattern<IREE::HAL::InterfaceBindingSubspanOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceBindingSubspanOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type newResultTy = getTypeConverter()->convertType(op.getType());
    if (!newResultTy)
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          llvm::formatv("failed to legalize memref type: {0}", op.getType()));

    auto newOp =
        rewriter.replaceOpWithNewOp<IREE::HAL::InterfaceBindingSubspanOp>(
            op, newResultTy, adaptor.getSet(), adaptor.getBinding(),
            adaptor.getDescriptorType(), adaptor.getByteOffset(),
            adaptor.getDynamicDims(), adaptor.getAlignmentAttr(),
            adaptor.getDescriptorFlagsAttr());
    LLVM_DEBUG(llvm::dbgs() << "Bf16Emulation: new op: " << newOp << "\n");
    (void)newOp;
    return success();
  }
};

struct ConvertMemRefAlloc final : OpConversionPattern<memref::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::AllocOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getType());
    if (!newTy)
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          llvm::formatv("failed to convert memref type: {0}", op.getType()));

    rewriter.replaceOpWithNewOp<memref::AllocOp>(
        op, newTy, adaptor.getDynamicSizes(), adaptor.getSymbolOperands(),
        adaptor.getAlignmentAttr());
    return success();
  }
};

struct ConvertMemRefLoad final : OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::LoadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type newResTy = getTypeConverter()->convertType(op.getType());
    if (!newResTy)
      return rewriter.notifyMatchFailure(
          op->getLoc(), llvm::formatv("failed to convert memref type: {0}",
                                      op.getMemRefType()));

    rewriter.replaceOpWithNewOp<memref::LoadOp>(
        op, newResTy, adaptor.getMemref(), adaptor.getIndices(),
        op.getNontemporal());
    return success();
  }
};

struct ConvertMemRefStore final : OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::StoreOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getMemRefType());
    if (!newTy)
      return rewriter.notifyMatchFailure(
          op->getLoc(), llvm::formatv("failed to convert memref type: {0}",
                                      op.getMemRefType()));

    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        op, adaptor.getValue(), adaptor.getMemref(), adaptor.getIndices(),
        op.getNontemporal());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

std::optional<Value> materializeArithBitcast(OpBuilder &builder, Type resultTy,
                                             mlir::ValueRange inputs,
                                             mlir::Location loc) {
  return builder.create<arith::BitcastOp>(loc, resultTy, inputs);
}

static void populateIreeBf16EmulationPatterns(RewritePatternSet &patterns,
                                              TypeConverter &typeConverter) {
  patterns.add<ConvertHalInterfaceBindingSubspan, ConvertMemRefAlloc,
               ConvertMemRefLoad, ConvertMemRefStore>(typeConverter,
                                                      patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Main pass
//===----------------------------------------------------------------------===//

struct SPIRVEmulateBf16Pass final
    : public SPIRVEmulateBf16Base<SPIRVEmulateBf16Pass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }

  void runOnOperation() override {
    ModuleOp op = getOperation();
    MLIRContext *ctx = &getContext();

    Bf16EmulationConverter typeConverter;
    typeConverter.addTargetMaterialization(materializeArithBitcast);
    typeConverter.addSourceMaterialization(materializeArithBitcast);

    // Run the main emulation pass.
    {
      ConversionTarget target(*ctx);
      target.addLegalOp<arith::BitcastOp>();
      target.addLegalOp<func::ReturnOp>();
      target.addDynamicallyLegalOp<func::FuncOp>([&typeConverter](
                                                     Operation *op) {
        return typeConverter.isLegal(cast<func::FuncOp>(op).getFunctionType());
      });
      target.addDynamicallyLegalDialect<
          arith::ArithDialect, func::FuncDialect, IREE::HAL::HALDialect,
          memref::MemRefDialect, vector::VectorDialect>(
          [&typeConverter](Operation *op) {
            bool legal = typeConverter.isLegal(op);
            LLVM_DEBUG(if (!legal) llvm::dbgs()
                       << "Bf16Emulation: illegal op: " << *op << "\n");
            return legal;
          });

      RewritePatternSet patterns(ctx);
      arith::populateExpandBFloat16Patterns(patterns);
      populateIreeBf16EmulationPatterns(patterns, typeConverter);

      if (failed(applyPartialConversion(op, target, std::move(patterns))))
        signalPassFailure();
    }
  }
};

}  // namespace

//===----------------------------------------------------------------------===//
// Public interface
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<ModuleOp>> createSPIRVEmulateBf16Pass() {
  return std::make_unique<SPIRVEmulateBf16Pass>();
}

}  // namespace iree_compiler
}  // namespace mlir
