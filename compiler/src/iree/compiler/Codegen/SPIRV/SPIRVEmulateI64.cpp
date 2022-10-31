// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file implements a pass to emulate 64-bit integer operations with 32-bit
// ones.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/WideIntEmulationConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "iree-spirv-emulate-i64"

namespace mlir {
namespace iree_compiler {
namespace {

//===----------------------------------------------------------------------===//
// Conversion patterns
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
            adaptor.getDynamicDims(), adaptor.getAlignmentAttr());
    LLVM_DEBUG(llvm::dbgs()
               << "WideIntegerEmulation: new op: " << newOp << "\n");
    (void)newOp;
    return success();
  }
};

// TODO(kuhar): Upstream memref conversion patterns.
struct ConvertMemrefLoad final : OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::LoadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type newResTy = getTypeConverter()->convertType(op.getType());
    if (!newResTy)
      return rewriter.notifyMatchFailure(
          op->getLoc(), llvm::formatv("failed to legalize memref type: {0}",
                                      op.getMemRefType()));

    auto newOp = rewriter.replaceOpWithNewOp<memref::LoadOp>(
        op, newResTy, adaptor.getMemref(), adaptor.getIndices());
    LLVM_DEBUG(llvm::dbgs()
               << "WideIntegerEmulation: new op: " << newOp << "\n");
    (void)newOp;
    return success();
  }
};

// TODO(kuhar): Upstream memref conversion patterns.
struct ConvertMemrefStore final : OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::StoreOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getMemRefType());
    if (!newTy)
      return rewriter.notifyMatchFailure(
          op->getLoc(), llvm::formatv("failed to legalize memref type: {0}",
                                      op.getMemRefType()));

    auto newOp = rewriter.replaceOpWithNewOp<memref::StoreOp>(
        op, adaptor.getValue(), adaptor.getMemref(), adaptor.getIndices());
    LLVM_DEBUG(llvm::dbgs()
               << "WideIntegerEmulation: new op: " << newOp << "\n");
    (void)newOp;
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

// TODO(kuhar): Upstream memref conversion patterns.
static void populateMemrefEmulationPatterns(
    arith::WideIntEmulationConverter &converter, RewritePatternSet &patterns) {
  patterns.add<ConvertMemrefLoad, ConvertMemrefStore>(converter,
                                                      patterns.getContext());
}

// TODO(kuhar): Upstream memref conversion patterns.
static void populateMemrefWideIntegerEmulationConversions(
    arith::WideIntEmulationConverter &typeConverter) {
  typeConverter.addConversion(
      [&typeConverter](MemRefType ty) -> Optional<Type> {
        auto intTy = ty.getElementType().dyn_cast<IntegerType>();
        if (!intTy) return ty;

        if (intTy.getIntOrFloatBitWidth() <=
            typeConverter.getMaxTargetIntBitWidth())
          return ty;

        Type newElemTy = typeConverter.convertType(intTy);
        if (!newElemTy) return None;

        return ty.cloneWith(None, newElemTy);
      });
}

static void populateIreeI64EmulationPatterns(
    arith::WideIntEmulationConverter &converter, RewritePatternSet &patterns) {
  patterns.add<ConvertHalInterfaceBindingSubspan>(converter,
                                                  patterns.getContext());
}

static bool supportsI64(ModuleOp op) {
  spirv::TargetEnvAttr attr = getSPIRVTargetEnvAttr(op);
  assert(attr && "Not a valid spirv module");
  spirv::TargetEnv env(attr);
  return env.allows(spirv::Capability::Int64);
}

//===----------------------------------------------------------------------===//
// Main pass
//===----------------------------------------------------------------------===//

struct SPIRVEmulateI64Pass final
    : public SPIRVEmulateI64Base<SPIRVEmulateI64Pass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }

  void runOnOperation() override {
    ModuleOp op = getOperation();
    if (supportsI64(op)) return;

    arith::WideIntEmulationConverter typeConverter(32);
    populateMemrefWideIntegerEmulationConversions(typeConverter);

    MLIRContext *ctx = &getContext();
    ConversionTarget target(*ctx);
    target.addDynamicallyLegalOp<func::FuncOp>([&typeConverter](Operation *op) {
      return typeConverter.isLegal(cast<func::FuncOp>(op).getFunctionType());
    });
    target.addDynamicallyLegalDialect<
        arith::ArithDialect, func::FuncDialect, IREE::HAL::HALDialect,
        memref::MemRefDialect, vector::VectorDialect>(
        [&typeConverter](Operation *op) {
          bool legal = typeConverter.isLegal(op);
          LLVM_DEBUG(if (!legal) llvm::dbgs()
                     << "WideIntegerEmulation: illegal op: " << *op << "\n");
          return legal;
        });

    RewritePatternSet patterns(ctx);
    arith::populateArithWideIntEmulationPatterns(typeConverter, patterns);
    populateMemrefEmulationPatterns(typeConverter, patterns);
    populateIreeI64EmulationPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace

//===----------------------------------------------------------------------===//
// Public interface
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<ModuleOp>> createSPIRVEmulateI64Pass() {
  return std::make_unique<SPIRVEmulateI64Pass>();
}

}  // namespace iree_compiler
}  // namespace mlir
