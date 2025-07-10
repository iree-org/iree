// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/EmulateNarrowType.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Transforms/NarrowTypeEmulationConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_AMDGPUEMULATENARROWTYPEPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

struct ConvertRawBufferCast final
    : OpConversionPattern<amdgpu::FatRawBufferCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(amdgpu::FatRawBufferCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!newTy) {
      return rewriter.notifyMatchFailure(
          op->getLoc(), llvm::formatv("failed to convert memref type: {0}",
                                      op.getResult().getType()));
    }
    if (newTy == op.getResult().getType()) {
      // Nothing to do.
      return failure();
    }

    // |validBytes| and |cacheSwizzleStride| are independent of element type
    // and don't need to be updated.
    rewriter.replaceOpWithNewOp<amdgpu::FatRawBufferCastOp>(
        op, newTy, adaptor.getSource(), adaptor.getValidBytes(),
        adaptor.getCacheSwizzleStride(), adaptor.getBoundsCheck(),
        adaptor.getResetOffset());
    return success();
  }
};

struct AMDGPUEmulateNarrowTypePass final
    : impl::AMDGPUEmulateNarrowTypePassBase<AMDGPUEmulateNarrowTypePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, vector::VectorDialect,
                    affine::AffineDialect, IREE::HAL::HALDialect>();
  }

  void runOnOperation() override {
    auto populateAMDGPUPatterns =
        [](arith::NarrowTypeEmulationConverter &typeConverter,
           RewritePatternSet &patterns, ConversionTarget &target) {
          auto opLegalCallback = [&typeConverter](Operation *op) {
            return typeConverter.isLegal(op);
          };
          target.addDynamicallyLegalDialect<amdgpu::AMDGPUDialect>(
              opLegalCallback);
          patterns.add<ConvertRawBufferCast>(typeConverter,
                                             patterns.getContext());
        };
    if (failed(emulateNarrowType(getOperation(), populateAMDGPUPatterns))) {
      return signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler
