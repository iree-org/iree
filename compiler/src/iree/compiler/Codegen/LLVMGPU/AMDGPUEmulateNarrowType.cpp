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
#include "mlir/Dialect/Arith/IR/Arith.h"
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
  using Base::Base;

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

/// Converts GatherToLDSOp when its memrefs change from sub-byte types
/// (e.g. f4E2M1FN) to byte-sized types (i8) during narrow type emulation.
/// The pattern linearizes multi-dimensional indices into the converted 1D
/// memref space and adjusts the transfer type accordingly.
struct ConvertGatherToLDS final : OpConversionPattern<amdgpu::GatherToLDSOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(amdgpu::GatherToLDSOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto origSrcType = cast<MemRefType>(op.getSrc().getType());
    auto origDstType = cast<MemRefType>(op.getDst().getType());
    auto newSrcType = cast<MemRefType>(adaptor.getSrc().getType());
    auto newDstType = cast<MemRefType>(adaptor.getDst().getType());

    // Only convert sub-byte element types.
    if (origSrcType.getElementTypeBitWidth() >= 8 &&
        origDstType.getElementTypeBitWidth() >= 8) {
      return failure();
    }

    // If types didn't change, nothing to do.
    if (newSrcType == origSrcType && newDstType == origDstType) {
      return failure();
    }

    Location loc = op.getLoc();
    int origSrcBits = origSrcType.getElementTypeBitWidth();
    int newSrcBits = newSrcType.getElementTypeBitWidth();
    int origDstBits = origDstType.getElementTypeBitWidth();
    int newDstBits = newDstType.getElementTypeBitWidth();

    // Only convert when the transfer vector's total bits are a multiple of
    // a byte. E.g. vector<3xf4E2M1FN> (12 bits) cannot be cleanly packed
    // into i8 elements.
    if (auto vecType = dyn_cast<VectorType>(op.getTransferType())) {
      int64_t totalBits =
          vecType.getNumElements() * vecType.getElementTypeBitWidth();
      if (totalBits % newSrcBits != 0) {
        return rewriter.notifyMatchFailure(
            op, "transfer vector bit-width is not a multiple of byte width");
      }
    }

    // Linearize source indices into a 1D byte-offset index.
    Value srcIdx = linearizeAndPack(rewriter, loc, op.getSrcIndices(),
                                    origSrcType, origSrcBits, newSrcBits);
    if (!srcIdx) {
      return rewriter.notifyMatchFailure(
          op, "failed to linearize source indices (dynamic strides)");
    }

    // Linearize destination indices.
    Value dstIdx = linearizeAndPack(rewriter, loc, op.getDstIndices(),
                                    origDstType, origDstBits, newDstBits);
    if (!dstIdx) {
      return rewriter.notifyMatchFailure(
          op, "failed to linearize destination indices (dynamic strides)");
    }

    // Adjust transfer type to use the new element type.
    Type newTransferType = convertTransferType(
        rewriter.getContext(), op.getTransferType(), origSrcBits, newSrcBits);

    auto newOp = amdgpu::GatherToLDSOp::create(
        rewriter, loc, adaptor.getSrc(), ValueRange{srcIdx}, adaptor.getDst(),
        ValueRange{dstIdx}, TypeAttr::get(newTransferType));
    if (op.getAsync()) {
      newOp.setAsync(true);
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  // Linearizes multi-dimensional indices into a 1D index for the packed
  // byte-addressable memref.
  //   linearIdx = sum(idx[i] * stride[i])
  //   packedIdx = linearIdx * origBits / newBits
  static Value linearizeAndPack(ConversionPatternRewriter &rewriter,
                                Location loc, ValueRange indices,
                                MemRefType origType, int origBits,
                                int newBits) {
    if (origBits == newBits) {
      // No packing needed; if also 1D, just pass through.
      if (indices.size() == 1) {
        return indices.front();
      }
    }

    auto [strides, offset] = origType.getStridesAndOffset();

    for (int64_t stride : strides) {
      if (ShapedType::isDynamic(stride)) {
        return nullptr;
      }
    }

    // Linearize: sum(idx[i] * stride[i]).
    Value linearIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
    for (auto [idx, stride] : llvm::zip(indices, strides)) {
      Value strideVal = arith::ConstantIndexOp::create(rewriter, loc, stride);
      Value product = arith::MulIOp::create(rewriter, loc, idx, strideVal);
      linearIdx = arith::AddIOp::create(rewriter, loc, linearIdx, product);
    }

    // Pack: convert from origBits-element units to newBits-element units.
    if (origBits != newBits) {
      assert(newBits > origBits && newBits % origBits == 0);
      int64_t packRatio = newBits / origBits;
      Value ratioVal = arith::ConstantIndexOp::create(rewriter, loc, packRatio);
      linearIdx = arith::DivUIOp::create(rewriter, loc, linearIdx, ratioVal);
    }

    return linearIdx;
  }

  // Converts the transfer type from sub-byte elements to byte-sized elements,
  // preserving the total transfer size in bits.
  static Type convertTransferType(MLIRContext *context, Type origType,
                                  int origBits, int newBits) {
    if (auto vecType = dyn_cast<VectorType>(origType)) {
      int64_t totalBits =
          vecType.getNumElements() * vecType.getElementTypeBitWidth();
      int64_t newElems = totalBits / newBits;
      return VectorType::get({newElems}, IntegerType::get(context, newBits));
    }
    return IntegerType::get(context, newBits);
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
          patterns.add<ConvertRawBufferCast, ConvertGatherToLDS>(
              typeConverter, patterns.getContext());
        };
    if (failed(emulateNarrowType(getOperation(), /*disableAtomic=*/true,
                                 populateAMDGPUPatterns))) {
      return signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler
