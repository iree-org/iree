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
    MemRefType origSrcType = op.getSrc().getType();
    MemRefType origDstType = op.getDst().getType();
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
    int64_t origSrcBits = origSrcType.getElementTypeBitWidth();
    int64_t newSrcBits = newSrcType.getElementTypeBitWidth();
    int64_t origDstBits = origDstType.getElementTypeBitWidth();
    int64_t newDstBits = newDstType.getElementTypeBitWidth();

    // Only convert when the transfer vector's total bits are a multiple of
    // the new element bit width. E.g. vector<3xf4E2M1FN> (12 bits) cannot
    // be cleanly packed into i8 elements.
    if (auto vecType = dyn_cast<VectorType>(op.getTransferType())) {
      int64_t totalBits =
          vecType.getNumElements() * vecType.getElementTypeBitWidth();
      if (totalBits % newSrcBits != 0) {
        return rewriter.notifyMatchFailure(
            op,
            "transfer vector bit-width is not a multiple of the new element "
            "bit width");
      }
    }

    // Check both source and destination convertibility before modifying IR.
    if (!canLinearizeAndPack(op.getSrcIndices(), origSrcType, origSrcBits,
                             newSrcBits)) {
      return rewriter.notifyMatchFailure(
          op, "failed to linearize source indices (dynamic or mismatched "
              "strides/offset, or invalid bit-width ratio)");
    }
    if (!canLinearizeAndPack(op.getDstIndices(), origDstType, origDstBits,
                             newDstBits)) {
      return rewriter.notifyMatchFailure(
          op, "failed to linearize destination indices (dynamic or mismatched "
              "strides/offset, or invalid bit-width ratio)");
    }

    // Linearize source indices into a 1D byte-offset index.
    Value srcIdx = linearizeAndPack(rewriter, loc, op.getSrcIndices(),
                                    origSrcType, origSrcBits, newSrcBits);

    // Linearize destination indices.
    Value dstIdx = linearizeAndPack(rewriter, loc, op.getDstIndices(),
                                    origDstType, origDstBits, newDstBits);

    // Adjust transfer type to use the new element type.
    Type newTransferType = convertTransferType(
        rewriter.getContext(), op.getTransferType(), origSrcBits, newSrcBits);

    amdgpu::GatherToLDSOp::create(
        rewriter, loc, adaptor.getSrc(), ValueRange{srcIdx}, adaptor.getDst(),
        ValueRange{dstIdx}, TypeAttr::get(newTransferType), op.getAsyncAttr());

    rewriter.eraseOp(op);
    return success();
  }

private:
  // Checks whether linearizeAndPack can succeed without modifying IR.
  static bool canLinearizeAndPack(ValueRange indices, MemRefType origType,
                                  int64_t origBits, int64_t newBits) {
    auto [strides, offset] = origType.getStridesAndOffset();
    if (ShapedType::isDynamic(offset)) {
      return false;
    }
    for (int64_t stride : strides) {
      if (ShapedType::isDynamic(stride)) {
        return false;
      }
    }
    if (indices.size() != strides.size()) {
      return false;
    }
    if (origBits != newBits &&
        (newBits <= origBits || newBits % origBits != 0)) {
      return false;
    }
    return true;
  }

  // Linearizes multi-dimensional indices into a 1D index for the packed
  // byte-addressable memref. The caller must ensure canLinearizeAndPack()
  // returns true before calling this.
  //   linearIdx = offset + sum(idx[i] * stride[i])
  //   packedIdx = linearIdx / (newBits / origBits)
  static Value linearizeAndPack(ConversionPatternRewriter &rewriter,
                                Location loc, ValueRange indices,
                                MemRefType origType, int64_t origBits,
                                int64_t newBits) {
    auto [strides, offset] = origType.getStridesAndOffset();

    // Linearize: offset + sum(idx[i] * stride[i]).
    auto overflowFlags =
        arith::IntegerOverflowFlags::nsw | arith::IntegerOverflowFlags::nuw;
    Value linearIdx = arith::ConstantIndexOp::create(rewriter, loc, offset);
    for (auto [idx, stride] : llvm::zip(indices, strides)) {
      Value strideVal = arith::ConstantIndexOp::create(rewriter, loc, stride);
      Value product =
          arith::MulIOp::create(rewriter, loc, idx, strideVal, overflowFlags);
      linearIdx = arith::AddIOp::create(rewriter, loc, linearIdx, product,
                                        overflowFlags);
    }

    // Pack: convert from origBits-element units to newBits-element units.
    if (origBits != newBits) {
      int64_t packRatio = newBits / origBits;
      Value ratioVal = arith::ConstantIndexOp::create(rewriter, loc, packRatio);
      linearIdx = arith::DivUIOp::create(rewriter, loc, linearIdx, ratioVal);
    }

    return linearIdx;
  }

  // Converts the transfer type from sub-byte elements to byte-sized elements,
  // preserving the total transfer size in bits. The caller must ensure
  // totalBits is a multiple of newBits (the op verifier enforces that
  // transfer sizes are 8, 16, 32, 96, or 128 bits, all multiples of 8).
  static Type convertTransferType(MLIRContext *context, Type origType,
                                  int64_t origBits, int64_t newBits) {
    if (auto vecType = dyn_cast<VectorType>(origType)) {
      int64_t totalBits =
          vecType.getNumElements() * vecType.getElementTypeBitWidth();
      assert(totalBits % newBits == 0 &&
             "transfer size must be a multiple of the new element bit width");
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
