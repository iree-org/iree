// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/UKernelOps.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-expand-strided-metadata"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_IREEEXPANDSTRIDEDMETADATAPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {
/// Helper struct to return the offset, sizes and strides
/// of a `source` of a `memref.extract_strided_metadata` op.
struct DescriptorInfo {
  OpFoldResult offset;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
};
} // namespace

/// Returns an AffineMap for an add or a mul.
static AffineMap getAddMap(MLIRContext *context) {
  AffineExpr s0, s1;
  bindSymbols(context, s0, s1);
  return AffineMap::get(0, 2, s0 + s1);
}
static AffineMap getMulMap(MLIRContext *context) {
  AffineExpr s0, s1;
  bindSymbols(context, s0, s1);
  return AffineMap::get(0, 2, s0 * s1);
}

/// Returns the strides based on the sizes assuming that the `memref`
/// has default layout, i.e. it is not a result of a subview.
static SmallVector<OpFoldResult>
getStridesFromSizes(RewriterBase &rewriter, Location loc,
                    ArrayRef<OpFoldResult> sizes) {
  if (sizes.size() == 0) {
    return {};
  }
  SmallVector<OpFoldResult> strides(sizes.size());
  strides.back() = rewriter.getIndexAttr(1);
  if (sizes.size() == 1) {
    return strides;
  }
  AffineMap mulMap = getMulMap(rewriter.getContext());
  for (int i = sizes.size() - 2; i >= 0; --i) {
    strides[i] = affine::makeComposedFoldedAffineApply(
        rewriter, loc, mulMap, {strides[i + 1], sizes[i + 1]});
  }
  return strides;
}

static FailureOr<DescriptorInfo> resolveBufferDescriptorForInterfaceBinding(
    IREE::HAL::InterfaceBindingSubspanOp binding, RewriterBase &rewriter,
    Location loc) {
  auto memRefType = llvm::cast<MemRefType>(binding.getResult().getType());
  int rank = memRefType.getRank();
  DescriptorInfo resultDescriptor;

  // Compute sizes.
  auto dynamicDimIt = binding.getDynamicDims().begin();
  for (int i = 0; i < rank; ++i) {
    if (memRefType.isDynamicDim(i)) {
      resultDescriptor.sizes.push_back(*dynamicDimIt);
      dynamicDimIt++;
    } else {
      resultDescriptor.sizes.push_back(
          rewriter.getIndexAttr(memRefType.getDimSize(i)));
    }
  }
  // Strides.
  resultDescriptor.strides =
      getStridesFromSizes(rewriter, loc, resultDescriptor.sizes);

  // Offset.
  resultDescriptor.offset = convertByteOffsetToElementOffset(
      rewriter, loc, binding.getByteOffset(), memRefType.getElementType());
  return resultDescriptor;
}

/// Replaces the offsets, sizes and strides based on values provided
/// by `DescriptorInfo` object.
template <typename OpTy>
static void
replaceOffsetSizesAndStridesWith(RewriterBase &rewriter, OpTy op,
                                 const DescriptorInfo &resultDescriptor) {
  int rank = resultDescriptor.sizes.size();
  assert(rank == resultDescriptor.strides.size() &&
         "expected number of sizes and strides to match");
  assert(op.getSizes().size() == rank &&
         "expected as many size replacements as the number of sizes in the "
         "original operation");
  assert(op.getStrides().size() == rank &&
         "expected as many strides replacements as the number of strides in "
         "the original operation");
  Location loc = op.getLoc();
  for (int i = 0; i < rank; ++i) {
    // Sizes
    rewriter.replaceAllUsesWith(op.getSizes()[i],
                                getValueOrCreateConstantIndexOp(
                                    rewriter, loc, resultDescriptor.sizes[i]));
    // Strides
    rewriter.replaceAllUsesWith(
        op.getStrides()[i], getValueOrCreateConstantIndexOp(
                                rewriter, loc, resultDescriptor.strides[i]));
  }
  // Offset
  rewriter.replaceAllUsesWith(
      op.getOffset(),
      getValueOrCreateConstantIndexOp(rewriter, loc, resultDescriptor.offset));
}

namespace {

struct ResolveExtractMetadataFromHalInterfaceBindingSubspan
    : public OpRewritePattern<memref::ExtractStridedMetadataOp> {
  using OpRewritePattern<memref::ExtractStridedMetadataOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::ExtractStridedMetadataOp op,
                                PatternRewriter &rewriter) const override {
    auto binding =
        op.getSource()
            .template getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    if (!binding)
      return failure();
    auto memRefType = llvm::cast<MemRefType>(binding.getResult().getType());

    auto loc = op.getLoc();
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(binding);
    FailureOr<DescriptorInfo> resultDescriptor =
        resolveBufferDescriptorForInterfaceBinding(binding, rewriter, loc);
    if (failed(resultDescriptor)) {
      return rewriter.notifyMatchFailure(
          op, "failed to resolve descriptor with source being binding op");
    }

    bool bindsBasePointer =
        memRefType.getRank() == 0 && memRefType.getLayout().isIdentity();
    // For the base buffer of the `hal.interface.binding.subspan` create a 1D
    // buffer with zero offset. For example, if the
    // `hal.interface.binding.subspan` is
    //
    // ```mlir
    //  hal.interface.binding.subspan layout(#pipeline_layout)
    //  binding(1) offset(%offset)
    //      : memref<?x?xf32, strided<[?, 1], offset: 64]>>{%s0, %s1}
    // ```
    //
    // convert it to
    //
    // ```mlir
    //  #map = affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>
    //  %linearSize = affine.apply #map()[%offset, %s0, %s1]
    //  %c0 = arith.constant 0 : index
    //  hal.interface.binding.subspan layout(#pipeline_layout)
    //  binding(1) offset(%c0)
    //      : memref<?xf32>{%linearSize}
    // ```
    //
    // Only the base pointer of this subspan is needed, so creating a
    // subspan with zero offset (with original offset folded into the size)
    // is realistic representation of what the IR needs.
    AffineMap mulMap = getMulMap(rewriter.getContext());
    OpFoldResult linearizedMemrefSize = rewriter.getIndexAttr(1);
    for (auto size : resultDescriptor->sizes) {
      linearizedMemrefSize = affine::makeComposedFoldedAffineApply(
          rewriter, loc, mulMap, {linearizedMemrefSize, size});
    }
    AffineMap addMap = getAddMap(rewriter.getContext());
    linearizedMemrefSize = affine::makeComposedFoldedAffineApply(
        rewriter, loc, addMap,
        {linearizedMemrefSize, resultDescriptor->offset});

    SmallVector<int64_t> staticLinearShape;
    SmallVector<Value> dynamicLinearShape;
    dispatchIndexOpFoldResult(linearizedMemrefSize, dynamicLinearShape,
                              staticLinearShape);

    MemRefType newBufferType;
    IREE::HAL::InterfaceBindingSubspanOp newBinding;
    if (bindsBasePointer) {
      newBufferType = memRefType;
      newBinding = binding;
    } else {
      newBufferType = MemRefType::get(
          staticLinearShape, memRefType.getElementType(),
          MemRefLayoutAttrInterface(), memRefType.getMemorySpace());
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      newBinding = rewriter.create<IREE::HAL::InterfaceBindingSubspanOp>(
          loc, newBufferType, binding.getLayoutAttr(), binding.getBindingAttr(),
          zero, dynamicLinearShape, binding.getAlignmentAttr(),
          binding.getDescriptorFlagsAttr());
    }
    SmallVector<Value> results;
    results.reserve(memRefType.getRank() * 2 + 2);
    auto baseBufferType = llvm::cast<MemRefType>(op.getBaseBuffer().getType());
    if (!op.getBaseBuffer().use_empty()) {
      if (newBufferType == baseBufferType) {
        results.push_back(newBinding);
      } else {
        Value reinterpretCast = rewriter.create<memref::ReinterpretCastOp>(
            loc, baseBufferType, newBinding, /*offset=*/0,
            /*sizes=*/ArrayRef<int64_t>(),
            /*strides=*/ArrayRef<int64_t>());
        results.push_back(reinterpretCast);
      }
    } else {
      results.push_back(nullptr);
    }

    results.push_back(getValueOrCreateConstantIndexOp(
        rewriter, loc, resultDescriptor->offset));
    results.append(getValueOrCreateConstantIndexOp(rewriter, loc,
                                                   resultDescriptor->sizes));
    results.append(getValueOrCreateConstantIndexOp(rewriter, loc,
                                                   resultDescriptor->strides));

    rewriter.replaceOp(op, results);
    return success();
  }
};

// Converts IREE::Codegen::ExtractStridedMetadataOp to
// MemRef::ExtractStridedMetadataOp
struct ConvertCodegenIREEExtractMetadataToMemRef
    : public OpRewritePattern<IREE::Codegen::ExtractStridedMetadataOp> {
  using OpRewritePattern<
      IREE::Codegen::ExtractStridedMetadataOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::Codegen::ExtractStridedMetadataOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<memref::ExtractStridedMetadataOp>(
        op, op.getSource());
    return success();
  }
};

struct IREEExpandStridedMetadataPass final
    : impl::IREEExpandStridedMetadataPassBase<IREEExpandStridedMetadataPass> {
  using impl::IREEExpandStridedMetadataPassBase<
      IREEExpandStridedMetadataPass>::IREEExpandStridedMetadataPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    IREE::Codegen::IREECodegenDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void populateIREEResolveExtractStridedMetadataPatterns(
    RewritePatternSet &patterns) {
  memref::populateResolveExtractStridedMetadataPatterns(patterns);
  patterns.insert<ResolveExtractMetadataFromHalInterfaceBindingSubspan>(
      patterns.getContext());
  patterns.insert<ConvertCodegenIREEExtractMetadataToMemRef>(
      patterns.getContext());
}

void IREEExpandStridedMetadataPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  populateIREEResolveExtractStridedMetadataPatterns(patterns);
  populateRemoveDeadMemAllocPatterns(patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }

  if (!allowUnresolved) {
    SmallVector<Operation *> remaining;
    getOperation()->walk([&](Operation *op) {
      if (isa<memref::ExtractStridedMetadataOp>(op)) {
        remaining.push_back(op);
      }
    });

    if (!remaining.empty()) {
      auto diag = getOperation()->emitError()
                  << "Unable to resolve all strided buffer descriptors:";
      for (auto *op : remaining) {
        diag.attachNote(op->getLoc()) << "remaining live use";
      }
      signalPassFailure();
    }
  }
}
} // namespace mlir::iree_compiler
