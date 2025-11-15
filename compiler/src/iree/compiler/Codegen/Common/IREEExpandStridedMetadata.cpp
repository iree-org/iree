// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/UKernelOps.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "mlir/Dialect/AMDGPU/Transforms/Passes.h"
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

/// Walks up the def-use chain to find the HAL interface binding subspan
/// that a memref value derives from, passing through transparent wrapper ops.
static std::optional<IREE::HAL::InterfaceBindingSubspanOp>
getSourceInterfaceBinding(Value memrefValue) {
  Value source = memrefValue;
  // Walk through transparent wrapper operations.
  while (source) {
    if (auto binding =
            source.getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>()) {
      return binding;
    }
    if (auto assumeAlign = source.getDefiningOp<memref::AssumeAlignmentOp>()) {
      source = assumeAlign.getOperand();
      continue;
    }
    // No more transparent ops to pass through.
    break;
  }
  return std::nullopt;
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
  auto memRefType = cast<MemRefType>(binding.getResult().getType());
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

/// Converts memref.extract_strided_metadata to
/// iree_codegen.extract_strided_metadata when the source derives from a HAL
/// interface binding. This preserves SSA links through buffer binding
/// optimizations that update offsets.
struct ConvertMemRefExtractMetadataToIREECodegen
    : public OpRewritePattern<memref::ExtractStridedMetadataOp> {
  using OpRewritePattern<memref::ExtractStridedMetadataOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::ExtractStridedMetadataOp op,
                                PatternRewriter &rewriter) const override {
    if (!getSourceInterfaceBinding(op.getSource()))
      return failure();
    // Replace with iree_codegen version which doesn't fold.
    rewriter.replaceOpWithNewOp<IREE::Codegen::ExtractStridedMetadataOp>(
        op, op.getSource());
    return success();
  }
};

struct ResolveExtractMetadataFromHalInterfaceBindingSubspan
    : public OpRewritePattern<IREE::Codegen::ExtractStridedMetadataOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(IREE::Codegen::ExtractStridedMetadataOp op,
                                PatternRewriter &rewriter) const override {
    auto binding = getSourceInterfaceBinding(op.getSource());
    if (!binding)
      return failure();
    auto memRefType = cast<MemRefType>(binding->getResult().getType());

    auto loc = op.getLoc();
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(*binding);
    FailureOr<DescriptorInfo> resultDescriptor =
        resolveBufferDescriptorForInterfaceBinding(*binding, rewriter, loc);
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
      newBinding = *binding;
    } else {
      newBufferType = MemRefType::get(
          staticLinearShape, memRefType.getElementType(),
          MemRefLayoutAttrInterface(), memRefType.getMemorySpace());
      Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
      newBinding = IREE::HAL::InterfaceBindingSubspanOp::create(
          rewriter, loc, newBufferType, binding->getLayoutAttr(),
          binding->getBindingAttr(), zero, dynamicLinearShape,
          binding->getAlignmentAttr(), binding->getDescriptorFlagsAttr());
    }
    SmallVector<Value> results;
    results.reserve(memRefType.getRank() * 2 + 2);
    auto baseBufferType = cast<MemRefType>(op.getBaseBuffer().getType());
    if (!op.getBaseBuffer().use_empty()) {
      if (newBufferType == baseBufferType) {
        results.push_back(newBinding);
      } else {
        Value reinterpretCast = memref::ReinterpretCastOp::create(
            rewriter, loc, baseBufferType, newBinding, /*offset=*/0,
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

/// Converts iree_codegen.extract_strided_metadata to
/// memref.extract_strided_metadata. Only applies when the source is NOT from
/// a HAL binding (those are resolved by
/// ResolveExtractMetadataFromHalInterfaceBindingSubspan).
struct ConvertIREECodegenExtractMetadataToMemRef
    : public OpRewritePattern<IREE::Codegen::ExtractStridedMetadataOp> {
  using OpRewritePattern<
      IREE::Codegen::ExtractStridedMetadataOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::Codegen::ExtractStridedMetadataOp op,
                                PatternRewriter &rewriter) const override {
    // GUARD: Don't convert back to memref if source is still from HAL binding.
    // Pattern ResolveExtractMetadataFromHalInterfaceBindingSubspan must
    // resolve these first to preserve SSA links through buffer binding
    // optimizations.
    if (getSourceInterfaceBinding(op.getSource()))
      return failure();

    // Only convert ops that don't have HAL bindings (or are already resolved).
    rewriter.replaceOpWithNewOp<memref::ExtractStridedMetadataOp>(
        op, op.getSource());
    return success();
  }
};

struct IREEExpandStridedMetadataPass final
    : impl::IREEExpandStridedMetadataPassBase<IREEExpandStridedMetadataPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    IREE::Codegen::IREECodegenDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void populateIREEResolveExtractStridedMetadataPatterns(
    RewritePatternSet &patterns, bool allowSubviewExpansion) {
  if (allowSubviewExpansion) {
    memref::populateExpandStridedMetadataPatterns(patterns);
  } else {
    memref::populateResolveExtractStridedMetadataPatterns(patterns);
  }
  amdgpu::populateAmdgpuResolveStridedMetadataPatterns(patterns);

  // NOTE: the pattern benefits below are a secondary defense: each pattern
  // guards itself and only runs if required so they shouldn't be required but
  // since we're doing the back-and-forth this helps make it explicit. Ideally
  // we'd do this in two passes (fully resolve -> convert back to memref for
  // subsequent passes expecting it).

  // Convert memref extract ops to iree_codegen version to preserve SSA links.
  patterns.insert<ConvertMemRefExtractMetadataToIREECodegen>(
      patterns.getContext(), PatternBenefit(1));
  // Resolve iree_codegen extract ops from HAL bindings to concrete values.
  // Highest benefit ensures this runs before the fallback conversion.
  patterns.insert<ResolveExtractMetadataFromHalInterfaceBindingSubspan>(
      patterns.getContext(), PatternBenefit(3));
  // Convert remaining iree_codegen extract ops to memref for upstream
  // resolution. Lowest benefit ensures HAL binding resolution happens first.
  patterns.insert<ConvertIREECodegenExtractMetadataToMemRef>(
      patterns.getContext(), PatternBenefit(0));
}

void IREEExpandStridedMetadataPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  populateIREEResolveExtractStridedMetadataPatterns(patterns,
                                                    allowSubviewExpansion);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }

  if (!allowUnresolved) {
    SmallVector<Operation *> remaining;
    getOperation()->walk([&](Operation *op) {
      if (isa<memref::ExtractStridedMetadataOp,
              IREE::Codegen::ExtractStridedMetadataOp>(op)) {
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
