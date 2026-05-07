// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Transforms.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmgpu-lower-1d-vector-ops"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPULOWER1DVECTOROPSPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

/// Converts 1D vector.multi_reduction directly to vector.reduction.
///
/// Example:
/// ```mlir
/// // Before
/// %r = vector.multi_reduction <add>, %v, %acc [0] : vector<Nxf32> to f32
///
/// // After
/// %r = vector.reduction <add>, %v, %acc : vector<Nxf32> into f32
/// ```
struct OneDimMultiReductionToReduction
    : public vector::MaskableOpRewritePattern<vector::MultiDimReductionOp> {
  using MaskableOpRewritePattern::MaskableOpRewritePattern;

  FailureOr<Value>
  matchAndRewriteMaskableOp(vector::MultiDimReductionOp multiReductionOp,
                            vector::MaskingOpInterface maskingOp,
                            PatternRewriter &rewriter) const override {
    auto srcRank = multiReductionOp.getSourceVectorType().getRank();
    if (srcRank != 1) {
      return failure();
    }

    if (!multiReductionOp.isReducedDim(0)) {
      return failure();
    }

    auto loc = multiReductionOp.getLoc();
    Value mask = maskingOp ? maskingOp.getMask() : Value();

    Operation *reductionOp = vector::ReductionOp::create(
        rewriter, loc, multiReductionOp.getKind(), multiReductionOp.getSource(),
        multiReductionOp.getAcc());

    if (mask) {
      reductionOp = mlir::vector::maskOperation(rewriter, reductionOp, mask);
    }

    return reductionOp->getResult(0);
  }
};

/// Collapse trailing unit dimensions from the memref source of a
/// vector.transfer_read, adjusting the permutation map accordingly.
/// E.g. transfer_read on memref<8x8x1x1x1x1xf32> with map
///   (d0,d1,d2,d3,d4,d5) -> (d1)
/// becomes transfer_read on memref<8x8xf32> with map (d0,d1) -> (d1).
struct CollapseTrailingUnitDimsTransferRead
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp read,
                                PatternRewriter &rewriter) const override {
    auto memRefType = dyn_cast<MemRefType>(read.getShapedType());
    if (!memRefType)
      return rewriter.notifyMatchFailure(read, "not a memref source");

    int64_t rank = memRefType.getRank();
    if (rank <= 1)
      return rewriter.notifyMatchFailure(read, "rank too low to collapse");

    ArrayRef<int64_t> shape = memRefType.getShape();
    int64_t numTrailingUnitDims = 0;
    for (int64_t i = rank - 1; i >= 0; --i) {
      if (shape[i] != 1)
        break;
      ++numTrailingUnitDims;
    }
    if (numTrailingUnitDims == 0)
      return rewriter.notifyMatchFailure(read, "no trailing unit dims");

    AffineMap map = read.getPermutationMap();
    int64_t firstUnitDim = rank - numTrailingUnitDims;
    for (AffineExpr expr : map.getResults()) {
      if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
        if (static_cast<int64_t>(dimExpr.getPosition()) >= firstUnitDim)
          return rewriter.notifyMatchFailure(
              read, "permutation map references a trailing unit dim");
      }
    }

    int64_t newRank = firstUnitDim;

    SmallVector<ReassociationIndices> reassoc;
    for (int64_t i = 0; i < newRank - 1; ++i)
      reassoc.push_back({i});
    ReassociationIndices lastGroup;
    for (int64_t i = newRank - 1; i < rank; ++i)
      lastGroup.push_back(i);
    reassoc.push_back(lastGroup);

    Location loc = read.getLoc();
    auto collapsed = memref::CollapseShapeOp::create(rewriter, loc,
                                                     read.getBase(), reassoc);

    AffineMap newMap =
        AffineMap::get(newRank, 0, map.getResults(), map.getContext());

    SmallVector<Value> newIndices(read.getIndices().begin(),
                                 read.getIndices().begin() + newRank);

    SmallVector<bool> newInBounds;
    if (read.getInBoundsAttr()) {
      for (int64_t i = 0; i < static_cast<int64_t>(
                                   read.getInBoundsAttr().size()); ++i) {
        newInBounds.push_back(
            cast<BoolAttr>(read.getInBoundsAttr()[i]).getValue());
      }
    }

    rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
        read, read.getVectorType(), collapsed, newIndices,
        AffineMapAttr::get(newMap), read.getPadding(), read.getMask(),
        newInBounds.empty() ? ArrayAttr()
                            : rewriter.getBoolArrayAttr(newInBounds));
    return success();
  }
};

/// Collapse trailing unit dimensions from the memref dest of a
/// vector.transfer_write.
struct CollapseTrailingUnitDimsTransferWrite
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp write,
                                PatternRewriter &rewriter) const override {
    auto memRefType = dyn_cast<MemRefType>(write.getShapedType());
    if (!memRefType)
      return rewriter.notifyMatchFailure(write, "not a memref dest");

    int64_t rank = memRefType.getRank();
    if (rank <= 1)
      return rewriter.notifyMatchFailure(write, "rank too low to collapse");

    ArrayRef<int64_t> shape = memRefType.getShape();
    int64_t numTrailingUnitDims = 0;
    for (int64_t i = rank - 1; i >= 0; --i) {
      if (shape[i] != 1)
        break;
      ++numTrailingUnitDims;
    }
    if (numTrailingUnitDims == 0)
      return rewriter.notifyMatchFailure(write, "no trailing unit dims");

    AffineMap map = write.getPermutationMap();
    int64_t firstUnitDim = rank - numTrailingUnitDims;
    for (AffineExpr expr : map.getResults()) {
      if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
        if (static_cast<int64_t>(dimExpr.getPosition()) >= firstUnitDim)
          return rewriter.notifyMatchFailure(
              write, "permutation map references a trailing unit dim");
      }
    }

    int64_t newRank = firstUnitDim;

    SmallVector<ReassociationIndices> reassoc;
    for (int64_t i = 0; i < newRank - 1; ++i)
      reassoc.push_back({i});
    ReassociationIndices lastGroup;
    for (int64_t i = newRank - 1; i < rank; ++i)
      lastGroup.push_back(i);
    reassoc.push_back(lastGroup);

    Location loc = write.getLoc();
    auto collapsed = memref::CollapseShapeOp::create(rewriter, loc,
                                                     write.getBase(), reassoc);

    AffineMap newMap =
        AffineMap::get(newRank, 0, map.getResults(), map.getContext());

    SmallVector<Value> newIndices(write.getIndices().begin(),
                                 write.getIndices().begin() + newRank);

    SmallVector<bool> newInBounds;
    if (write.getInBoundsAttr()) {
      for (int64_t i = 0; i < static_cast<int64_t>(
                                    write.getInBoundsAttr().size()); ++i) {
        newInBounds.push_back(
            cast<BoolAttr>(write.getInBoundsAttr()[i]).getValue());
      }
    }

    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        write, write.getVector(), collapsed, newIndices,
        AffineMapAttr::get(newMap), write.getMask(),
        newInBounds.empty() ? ArrayAttr()
                            : rewriter.getBoolArrayAttr(newInBounds));
    return success();
  }
};

/// Lower `vector.transfer_read` to `vector.load` (unmasked) or
/// `vector.maskedload` (masked). Requires minor-identity permutation map,
/// unit stride on the most minor memref dimension, and no out-of-bounds dims.
struct TransferReadToVectorLoadLowering
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp read,
                                PatternRewriter &rewriter) const override {
    if (read.getVectorType().getRank() > 1) {
      return rewriter.notifyMatchFailure(read, "expected rank-1 or rank-0 vector");
    }

    if (!read.getPermutationMap().isMinorIdentity()) {
      return rewriter.notifyMatchFailure(read, "not minor identity map");
    }

    auto memRefType = dyn_cast<MemRefType>(read.getShapedType());
    if (!memRefType) {
      return rewriter.notifyMatchFailure(read, "not a memref source");
    }

    if (!memRefType.isLastDimUnitStride()) {
      return rewriter.notifyMatchFailure(read, "non-unit trailing stride");
    }

    if (read.hasOutOfBoundsDim()) {
      return rewriter.notifyMatchFailure(read, "has out-of-bounds dim");
    }

    Location loc = read.getLoc();
    VectorType vecType = read.getVectorType();
    if (read.getMask()) {
      Value fill = vector::BroadcastOp::create(rewriter, loc, vecType,
                                               read.getPadding());
      rewriter.replaceOp(read, vector::MaskedLoadOp::create(
                                   rewriter, loc, vecType, read.getBase(),
                                   read.getIndices(), read.getMask(), fill));
    } else {
      rewriter.replaceOp(read, vector::LoadOp::create(rewriter, loc, vecType,
                                                      read.getBase(),
                                                      read.getIndices()));
    }
    return success();
  }
};

/// Lower `vector.transfer_write` to `vector.store` (unmasked) or
/// `vector.maskedstore` (masked). Requires minor-identity permutation map,
/// unit stride on the most minor memref dimension, and no out-of-bounds dims.
struct TransferWriteToVectorStoreLowering
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp write,
                                PatternRewriter &rewriter) const override {
    if (write.getVectorType().getRank() > 1) {
      return rewriter.notifyMatchFailure(write, "expected rank-1 or rank-0 vector");
    }

    if (!write.getPermutationMap().isMinorIdentity()) {
      return rewriter.notifyMatchFailure(write, "not minor identity map");
    }

    auto memRefType = dyn_cast<MemRefType>(write.getShapedType());
    if (!memRefType) {
      return rewriter.notifyMatchFailure(write, "not a memref dest");
    }

    if (!memRefType.isLastDimUnitStride()) {
      return rewriter.notifyMatchFailure(write, "non-unit trailing stride");
    }

    if (write.hasOutOfBoundsDim()) {
      return rewriter.notifyMatchFailure(write, "has out-of-bounds dim");
    }

    Location loc = write.getLoc();
    if (write.getMask()) {
      vector::MaskedStoreOp::create(rewriter, loc, write.getBase(),
                                    write.getIndices(), write.getMask(),
                                    write.getVector());
    } else {
      vector::StoreOp::create(rewriter, loc, write.getVector(), write.getBase(),
                              write.getIndices());
    }
    rewriter.eraseOp(write);
    return success();
  }
};

struct LLVMGPULower1DVectorOpsPass final
    : impl::LLVMGPULower1DVectorOpsPassBase<LLVMGPULower1DVectorOpsPass> {

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<OneDimMultiReductionToReduction>(ctx);
    patterns.add<CollapseTrailingUnitDimsTransferRead,
                 CollapseTrailingUnitDimsTransferWrite>(ctx);
    patterns.add<TransferReadToVectorLoadLowering,
                 TransferWriteToVectorStoreLowering>(ctx);
    vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
    IREE::VectorExt::populateVectorTransferGatherScatterLoweringPatterns(
        patterns);
    IREE::VectorExt::TransferGatherOp::getCanonicalizationPatterns(patterns,
                                                                   ctx);
    IREE::VectorExt::TransferScatterOp::getCanonicalizationPatterns(patterns,
                                                                    ctx);
    IREE::VectorExt::populateLowerTransferGatherScatterToVectorPatterns(
        patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler
