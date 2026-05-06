// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPU1DVECTORCANONICALIZATIONSPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

/// true: vector
/// false: vector
/// pred: i1
///
/// select(pred, true, false) -> broadcast(pred)
/// select(pred, false, true) -> broadcast(not(pred))
///
/// Ideally, this would be a canonicalization pattern on arith::SelectOp, but
/// we cannot have arith depending on vector. Also, it would implicitly force
/// users only using arith and vector dialect to use vector dialect. Since
/// upstream does not have a mechanism of registering canonicalization without
/// adding dependencies like this, we manually add it where it is needed.
struct FoldI1SelectToBroadcast final : OpRewritePattern<arith::SelectOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(arith::SelectOp selectOp,
                                PatternRewriter &rewriter) const override {
    auto vecType = dyn_cast<VectorType>(selectOp.getType());
    if (!vecType || !vecType.getElementType().isInteger(1)) {
      return failure();
    }

    // Vector conditionals do not need broadcast and are already handled by
    // the arith.select folder.
    Value pred = selectOp.getCondition();
    if (isa<VectorType>(pred.getType())) {
      return failure();
    }

    std::optional<int64_t> trueInt =
        getConstantIntValue(selectOp.getTrueValue());
    std::optional<int64_t> falseInt =
        getConstantIntValue(selectOp.getFalseValue());
    if (!trueInt || !falseInt) {
      return failure();
    }

    // Redundant selects are already handled by arith.select canonicalizations.
    if (trueInt.value() == falseInt.value()) {
      return failure();
    }

    // The only remaining possibilities are:
    //
    // select(pred, true, false)
    // select(pred, false, true)

    // select(pred, false, true) -> select(not(pred), true, false)
    if (trueInt.value() == 0) {
      // TODO: flip the condition here to handle through the existing path.
      return failure();
    }

    /// select(pred, true, false) -> broadcast(pred)
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(
        selectOp, vecType.clone(rewriter.getI1Type()), pred);
    return success();
  }
};

/// Canonicalize a rank-1 single-element vector transfer op so that its
/// permutation map is the minor identity.
///
/// When the vector type is `vector<1xT>`, the permutation map is irrelevant
/// to which element is accessed: the single vector lane has iteration offset 0,
/// so the element is always at `memref[indices...]` regardless of which
/// memref/tensor dimension the map points at. Replacing the map with minor
/// identity unblocks lowering to vector.load / vector.store.
template <typename TransferOp>
struct CanonicalizeSize1TransferMap final : OpRewritePattern<TransferOp> {
  using OpRewritePattern<TransferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TransferOp op,
                                PatternRewriter &rewriter) const override {
    VectorType vecType = op.getVectorType();
    if (vecType.getRank() != 1 || vecType.getShape()[0] != 1) {
      return failure();
    }

    AffineMap map = op.getPermutationMap();
    if (map.isMinorIdentity()) {
      return failure();
    }

    int64_t srcRank = op.getShapedType().getRank();
    AffineMap minorIdentity =
        AffineMap::getMinorIdentityMap(srcRank, 1, rewriter.getContext());

    if constexpr (std::is_same_v<TransferOp, vector::TransferReadOp>) {
      rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
          op, vecType, op.getBase(), op.getIndices(),
          AffineMapAttr::get(minorIdentity), op.getPadding(), op.getMask(),
          op.getInBoundsAttr());
    } else {
      rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
          op, op.getVector(), op.getBase(), op.getIndices(),
          AffineMapAttr::get(minorIdentity), op.getMask(),
          op.getInBoundsAttr());
    }
    return success();
  }
};

struct LLVMGPU1DVectorCanonicalizationsPass final
    : impl::LLVMGPU1DVectorCanonicalizationsPassBase<
          LLVMGPU1DVectorCanonicalizationsPass> {

  void runOnOperation() override {
    mlir::FunctionOpInterface funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    vector::BroadcastOp::getCanonicalizationPatterns(patterns, ctx);
    arith::SelectOp::getCanonicalizationPatterns(patterns, ctx);
    patterns.add<FoldI1SelectToBroadcast>(ctx);
    patterns.add<CanonicalizeSize1TransferMap<vector::TransferReadOp>,
                 CanonicalizeSize1TransferMap<vector::TransferWriteOp>>(ctx);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler
