// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_CANONICALIZEPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

namespace {

static std::optional<SmallVector<OpFoldResult>> getDefiningMixedSizes(Value v) {
  if (auto empty = v.getDefiningOp<tensor::EmptyOp>()) {
    return empty.getMixedSizes();
  } else if (auto extract = v.getDefiningOp<tensor::ExtractSliceOp>()) {
    // TODO: Support rank reducing cases.
    if (extract.getSourceType().getRank() !=
        extract.getResultType().getRank()) {
      return {};
    }
    return extract.getMixedSizes();
  }
  return {};
}


struct ForwardStaticSliceExtract : OpRewritePattern<tensor::ExtractOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(tensor::ExtractOp read,
                                PatternRewriter &rewriter) const override {
    auto insert = read.getTensor().getDefiningOp<tensor::InsertSliceOp>();
    if (!insert || read.getIndices().size() != 1)
      return failure();
    RankedTensorType sourceType = insert.getSourceType();
    RankedTensorType destType = insert.getDestType();
    if (sourceType.getRank() != 1 || destType.getRank() != 1 ||
        !sourceType.hasStaticShape() || !destType.hasStaticShape() ||
        !insert.hasUnitStride())
      return failure();
    auto offset = getConstantIntValue(insert.getMixedOffsets()[0]);
    auto size = getConstantIntValue(insert.getMixedSizes()[0]);
    if (!offset || !size || *offset < 0 || *size <= 0 ||
        *size != sourceType.getDimSize(0) ||
        *offset > destType.getDimSize(0) ||
        *size > destType.getDimSize(0) - *offset)
      return failure();
    Location loc = read.getLoc();
    Value index = read.getIndices()[0];
    Value start = rewriter.create<arith::ConstantIndexOp>(loc, *offset);
    Value end = rewriter.create<arith::ConstantIndexOp>(loc, *offset + *size);
    Value ge = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sge, index, start);
    Value lt = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, index, end);
    Value inside = rewriter.create<arith::AndIOp>(loc, ge, lt);
    auto conditional = rewriter.create<scf::IfOp>(
        loc, read.getType(), inside,
        [&](OpBuilder &b, Location l) {
          Value local = b.create<arith::SubIOp>(l, index, start);
          Value value = b.create<tensor::ExtractOp>(
              l, insert.getSource(), ValueRange{local});
          b.create<scf::YieldOp>(l, value);
        },
        [&](OpBuilder &b, Location l) {
          Value value = b.create<tensor::ExtractOp>(
              l, insert.getDest(), ValueRange{index});
          b.create<scf::YieldOp>(l, value);
        });
    rewriter.replaceOp(read, conditional.getResults());
    return success();
  }
};

struct FoldFullInsertSlice : OpRewritePattern<tensor::InsertSliceOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertSliceOp,
                                PatternRewriter &rewriter) const override {
    if (!insertSliceOp.hasUnitStride() || !insertSliceOp.hasZeroOffset()) {
      return rewriter.notifyMatchFailure(insertSliceOp,
                                         "non-unit stride or non-zero offset.");
    }

    RankedTensorType sourceType = insertSliceOp.getSourceType();
    RankedTensorType resultType = insertSliceOp.getResultType();
    if (sourceType != resultType) {
      return rewriter.notifyMatchFailure(
          insertSliceOp,
          "unimplemented: Cast-like or reshape-like insert ops.");
    }

    std::optional<SmallVector<OpFoldResult>> mixedSizes =
        getDefiningMixedSizes(insertSliceOp.getDest());
    if (!mixedSizes) {
      return rewriter.notifyMatchFailure(
          insertSliceOp, "Could not find producer with list of tensor sizes.");
    }

    for (auto [insertSize, destSize] :
         llvm::zip_equal(insertSliceOp.getMixedSizes(), mixedSizes.value())) {
      if (isa<Value>(insertSize) || isa<Value>(destSize)) {
        if (insertSize != destSize) {
          return rewriter.notifyMatchFailure(insertSliceOp,
                                             "dynamic size mismatch");
        }
        continue;
      }

      // `getMixedSizes` for different ops returns different attribute types
      // (`index` or `i64`) so we compare the values of the ints directly here.
      int64_t staticInsertSize = getConstantIntValue(insertSize).value();
      int64_t staticDestSize = getConstantIntValue(insertSize).value();
      if (staticInsertSize != staticDestSize) {
        return rewriter.notifyMatchFailure(insertSliceOp,
                                           "static size mismatch");
      }
    }

    rewriter.replaceOp(insertSliceOp, insertSliceOp.getSource());
    return success();
  }
};

/// Convert an "affine.apply" operation into a sequence of arith ops.
class AffineApplyLowering : public OpRewritePattern<affine::AffineApplyOp> {
public:
  using Base::Base;

  LogicalResult matchAndRewrite(affine::AffineApplyOp op,
                                PatternRewriter &rewriter) const override {
    auto maybeExpandedMap =
        affine::expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(),
                                llvm::to_vector<8>(op.getOperands()));
    if (!maybeExpandedMap) {
      return failure();
    }
    rewriter.replaceOp(op, *maybeExpandedMap);
    return success();
  }
};

/// Canonicalize operations in nested regions.
struct CanonicalizePass : impl::CanonicalizePassBase<CanonicalizePass> {
  using IREE::Flow::impl::CanonicalizePassBase<
      CanonicalizePass>::CanonicalizePassBase;
  /// Initialize the canonicalizer by building the set of patterns used during
  /// execution.
  LogicalResult initialize(MLIRContext *context) override {
    // Inherit the same config defaults from the upstream canonicalizer pass.
    config.setUseTopDownTraversal().setRegionSimplificationLevel(
        mlir::GreedySimplifyRegionLevel::Normal);

    RewritePatternSet owningPatterns(context);
    for (auto *dialect : context->getLoadedDialects()) {
      dialect->getCanonicalizationPatterns(owningPatterns);
    }
    for (RegisteredOperationName op : context->getRegisteredOperations()) {
      op.getCanonicalizationPatterns(owningPatterns, context);
    }

    // Pull in some borderline/downstream canonicalizations for the Flow
    // compilation phase.
    tensor::populateMergeConsecutiveInsertExtractSlicePatterns(owningPatterns);
    owningPatterns.add<FoldFullInsertSlice>(context);
    owningPatterns.add<ForwardStaticSliceExtract>(context);
    owningPatterns.add<AffineApplyLowering>(context);

    patterns =
        std::make_shared<FrozenRewritePatternSet>(std::move(owningPatterns));
    return success();
  }
  void runOnOperation() override {
    // Canonicalization is best-effort. Non-convergence is not a pass failure.
    config.enableConstantCSE(cseConstants);
    LogicalResult didConverge =
        applyPatternsGreedily(getOperation(), *patterns, config);
    if (this->testConvergence && failed(didConverge)) {
      getOperation()->emitError("Canonicalizer failed to converge");
      return signalPassFailure();
    }
  }
  GreedyRewriteConfig config;
  std::shared_ptr<const FrozenRewritePatternSet> patterns;
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Flow
