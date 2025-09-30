// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-vectorize-memref-copy"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_VECTORIZEMEMREFCOPYPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

struct TileLinalgCopy final : OpRewritePattern<linalg::CopyOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(linalg::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<int64_t> bounds = copyOp.getStaticLoopRanges();
    if (std::all_of(bounds.begin(), bounds.end() - 1,
                    [](int64_t b) { return b == 1; })) {
      return rewriter.notifyMatchFailure(copyOp, "is already tiled");
    }
    std::optional<SmallVector<int64_t>> maybeStaticTileSizes =
        getCopyTileSizes(rewriter, copyOp);
    if (!maybeStaticTileSizes.has_value()) {
      return rewriter.notifyMatchFailure(copyOp,
                                         "could not retrieve tile sizes");
    }

    auto tilingInterfaceOp = cast<TilingInterface>(copyOp.getOperation());
    rewriter.setInsertionPoint(tilingInterfaceOp);
    SmallVector<OpFoldResult> tileSizes = getAsIndexOpFoldResult(
        rewriter.getContext(), maybeStaticTileSizes.value());

    scf::SCFTilingOptions tilingOptions;
    tilingOptions.setTileSizes(tileSizes);
    tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);

    FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCF(rewriter, tilingInterfaceOp, tilingOptions);
    if (failed(tilingResult)) {
      return rewriter.notifyMatchFailure(copyOp, "tiling failed");
    }
    if (tilingInterfaceOp->use_empty()) {
      rewriter.eraseOp(tilingInterfaceOp);
    }
    return success();
  }
};

struct ConvertMemrefCopyToLinalgCopy final : OpRewritePattern<memref::CopyOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    Location loc = copyOp.getLoc();
    SmallVector<OpFoldResult> bounds =
        memref::getMixedSizes(rewriter, loc, copyOp.getSource());
    if (std::all_of(bounds.begin(), bounds.end() - 1, [](OpFoldResult ofr) {
          return isConstantIntValue(ofr, 1);
        })) {
      return rewriter.notifyMatchFailure(copyOp, "is already tiled");
    }
    linalg::CopyOp::create(rewriter, copyOp.getLoc(), copyOp.getSource(),
                           copyOp.getTarget());
    rewriter.eraseOp(copyOp);
    return success();
  }
};

struct ConvertLinalgCopyToMemrefCopy final : OpRewritePattern<linalg::CopyOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(linalg::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    if (copyOp.hasPureTensorSemantics()) {
      return failure();
    }
    SmallVector<int64_t> bounds = copyOp.getStaticLoopRanges();
    if (!std::all_of(bounds.begin(), bounds.end() - 1,
                     [](int64_t b) { return b == 1; })) {
      return rewriter.notifyMatchFailure(copyOp, "should first be tiled");
    }
    memref::CopyOp::create(rewriter, copyOp.getLoc(),
                           copyOp.getDpsInputOperand(0)->get(),
                           copyOp.getDpsInitOperand(0)->get());
    rewriter.eraseOp(copyOp);
    return success();
  }
};

struct VectorizeMemrefCopyPass final
    : impl::VectorizeMemrefCopyPassBase<VectorizeMemrefCopyPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    auto funcOp = getOperation();

    // First convert all `memref.copy` operations to `linalg.copy` so that they
    // can be tiled. Tiling them avoids copies with dynamic dimensions if the
    // dynamic dimension is not the innermost. Afterwards, tiled `linalg.copy`
    // operations are converted back to `memref.copy` operations and vectorized.
    RewritePatternSet patterns(ctx);
    patterns.add<ConvertMemrefCopyToLinalgCopy>(&getContext());
    patterns.add<TileLinalgCopy>(&getContext());
    patterns.add<ConvertLinalgCopyToMemrefCopy>(&getContext());
    patterns.add<linalg::CopyVectorizationPattern>(&getContext());
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
  }
};

} // namespace
} // namespace mlir::iree_compiler
