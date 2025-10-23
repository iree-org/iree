// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-vectorize-memref-copy"

constexpr char kIsTiled[] = "_is_tiled";

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_VECTORIZEMEMREFCOPYPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

struct TileLinalgCopy final : OpRewritePattern<memref::CopyOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    if (copyOp->hasAttr(kIsTiled)) {
      return rewriter.notifyMatchFailure(copyOp, "already tiled");
    }
    auto linalgCopy = linalg::CopyOp::create(
        rewriter, copyOp.getLoc(), copyOp.getSource(), copyOp.getTarget());
    std::optional<SmallVector<int64_t>> maybeStaticTileSizes =
        getCopyTileSizes(linalgCopy);
    if (!maybeStaticTileSizes.has_value()) {
      rewriter.eraseOp(linalgCopy);
      return rewriter.notifyMatchFailure(copyOp,
                                         "could not retrieve tile sizes");
    }
    SmallVector<int64_t> staticBounds = linalgCopy.getStaticLoopRanges();

    auto tilingInterfaceOp = cast<TilingInterface>(linalgCopy.getOperation());
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
    // Put an marker on the tiled ops, so it's easy to recognize that they
    // shouldn't be tiled again.
    for (Operation *tiledOp : tilingResult->tiledOps) {
      tiledOp->setAttr(kIsTiled, mlir::UnitAttr::get(copyOp.getContext()));
    }
    // Put an marker on the loop ops, so they can be targeted for
    // simplification.
    for (LoopLikeOpInterface loop : llvm::reverse(tilingResult->loops)) {
      loop->setAttr(kIsTiled, mlir::UnitAttr::get(loop.getContext()));
    }
    if (tilingInterfaceOp->use_empty()) {
      rewriter.eraseOp(tilingInterfaceOp);
    }
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
    auto newCopy = memref::CopyOp::create(rewriter, copyOp.getLoc(),
                                          copyOp.getDpsInputOperand(0)->get(),
                                          copyOp.getDpsInitOperand(0)->get());
    newCopy->setAttrs(copyOp->getAttrs());
    rewriter.eraseOp(copyOp);
    return success();
  }
};

/// TODO(#22245): Enable vector masking for unaligned/dynamic copies to improve
/// copy performance further.
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
    patterns.add<TileLinalgCopy>(&getContext());
    patterns.add<linalg::CopyVectorizationPattern>(&getContext());
    patterns.add<ConvertLinalgCopyToMemrefCopy>(&getContext());
    // Try to remove generated single iteration loops and canonicalize generated
    // subview operations.
    populateRemoveSingleIterationLoopPattern(
        patterns,
        [&](scf::ForOp forOp) -> bool { return forOp->hasAttr(kIsTiled); });
    memref::SubViewOp::getCanonicalizationPatterns(patterns, ctx);
    (void)applyPatternsGreedily(funcOp, std::move(patterns));

    // Clean up the temporary isTiled markers.
    funcOp->walk([](Operation *op) {
      if (op->hasAttr(kIsTiled)) {
        op->removeAttr(kIsTiled);
      }
    });
  }
};

} // namespace
} // namespace mlir::iree_compiler
