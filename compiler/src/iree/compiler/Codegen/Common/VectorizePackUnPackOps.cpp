// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-decompose-pack-unpack-ops"

namespace mlir {
namespace iree_compiler {
namespace {

struct DecomposePackUnPackOpsPass
    : public DecomposePackUnPackOpsBase<DecomposePackUnPackOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<linalg::LinalgDialect, func::FuncDialect, arith::ArithDialect,
                scf::SCFDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override;
};

struct VectorizePackUnPackOpsPass
    : public VectorizePackUnPackOpsBase<VectorizePackUnPackOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, func::FuncDialect,
                    arith::ArithDialect, scf::SCFDialect, tensor::TensorDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override;
};
}  // namespace

void DecomposePackUnPackOpsPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto funcOp = getOperation();

  // Apply tiling to make outer dims be all 1s.
  {
    IRRewriter rewriter(ctx);
    auto packOptions = scf::SCFTileAndFuseOptions().setTilingOptions(
        scf::SCFTilingOptions().setTileSizeComputationFunction(
            [](OpBuilder &builder, Operation *op) -> SmallVector<Value> {
              auto packOp = cast<tensor::PackOp>(op);

              // Do nothing if any of inner tile sizes is dynamic.
              if (llvm::any_of(packOp.getMixedTiles(), [](OpFoldResult tile) {
                    return tile.is<Value>();
                  })) {
                return {};
              }

              int inputRank = packOp.getSourceRank();
              SmallVector<Value> tileSizes(
                  inputRank,
                  builder.create<arith::ConstantIndexOp>(packOp.getLoc(), 1));
              return tileSizes;
            }));
    funcOp->walk([&](tensor::PackOp op) {
      FailureOr<scf::SCFTileAndFuseResult> tileAndFuseResult =
          scf::tileConsumerAndFuseProducerGreedilyUsingSCFForOp(
              rewriter, cast<TilingInterface>(op.getOperation()), packOptions);
      if (failed(tileAndFuseResult)) return signalPassFailure();
      rewriter.replaceOp(op, tileAndFuseResult->replacements[op.getResult()]);
    });

    auto unpackTilingOptions =
        scf::SCFTilingOptions().setTileSizeComputationFunction(
            [](OpBuilder &builder, Operation *op) {
              Location loc = op->getLoc();
              auto unpackOp = cast<tensor::UnPackOp>(op);
              int numLoops = unpackOp.getDestRank();
              auto dimAndTileMapping = unpackOp.getDimAndTileMapping();
              SmallVector<Value> tileSizes;
              for (int i = 0; i < numLoops; ++i) {
                if (dimAndTileMapping.count(i)) {
                  tileSizes.push_back(getValueOrCreateConstantIndexOp(
                      builder, loc, dimAndTileMapping[i]));
                } else {
                  tileSizes.push_back(
                      builder.create<arith::ConstantIndexOp>(loc, 1));
                }
              }
              return tileSizes;
            });
    funcOp->walk([&](tensor::UnPackOp op) {
      FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCFForOp(
          rewriter, cast<TilingInterface>(op.getOperation()),
          unpackTilingOptions);
      if (failed(tilingResult)) return signalPassFailure();
      rewriter.replaceOp(op, tilingResult->replacements);
    });
  }

  LLVM_DEBUG({
    llvm::dbgs()
        << "--- After applying tiling that makes outer dims be all 1s ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // Generalize pack and unpack ops and canonicalize tiled ops.
  {
    RewritePatternSet patterns(ctx);
    linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
    memref::populateResolveRankedShapeTypeResultDimsPatterns(patterns);
    patterns.add<linalg::GeneralizeOuterUnitDimsPackOpPattern,
                 linalg::GeneralizeOuterUnitDimsUnPackOpPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  LLVM_DEBUG({
    llvm::dbgs()
        << "--- After generalizing tensor.pack and tensor.unpack ops ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });
}

void VectorizePackUnPackOpsPass::runOnOperation() {
  MLIRContext *ctx = &getContext();

  // Kick in generic vectorizer.
  RewritePatternSet patterns(ctx);
  patterns.add<IREE::LinalgExt::LinalgVectorizationPattern>(ctx);
  linalg::populatePadOpVectorizationPatterns(patterns);
  vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
  vector::TransferReadOp::getCanonicalizationPatterns(patterns, ctx);
  vector::TransferWriteOp::getCanonicalizationPatterns(patterns, ctx);
  // TODO(hanchung): Capture the failure after the vectorization pattern
  // rewrite converges.
  (void)(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)));
}

std::unique_ptr<OperationPass<func::FuncOp>>
createDecomposePackUnPackOpsPass() {
  return std::make_unique<DecomposePackUnPackOpsPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
createVectorizePackUnPackOpsPass() {
  return std::make_unique<VectorizePackUnPackOpsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
