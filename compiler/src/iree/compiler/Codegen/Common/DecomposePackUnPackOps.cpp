// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-decompose-pack-unpack-ops"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_DECOMPOSEPACKUNPACKOPSPASS
#define GEN_PASS_DEF_DECOMPOSEBOUNDARYPACKUNPACKOPSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

using PackUnPackControlFn = std::function<LogicalResult(Operation *)>;

namespace {

//===----------------------------------------------------------------------===//
// Shared rewrite patterns
//===----------------------------------------------------------------------===//

/// A wrapper pattern that calls linalg::lowerPack on tensor::PackOp. It lowers
/// a tensor.pack op to tensor.pad + tensor.expand_shape + linalg.transpose ops.
struct LowerPackPattern : public OpRewritePattern<tensor::PackOp> {
  using OpRewritePattern<tensor::PackOp>::OpRewritePattern;

  explicit LowerPackPattern(MLIRContext *context,
                            std::optional<PackUnPackControlFn> controlFn)
      : OpRewritePattern(context), controlFn(controlFn) {}

  LogicalResult matchAndRewrite(tensor::PackOp op,
                                PatternRewriter &rewriter) const override {
    if (controlFn && failed(controlFn.value()(op))) {
      return failure();
    }
    FailureOr<linalg::LowerPackResult> res = linalg::lowerPack(rewriter, op);
    if (failed(res)) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower to pad + expand + transpose");
    }
    return success();
  }

private:
  std::optional<PackUnPackControlFn> controlFn;
};

/// A warpper pattern that calls linalg::lowerUnPack on tensor::UnPackOp. It
/// lowers a tensor.unpack op to tensor.empty + linalg.transpose +
/// tensor.collapse_shape + tensor.extract_slice ops.
struct LowerUnPackPattern : public OpRewritePattern<tensor::UnPackOp> {
  using OpRewritePattern<tensor::UnPackOp>::OpRewritePattern;

  explicit LowerUnPackPattern(MLIRContext *context,
                              std::optional<PackUnPackControlFn> controlFn)
      : OpRewritePattern(context), controlFn(controlFn) {}

  LogicalResult matchAndRewrite(tensor::UnPackOp op,
                                PatternRewriter &rewriter) const override {
    if (controlFn && failed(controlFn.value()(op))) {
      return failure();
    }
    FailureOr<linalg::LowerUnPackOpResult> res =
        linalg::lowerUnPack(rewriter, op);
    if (failed(res)) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower to empty + transpose + reshape + extract_slice");
    }
    return success();
  }

private:
  std::optional<PackUnPackControlFn> controlFn;
};

//===----------------------------------------------------------------------===//
// Shared pass implementation
//===----------------------------------------------------------------------===//

static LogicalResult commonRunOnOperation(
    MLIRContext *ctx, FunctionOpInterface funcOp, bool useOnlyReshapes,
    bool tileOuterToOne,
    std::optional<PackUnPackControlFn> controlFn = std::nullopt) {
  // Generalization patterns for outer unit dims have higher priority because
  // they do not generate reshape ops.
  if (!useOnlyReshapes) {
    RewritePatternSet patterns(ctx);
    patterns.add<linalg::GeneralizeOuterUnitDimsPackOpPattern,
                 linalg::GeneralizeOuterUnitDimsUnPackOpPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitError(
          "failed to apply generalization patterns on pack/unpack ops for "
          "outer unit dims cases");
      return failure();
    }
  }

  // Do not convert pack and unpack ops if outer dims are expected to always be
  // tiled to one.
  if (!tileOuterToOne) {
    RewritePatternSet patterns(ctx);
    patterns.add<LowerPackPattern, LowerUnPackPattern>(ctx, controlFn);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitError(
          "failed to apply generalization patterns on pack/unpack ops for "
          "general cases.");
      return failure();
    }
  }

  // TODO(hanchung): Below is a fallback solution for tensor.pack/unpack
  // decomposition. They will be retired after lowerPack and lowerUnPack handle
  // all the cases.

  // Apply tiling to make outer dims be all 1s.
  {
    IRRewriter rewriter(ctx);
    auto packOptions = scf::SCFTileAndFuseOptions().setTilingOptions(
        scf::SCFTilingOptions().setTileSizeComputationFunction(
            [](OpBuilder &builder, Operation *op) -> SmallVector<OpFoldResult> {
              auto packOp = cast<tensor::PackOp>(op);

              // Do nothing if any of inner tile sizes is dynamic.
              if (llvm::any_of(packOp.getMixedTiles(), [](OpFoldResult tile) {
                    return tile.is<Value>();
                  })) {
                return {};
              }

              int inputRank = packOp.getSourceRank();
              SmallVector<OpFoldResult> tileSizes(inputRank,
                                                  builder.getIndexAttr(1));
              return tileSizes;
            }));
    {
      WalkResult status = funcOp->walk([&](tensor::PackOp op) {
        if (controlFn && failed(controlFn.value()(op))) {
          return WalkResult::advance();
        }
        FailureOr<scf::SCFTileAndFuseResult> tileAndFuseResult =
            scf::tileConsumerAndFuseProducersUsingSCF(
                rewriter, cast<TilingInterface>(op.getOperation()),
                packOptions);
        if (failed(tileAndFuseResult))
          return WalkResult::interrupt();
        rewriter.replaceOp(op, tileAndFuseResult->replacements[op.getResult()]);
        return WalkResult::advance();
      });
      if (status.wasInterrupted()) {
        return failure();
      }
    }

    auto unpackTilingOptions =
        scf::SCFTilingOptions().setTileSizeComputationFunction(
            [](OpBuilder &builder, Operation *op) {
              auto unpackOp = cast<tensor::UnPackOp>(op);
              int numLoops = unpackOp.getDestRank();
              auto dimAndTileMapping = unpackOp.getDimAndTileMapping();
              SmallVector<OpFoldResult> tileSizes;
              for (int i = 0; i < numLoops; ++i) {
                if (dimAndTileMapping.count(i)) {
                  tileSizes.push_back(dimAndTileMapping[i]);
                } else {
                  tileSizes.push_back(builder.getIndexAttr(1));
                }
              }
              return tileSizes;
            });
    {
      WalkResult status = funcOp->walk([&](tensor::UnPackOp op) {
        if (controlFn && failed(controlFn.value()(op))) {
          return WalkResult::advance();
        }
        FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCF(
            rewriter, cast<TilingInterface>(op.getOperation()),
            unpackTilingOptions);
        if (failed(tilingResult))
          return WalkResult::interrupt();
        rewriter.replaceOp(op, tilingResult->replacements);
        return WalkResult::advance();
      });
      if (status.wasInterrupted()) {
        return failure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs()
          << "--- After applying tiling that makes outer dims be all 1s ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }

  // Canonicalize tiled ops.
  {
    RewritePatternSet patterns(ctx);
    linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    ctx->getOrLoadDialect<tensor::TensorDialect>()->getCanonicalizationPatterns(
        patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return failure();
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "--- After canonicalizing tiled ops ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  {
    RewritePatternSet patterns(ctx);
    if (useOnlyReshapes) {
      patterns.add<LowerPackPattern, LowerUnPackPattern>(ctx, controlFn);
    } else {
      patterns.add<linalg::GeneralizeOuterUnitDimsPackOpPattern,
                   linalg::GeneralizeOuterUnitDimsUnPackOpPattern>(ctx);
    }
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return failure();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// DecomposePackUnPackOpsPass
//===----------------------------------------------------------------------===//

struct DecomposePackUnPackOpsPass final
    : impl::DecomposePackUnPackOpsPassBase<DecomposePackUnPackOpsPass> {
  using impl::DecomposePackUnPackOpsPassBase<
      DecomposePackUnPackOpsPass>::DecomposePackUnPackOpsPassBase;

  void runOnOperation() override;
};

} // namespace

void DecomposePackUnPackOpsPass::runOnOperation() {
  if (failed(commonRunOnOperation(&getContext(), getOperation(),
                                  useOnlyReshapes, tileOuterToOne))) {
    return signalPassFailure();
  }
}

//===----------------------------------------------------------------------===//
// DecomposeBoundaryPackUnPackOpsPass
//===----------------------------------------------------------------------===//

namespace {

struct DecomposeBoundaryPackUnPackOpsPass final
    : impl::DecomposeBoundaryPackUnPackOpsPassBase<
          DecomposeBoundaryPackUnPackOpsPass> {
  using impl::DecomposeBoundaryPackUnPackOpsPassBase<
      DecomposeBoundaryPackUnPackOpsPass>::
      DecomposeBoundaryPackUnPackOpsPassBase;

  void runOnOperation() override;
};

} // namespace

/// Check if the given `op` is a pack or unpack op with padding.
static bool hasPadding(Operation *op) {
  auto needsPad = [](ShapedType unpackedType, ArrayRef<int64_t> innerDimPos,
                     ArrayRef<int64_t> staticInnerTiles) {
    for (auto [dimPos, tile] : llvm::zip_equal(innerDimPos, staticInnerTiles)) {
      if (unpackedType.isDynamicDim(dimPos) || ShapedType::isDynamic(tile) ||
          unpackedType.getDimSize(dimPos) % tile != 0) {
        return true;
      }
    }
    return false;
  };
  auto packOp = dyn_cast<tensor::PackOp>(op);
  if (packOp && needsPad(packOp.getSourceType(), packOp.getInnerDimsPos(),
                         packOp.getStaticInnerTiles())) {
    return true;
  }
  auto unPackOp = dyn_cast<tensor::UnPackOp>(op);
  if (unPackOp && needsPad(unPackOp.getDestType(), unPackOp.getInnerDimsPos(),
                           unPackOp.getStaticInnerTiles())) {
    return true;
  }
  return false;
}

/// Control function for decomposing pack and unpack ops. Returns true if the
/// op is an unpadded pack or unpack op, and it is at the boundary of a
/// dispatch. The following conditions need to be met:
/// 1. The PackOp or UnPackOp must have no padding.
/// 2. If the op is a PackOp, then its producer must be a dispatch tensor load.
/// 3. If the op is an UnPackOp, then all of its consumers must be dispatch
///    tensor stores.
static LogicalResult isUnpaddedAndAtBoundary(Operation *op) {
  if (!isa<tensor::PackOp>(op) && !isa<tensor::UnPackOp>(op)) {
    return failure();
  }
  if (hasPadding(op)) {
    return failure();
  }

  // If the producer is a dispatch tensor load, then the `op` is decomposable
  // if it is a PackOp.
  if (isa<tensor::PackOp>(op) &&
      op->getOperand(0).getDefiningOp<IREE::Flow::DispatchTensorLoadOp>()) {
    return success();
  }
  // If all consumers are dispatch tensor stores, then the `op` is decomposable
  // if it is an UnPackOp.
  if (isa<tensor::UnPackOp>(op) &&
      llvm::all_of(op->getUsers(), [&](Operation *user) {
        return isa<IREE::Flow::DispatchTensorStoreOp>(user);
      })) {
    return success();
  }
  return failure();
}

void DecomposeBoundaryPackUnPackOpsPass::runOnOperation() {
  if (failed(commonRunOnOperation(&getContext(), getOperation(),
                                  /*useOnlyReshapes=*/true, tileOuterToOne,
                                  isUnpaddedAndAtBoundary))) {
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler
