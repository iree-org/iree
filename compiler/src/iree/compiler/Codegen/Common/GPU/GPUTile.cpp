// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- GPUTile.cpp ------------------------------------------------------===//
//
// This pass tiles and Linalg ops with tensor semantics to invocations.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-tile"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUTILEPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {
//===----------------------------------------------------------------------===//
// Tiling and fusion utilities
//===----------------------------------------------------------------------===//

/// Collects computation ops which we will use as anchor to tile and fuse.
static FailureOr<IREE::Codegen::LoweringConfigAttr>
collectComputeOps(mlir::FunctionOpInterface funcOp,
                  SmallVectorImpl<Operation *> &computeOps) {
  // If there are `scf.if` ops which have linalg ops, we have both a fast and
  // slow paths for padding handling. Then we need to scan both regions to
  // discover such computation ops so that we can tile and fuse both regions.
  SmallVector<scf::IfOp, 1> ifOps;
  funcOp.walk<WalkOrder::PreOrder>([&ifOps](scf::IfOp ifOp) -> WalkResult {
    if (ifOp->getParentOfType<linalg::LinalgOp>()) {
      // Exclude scf.if in linalg op
      return WalkResult::skip();
    } else {
      ifOps.push_back(ifOp);
      return WalkResult::advance();
    }
  });

  SmallVector<IREE::Codegen::LoweringConfigAttr> configs;
  if (ifOps.empty()) {
    computeOps = getComputeOps(funcOp);
    for (Operation *op : computeOps) {
      if (auto config =
              getLoweringConfig<IREE::Codegen::LoweringConfigAttr>(op))
        configs.push_back(config);
    }
    if (computeOps.size() > 1) {
      // Only keep the last compute ops.
      std::reverse(computeOps.begin(), computeOps.end());
      computeOps.resize(1);
    }
  } else {
    if (ifOps.size() > 1) {
      return funcOp.emitError("expected to contain <= 1 scf.if ops");
    }

    ifOps.front()->walk([&configs](Operation *op) {
      if (isa<linalg::LinalgOp, TilingInterface>(op)) {
        if (auto config =
                getLoweringConfig<IREE::Codegen::LoweringConfigAttr>(op))
          configs.push_back(config);
      }
    });

    for (Operation &op : llvm::reverse(*ifOps.front().thenBlock())) {
      if (isa<linalg::LinalgOp, TilingInterface>(op)) {
        computeOps.push_back(&op);
        break;
      }
    }
    if (Block *elseBlock = ifOps.front().elseBlock()) {
      for (Operation &op : llvm::reverse(*elseBlock)) {
        if (isa<linalg::LinalgOp, TilingInterface>(op)) {
          computeOps.push_back(&op);
          break;
        }
      }
    }
  }
  if (configs.empty()) {
    return funcOp.emitError("missing lowering configuration");
  }
  if (!llvm::all_equal(configs)) {
    return funcOp.emitError("contains conflicting lowering configuration");
  }
  return configs.front();
}

static LogicalResult tileAndDistributeToThreads(TilingInterface consumerOp,
                                                ArrayRef<int64_t> tileSizes) {
  MLIRContext *context = consumerOp->getContext();
  IRRewriter rewriter(context);
  SmallVector<OpFoldResult> tileSizesOfr =
      getAsIndexOpFoldResult(context, tileSizes);
  scf::SCFTilingOptions tilingOptions;
  tilingOptions.setTileSizes(tileSizesOfr);
  scf::SCFTileAndFuseOptions tileAndFuseOptions;
  tileAndFuseOptions.setTilingOptions(tilingOptions);
  tileAndFuseOptions.setFusionControlFn(
      [](tensor::ExtractSliceOp sliceOp, OpResult origProducer,
         bool isDestinationOperand)
          -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
        if (isa<tensor::PadOp>(origProducer.getOwner())) {
          return std::nullopt;
        }
        return scf::SCFTileAndFuseOptions::ControlFnResult{false};
      });
  FailureOr<scf::SCFTileAndFuseResult> tileAndFuseResult =
      scf::tileConsumerAndFuseProducersUsingSCF(rewriter, consumerOp,
                                                tileAndFuseOptions);

  if (failed(tileAndFuseResult)) {
    return consumerOp.emitOpError("failed tiling and fusing producers");
  }

  SmallVector<Value> replacements;
  replacements.resize(consumerOp->getNumResults());
  for (const auto &[index, result] :
       llvm::enumerate(consumerOp->getResults())) {
    replacements[index] = tileAndFuseResult->replacements.lookup(result);
  }
  consumerOp->replaceAllUsesWith(replacements);

  // We don't distribute here; instead, it will be done in a later step
  // after bufferization. So add attributes to the tiled loop nest to
  // indicate that they should be distributed to invocations.
  ArrayRef<LoopLikeOpInterface> loops = tileAndFuseResult.value().loops;
  const char *attrName = getGPUDistributeAttrName();
  // We can have more than 3 dimensions being tiled (e.g., for convolutions with
  // non-1 batch). But only the innermost 3 dimensions are distributed.
  for (auto [dim, loop] : zip(llvm::seq(0, 3), llvm::reverse(loops))) {
    loop->setAttr(attrName, rewriter.getIndexAttr(dim));
  }
  return success();
}

/// Tiles reduction dimensions.
static LogicalResult
tileReduction(mlir::FunctionOpInterface funcOp,
              const scf::SCFTileSizeComputationFunction &computeFn) {
  auto filter = LinalgTransformationFilter().setMatchByDefault();
  auto options =
      scf::SCFTilingOptions().setTileSizeComputationFunction(computeFn);
  auto result = tileLinalgOpsWithFilter(funcOp, options, filter);

  LLVM_DEBUG({
    llvm::dbgs() << "--- After tiling reduction dimensions  ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });
  return result;
}

/// Fuses `tensor.pad` ops into the the materalized loop nests containing
/// their consumer ops.
static void fusePadIntoConsumer(mlir::FunctionOpInterface funcOp) {
  MLIRContext *context = funcOp.getContext();
  RewritePatternSet patterns(context);
  patterns.insert<linalg::ExtractSliceOfPadTensorSwapPattern>(
      context, [](tensor::ExtractSliceOp) { return false; });
  (void)applyPatternsGreedily(funcOp, std::move(patterns));

  LLVM_DEBUG({
    llvm::dbgs() << "--- After fusing padding into consumers ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });
};

/// Concretizes `tensor.pad` ops' result shapes.
static void concretizePadShape(mlir::FunctionOpInterface funcOp) {
  MLIRContext *context = funcOp.getContext();
  RewritePatternSet patterns(context);
  SmallVector<int64_t> numWorkgroups = getStaticNumWorkgroups(funcOp);
  populateConcretizePadResultShapePatterns(patterns, numWorkgroups);
  populateFoldAffineMinInDistributedLoopsPatterns(patterns, numWorkgroups);
  (void)applyPatternsGreedily(funcOp, std::move(patterns));

  LLVM_DEBUG({
    llvm::dbgs() << "--- After concretizing pad result shape ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });
}

/// Tiles one of the convolution output window dimensions with size 1 to prepare
/// for downsizing 2-D convolution ops into 1-D ones.
static LogicalResult tileAndUnrollConvWindow(mlir::FunctionOpInterface funcOp,
                                             ArrayRef<OpFoldResult> tileSizes) {
  SmallVector<linalg::ConvolutionOpInterface, 1> convOps;
  funcOp.walk([&convOps](linalg::ConvolutionOpInterface convOp) {
    convOps.push_back(convOp);
  });

  for (linalg::ConvolutionOpInterface convOp : convOps) {
    auto consumerOp = cast<linalg::LinalgOp>(*convOp);
    IRRewriter rewriter(funcOp.getContext());
    FailureOr<scf::SCFTileAndFuseResult> tileAndFuseResult =
        scf::tileConsumerAndFuseProducersUsingSCF(
            rewriter, cast<TilingInterface>(consumerOp.getOperation()),
            scf::SCFTileAndFuseOptions().setTilingOptions(
                scf::SCFTilingOptions().setTileSizes(tileSizes)));

    if (failed(tileAndFuseResult)) {
      return consumerOp.emitOpError("failed tiling and fusing producers");
    }

    SmallVector<Value> replacements;
    replacements.resize(consumerOp->getNumResults());
    for (const auto &[index, result] :
         llvm::enumerate(consumerOp->getResults())) {
      replacements[index] = tileAndFuseResult->replacements.lookup(result);
    }
    consumerOp->replaceAllUsesWith(replacements);

    // Fully unroll the generated loop. This allows us to remove the loop
    // for parallel output window dimension, so it helps future vector
    // transformations.

    ArrayRef<LoopLikeOpInterface> loops = tileAndFuseResult.value().loops;
    if (!loops.empty()) {
      assert(loops.size() == 1);
      scf::ForOp loopOp = cast<scf::ForOp>(loops.front());
      IntegerAttr ub;
      if (!matchPattern(loopOp.getUpperBound(), m_Constant(&ub))) {
        return loopOp.emitOpError("upper bound should be a constant");
      }
      if (failed(mlir::loopUnrollByFactor(loopOp, ub.getInt()))) {
        return loopOp.emitOpError("failed unrolling by factor 1");
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After tiling convolution output window ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Main pass
//===----------------------------------------------------------------------===//

struct GPUTilePass final : impl::GPUTilePassBase<GPUTilePass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    FunctionOpInterface funcOp = getOperation();

    // Try to find computation ops which we will use as anchor to tile and fuse.
    SmallVector<Operation *> computeOps;
    FailureOr<IREE::Codegen::LoweringConfigAttr> loweringConfig =
        collectComputeOps(funcOp, computeOps);
    if (failed(loweringConfig))
      return signalPassFailure();
    assert(computeOps.size() <= 2);

    // Now tile the last computation op to invocations and fuse all operand
    // computation ops into the materialized loop nest.
    auto threadTileSizes = loweringConfig->getTileSizeVals(1);
    for (Operation *computeOp : computeOps) {
      auto consumerOp = dyn_cast<TilingInterface>(computeOp);
      if (!consumerOp ||
          failed(tileAndDistributeToThreads(consumerOp, threadTileSizes)))
        return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After tiling to invocations ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    fusePadIntoConsumer(funcOp);

    concretizePadShape(funcOp);

    auto reductionTileComputeFn = getGPUScfTileSizeComputeFn(funcOp, 2);
    if (failed(reductionTileComputeFn) ||
        failed(tileReduction(funcOp, reductionTileComputeFn.value()))) {
      return signalPassFailure();
    }

    fusePadIntoConsumer(funcOp);

    SmallVector<OpFoldResult> windowTileSizes =
        getAsIndexOpFoldResult(context, loweringConfig->getTileSizeVals(3));
    if (failed(tileAndUnrollConvWindow(funcOp, windowTileSizes))) {
      return signalPassFailure();
    }

    concretizePadShape(funcOp);

    { // Downsize n-D (n > 1) convolutions to 1-D.
      RewritePatternSet patterns(context);
      linalg::populateDecomposeConvolutionPatterns(patterns);
      // Downsizing creates consecutive extract/insert slice ops. Merge them.
      tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
      // Pull in patterns to fold constant insert/extract slice op parameters.
      tensor::InsertSliceOp::getCanonicalizationPatterns(patterns, context);
      tensor::ExtractSliceOp::getCanonicalizationPatterns(patterns, context);
      // Pull in scf.for op canonicalization patterns to help hoisting across
      // multiple loops and remove loop carried values unused in the body.
      scf::ForOp::getCanonicalizationPatterns(patterns, context);
      (void)applyPatternsGreedily(funcOp, std::move(patterns));

      LLVM_DEBUG({
        llvm::dbgs() << "--- After Downsizing N-D convolution to 1-D  ---\n";
        funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
        llvm::dbgs() << "\n\n";
      });
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
