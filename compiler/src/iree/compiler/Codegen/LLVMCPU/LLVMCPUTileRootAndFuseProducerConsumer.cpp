// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/UKernelOps.h"
#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-tile-root-and-fuse-producers-consumers"

namespace mlir::iree_compiler {

namespace {

/// Implementation of tile root and fuse producers and consumers greedily.
static LogicalResult tileRootAndFuseProducerConsumerUsingSCF(
    RewriterBase &rewriter, TilingInterface root,
    const scf::SCFTileAndFuseOptions &options) {

  // This transformation is only valid for ops that return values (i.e. not
  // valid to use with operations that have memref operands).
  if (!root->getNumResults()) {
    return rewriter.notifyMatchFailure(
        root, "invalid pattern for op with no results");
  }

  // 1.  Tile root op and Fuse Producers.
  FailureOr<scf::SCFTileAndFuseResult> tiledResults =
      scf::tileConsumerAndFuseProducersUsingSCF(rewriter, root, options);

  if (failed(tiledResults)) {
    return rewriter.notifyMatchFailure(
        root, "failed to tile root and fuse producers");
  }

  // 2. Replace the producers with the tiled verison.
  SmallVector<Operation *> opsToReplace = {root};
  llvm::append_range(opsToReplace, tiledResults->fusedProducers);
  for (Operation *toReplace : opsToReplace) {
    for (OpResult res : toReplace->getResults())
      if (auto replacement = tiledResults->replacements.lookup(res)) {
        rewriter.replaceAllUsesWith(res, replacement);
      }

    if (toReplace->use_empty()) {
      rewriter.eraseOp(toReplace);
    }
  }

  // 3. Typically, the consumers of the tiled operation are slices of the
  //    results of the tiled operation. These are expressed in IR using
  //    `tensor.insert_slice` operations, whose outputs are the operands of the
  //    untiled operation. Create a worklist of these `tensor.insert_siices`
  //    operations. If the consumers of the source of the `tensor.insert_slices`
  //    can be tiled such that the tiled value is generated in-place, that
  //    effectively tiles + fuses the operations.
  auto addCandidateSlices = [](Operation *fusedOp,
                               std::queue<tensor::InsertSliceOp> &candidates) {
    for (auto *userOp : fusedOp->getResults().getUsers()) {
      if (auto sliceOp = llvm::dyn_cast<tensor::InsertSliceOp>(userOp)) {
        candidates.push(sliceOp);
      }
    }
  };

  // Collect the candidate slices which can be potential consumers that can be
  // fused.
  std::queue<tensor::InsertSliceOp> candidates;
  addCandidateSlices(tiledResults->tiledAndFusedOps.front(), candidates);

  while (!candidates.empty()) {

    // Traverse the slices in BFS fashion.
    tensor::InsertSliceOp candidateSliceOp = candidates.front();
    candidates.pop();

    FailureOr<scf::SCFFuseConsumerOfSliceResult> fusedResult =
        mlir::scf::tileAndFuseConsumerOfSlice(rewriter, candidateSliceOp);
    if (failed(fusedResult)) {
      LLVM_DEBUG(llvm::dbgs() << "failed to fuse consumer of slice: "
                              << candidateSliceOp << "\n");
      continue;
    }

    // Replace the original consumer operation with the tiled implementation.
    rewriter.replaceOp(fusedResult->origConsumerOperand->getOwner(),
                       fusedResult->tiledOps.front());

    // The result of the fused conumers might themselved be slices of
    // values produced by operations that implement the `TilingInterface`.
    // Add these operations to the worklist.
    addCandidateSlices(fusedResult->tiledAndFusedConsumerOperand->getOwner(),
                       candidates);
  }
  return success();
}

static LogicalResult tileRootAndFuseProducerConsumer(IRRewriter &rewriter,
                                                     TilingInterface rootOp,
                                                     int64_t tilingLevel) {

  SmallVector<OpFoldResult> tileSizes =
      getLoweringConfig(rootOp).getTilingLevelSizes(rewriter, tilingLevel,
                                                    rootOp);
  int64_t numLoops = rootOp.getLoopIteratorTypes().size();
  if (tileSizes.size() > numLoops)
    return failure();

  scf::SCFTilingOptions tilingOptions;
  tilingOptions.setTileSizes(tileSizes);

  scf::SCFTileAndFuseOptions tileAndFuseOptions;
  tileAndFuseOptions.setTilingOptions(tilingOptions);

  return tileRootAndFuseProducerConsumerUsingSCF(rewriter, rootOp,
                                                 tileAndFuseOptions);
}

/// This pass starts with the first TilingInterface operation that has
/// lowering_config attribute, tiles the op and fuses its  consumers and
/// producers recursively. The `tilingLevel` must be specified. It picks the
/// `tilingLevel`-th list as tiling sizes from lowering_config.
struct LLVMCPUTileRootAndFuseProducerConsumer
    : LLVMCPUTileRootAndFuseProducerConsumerBase<
          LLVMCPUTileRootAndFuseProducerConsumer> {
  LLVMCPUTileRootAndFuseProducerConsumer(int64_t tilingLevel = -1) {
    this->tilingLevel.setValue(tilingLevel);
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, affine::AffineDialect,
                    linalg::LinalgDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override;
};

void LLVMCPUTileRootAndFuseProducerConsumer::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();

  IRRewriter rewriter(funcOp);

  SmallVector<Operation *> tiledOpsWithLoweringConfig;
  // Collect ops with tilingInterface that has the loweringconfig attached.
  funcOp->walk([&](TilingInterface target) {
    if (IREE::Codegen::LoweringConfigAttrInterface loweringConfig =
            getLoweringConfig(target)) {
      if (loweringConfig.hasTilingLevel(tilingLevel)) {
        tiledOpsWithLoweringConfig.push_back(target);
      }
    }
  });

  FailureOr<Operation *> rootOp = getRootOperation(tiledOpsWithLoweringConfig);

  if (failed(rootOp)) {
    funcOp.emitError()
        << "no operation found with the lowering_config attribute.\n";
    return signalPassFailure();
  }

  if (failed(tileRootAndFuseProducerConsumer(
          rewriter, dyn_cast<TilingInterface>(rootOp.value()),
          tilingLevel.getValue()))) {
    funcOp.emitError() << "tiling of level " << tilingLevel.getValue()
                       << " failed\n";
    return signalPassFailure();
  }

  RewritePatternSet patterns =
      linalg::getLinalgTilingCanonicalizationPatterns(context);
  scf::populateSCFForLoopCanonicalizationPatterns(patterns);
  tensor::populateFoldTensorEmptyPatterns(patterns);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  // Pull in tensor dialect canonicalization patterns to fold tensor.cast
  // into producers when possible.
  context->getLoadedDialect<tensor::TensorDialect>()
      ->getCanonicalizationPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    LLVM_DEBUG(llvm::dbgs() << "----- cleanup failed -----\n");
    return signalPassFailure();
  }
}
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUTileRootAndFuseProducerConsumer(int64_t tilingLevel) {
  return std::make_unique<LLVMCPUTileRootAndFuseProducerConsumer>(tilingLevel);
}
} // namespace mlir::iree_compiler
