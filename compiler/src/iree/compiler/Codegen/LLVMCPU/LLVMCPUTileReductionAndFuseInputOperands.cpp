// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
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

#define DEBUG_TYPE "iree-llvmcpu-tile-reduction-and-fuse-input-operands"

namespace mlir::iree_compiler {

namespace {

/// This pass tiles the reduction operations and fuses the input operands of
/// the tiled reduction op with the producer of the input operand.
struct LLVMCPUTileReductionAndFuseInputOperands
    : LLVMCPUTileReductionAndFuseInputOperandsBase<
          LLVMCPUTileReductionAndFuseInputOperands> {
  LLVMCPUTileReductionAndFuseInputOperands(int64_t tilingLevel = -1) {
    this->tilingLevel.setValue(tilingLevel);
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, affine::AffineDialect,
                    linalg::LinalgDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override;
};

/// Starting from `op` walk all operands backwards to find all
/// potentially fusable operations, i.e. operations that implement
/// the `TilingInterface`.
static void collectTiledAndFusedOps(Operation *rootOp,
                                    llvm::SmallDenseSet<Operation *> &result) {
  SmallVector<Operation *> worklist;
  worklist.push_back(rootOp);
  result.insert(rootOp);
  while (!worklist.empty()) {
    Operation *current = worklist.pop_back_val();
    for (OpOperand &operand : current->getOpOperands()) {
      Operation *producer = operand.get().getDefiningOp();
      if (!producer || !isa<TilingInterface>(producer) ||
          result.count(producer))
        continue;
      worklist.push_back(producer);
      result.insert(producer);
    }
  }
}

/// Apply a tile and fuse transformation to all payload ops and store both the
/// tiled operation as well as the created tile loops.
static LogicalResult
applyTileAndFuseToEachRoot(RewriterBase &rewriter,
                           TilingInterface tilingInterfaceOp,
                           int64_t tilingLevel) {

  mlir::DominanceInfo dominanceInfo(tilingInterfaceOp);
  llvm::SmallDenseSet<Operation *> tiledAndFusedOps;
  collectTiledAndFusedOps(tilingInterfaceOp, tiledAndFusedOps);

  llvm::DenseSet<Operation *> yieldReplacementsFor;
  for (auto op : tiledAndFusedOps) {
    if (llvm::any_of(op->getUsers(), [&](Operation *user) {
          return dominanceInfo.properlyDominates(tilingInterfaceOp, user);
        })) {
      yieldReplacementsFor.insert(op);
    }
  }

  SmallVector<OpFoldResult> tileSizes =
      getLoweringConfig(tilingInterfaceOp)
          .getTilingLevelSizes(rewriter, tilingLevel, tilingInterfaceOp);

  // Pad the tile sizes with zero.
  auto zero = rewriter.getIndexAttr(0);
  int64_t numLoops = tilingInterfaceOp.getLoopIteratorTypes().size();
  if (tileSizes.size() > numLoops) {
    return failure();
  }
  while (tileSizes.size() < numLoops) {
    tileSizes.push_back(zero);
  }

  scf::SCFTilingOptions tilingOptions;
  tilingOptions.setTileSizes(tileSizes);

  scf::SCFTileAndFuseOptions tileAndFuseOptions;
  tileAndFuseOptions.setTilingOptions(tilingOptions);

  scf::SCFTileAndFuseOptions::ControlFnTy controlFn =
      [&](tensor::ExtractSliceOp candidateSliceOp, OpResult originalProducer,
          bool isDestinationOperand) {
        Operation *owner = originalProducer.getOwner();
        bool yieldProducerReplacement = yieldReplacementsFor.contains(owner);
        // Do not fuse destination operands.
        bool shouldFuse = !isDestinationOperand;
        return std::make_tuple(shouldFuse, yieldProducerReplacement);
      };
  tileAndFuseOptions.setFusionControlFn(controlFn);

  FailureOr<scf::SCFTileAndFuseResult> tiledResults =
      scf::tileConsumerAndFuseProducersUsingSCF(rewriter, tilingInterfaceOp,
                                                tileAndFuseOptions);
  if (failed(tiledResults)) {
    return failure();
  }

  // Perform the replacement of tiled and fused values.
  SmallVector<Operation *> opsToReplace{tilingInterfaceOp};
  llvm::append_range(opsToReplace, tiledResults->fusedProducers);
  for (Operation *toReplace : opsToReplace) {
    for (OpResult res : toReplace->getResults())
      if (auto replacement = tiledResults->replacements.lookup(res)) {
        Operation *replacementOp = replacement.getDefiningOp();
        rewriter.replaceUsesWithIf(res, replacement, [&](OpOperand &use) {
          Operation *user = use.getOwner();
          return dominanceInfo.properlyDominates(replacementOp, user);
        });
      }

    if (toReplace->use_empty()) {
      rewriter.eraseOp(toReplace);
    }
  }

  return success();
}

// Returns the first reduction operation with lowering config at the given
// tiling level.
static std::optional<TilingInterface>
getReductionOpWithLoweringConfig(Operation *funcOp, int64_t tilingLevel) {
  TilingInterface target = nullptr;
  funcOp->walk([&](TilingInterface op) {
    if (!llvm::any_of(op.getLoopIteratorTypes(), [](auto iter) {
          return iter == utils::IteratorType::reduction;
        })) {
      WalkResult::advance();
    }
    if (IREE::Codegen::LoweringConfigAttrInterface loweringConfig =
            getLoweringConfig(op)) {
      if (loweringConfig.hasTilingLevel(tilingLevel)) {
        target = op;
        WalkResult::interrupt();
      }
    }
  });
  return target;
}

void LLVMCPUTileReductionAndFuseInputOperands::runOnOperation() {
  if (tilingLevel == -1) {
    LLVM_DEBUG(llvm::dbgs() << "tilingLevel not set, skip tiling\n");
    return;
  }
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();

  auto targetOp = getReductionOpWithLoweringConfig(funcOp, tilingLevel);

  if (*targetOp) {
    IRRewriter rewriter(funcOp);
    if (failed(applyTileAndFuseToEachRoot(rewriter, targetOp.value(),
                                          tilingLevel))) {
      funcOp.emitError() << "tiling of level " << tilingLevel << " failed\n";
      return signalPassFailure();
    }
  }

  RewritePatternSet patterns =
      linalg::getLinalgTilingCanonicalizationPatterns(context);
  scf::populateSCFForLoopCanonicalizationPatterns(patterns);
  tensor::populateFoldTensorEmptyPatterns(patterns);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  context->getLoadedDialect<tensor::TensorDialect>()
      ->getCanonicalizationPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    LLVM_DEBUG(llvm::dbgs() << "----- cleanup failed -----\n");
    return signalPassFailure();
  }
}
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUTileReductionAndFuseInputOperandsPass(int64_t tilingLevel) {
  return std::make_unique<LLVMCPUTileReductionAndFuseInputOperands>(
      tilingLevel);
}

} // namespace mlir::iree_compiler
