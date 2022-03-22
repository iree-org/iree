// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//=== TileAndDistributeToWorkgroupsPass.cpp - Tile to workgroups pass ----===//
//
// This pass distributes the operations within the module to workgroups. This
// pass is created to move tile and distribution out of flow level and into
// the backends. For now this is mostly a bridge pass to connect things during
// the transition, and eventually might just be deprecated in favor of a
// utility method.
//
//===---------------------------------------------------------------------===//
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Transforms.h"
#include "iree/compiler/Codegen/Common/DestructiveUpdateUtils.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/PartitionableLoopsInterface.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-tile-and-distribute-to-workgroups"

namespace mlir {
namespace iree_compiler {

//===---------------------------------------------------------------------===//
// Patterns and methods for tile and distribute of Linalg ops to workgroups.
//===---------------------------------------------------------------------===//

// Get the lowering configuration for the operation within the dispatch.
// This looks for tile sizes by looking for lowering configuration.
static FailureOr<SmallVector<int64_t>> getTileSizesFromLoweringConfig(
    ArrayRef<Operation *> computeOps, MLIRContext *context) {
  if (computeOps.empty()) return SmallVector<int64_t>{};

  Optional<SmallVector<int64_t>> distributedTileSizes;
  for (auto op : computeOps) {
    auto partitionbleLoopInterface =
        dyn_cast<IREE::Flow::PartitionableLoopsInterface>(op);
    if (!partitionbleLoopInterface) continue;
    IREE::Codegen::LoweringConfigAttr currLoweringConfig =
        getLoweringConfig(op);
    if (!currLoweringConfig) continue;
    SmallVector<unsigned> partitionableLoops =
        partitionbleLoopInterface.getPartitionableLoops(kNumMaxParallelDims);
    SmallVector<int64_t> tileSizes = currLoweringConfig.getTileSizeVals(0);
    SmallVector<int64_t> currDistributedTileSizes;
    if (!partitionableLoops.empty()) {
      currDistributedTileSizes.resize(partitionableLoops.back() + 1, 0);
    }
    for (auto loopID : partitionableLoops) {
      if (loopID < tileSizes.size()) {
        currDistributedTileSizes[loopID] = tileSizes[loopID];
      }
    }
    if (distributedTileSizes) {
      if (currDistributedTileSizes != distributedTileSizes) {
        // Inconsistent distributed tile sizes. Abort.
        return static_cast<LogicalResult>(
            computeOps.front()->emitOpError("inconsistent distribution of ops "
                                            "for first level of distribution"));
      }
    } else {
      distributedTileSizes = currDistributedTileSizes;
    }
  }
  if (distributedTileSizes) {
    return distributedTileSizes.getValue();
  }
  return SmallVector<int64_t>{};
}

// Pull in producers into the tiled operation.
static void pullInProducers(linalg::LinalgOp tiledOp,
                            ValueRange untiledOperands,
                            PatternRewriter &rewriter) {
  for (auto en : llvm::enumerate(untiledOperands)) {
    auto producer = en.value().getDefiningOp<linalg::LinalgOp>();
    if (!producer) continue;

    OpResult opResult = en.value().cast<OpResult>();
    auto maybeFusionInfo = linalg::fuseProducerOfTensor(
        rewriter, producer->getResult(opResult.getResultNumber()),
        tiledOp->getOpOperand(en.index()));
    if (failed(maybeFusionInfo)) continue;

    // If the fusion was successfull recurse over the current producers operands
    // and fuse them in as well.
    SmallVector<Value> origProducerOperands =
        producer.getInputAndOutputOperands();
    pullInProducers(maybeFusionInfo->fusedProducer, origProducerOperands,
                    rewriter);
  }
}

namespace {
// Rewrite pattern to ensure only ops with tensor semantics are tiled.
struct TileAndDistributeLinalgOpsPattern : public linalg::LinalgTilingPattern {
  using Base = linalg::LinalgTilingPattern;
  TileAndDistributeLinalgOpsPattern(MLIRContext *context,
                                    linalg::LinalgTilingOptions options,
                                    linalg::LinalgTransformationFilter marker,
                                    PatternBenefit benefit = 1)
      : Base(context, options, marker, benefit) {}

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> untiledOperands = linalgOp.getInputAndOutputOperands();
    FailureOr<linalg::TiledLinalgOp> tiledLinalgOpOr =
        Base::returningMatchAndRewrite(linalgOp, rewriter);
    if (failed(tiledLinalgOpOr)) {
      return failure();
    }
    if (tiledLinalgOpOr->loops.empty()) {
      // If there are no loops, there is nothing to do.
      return success();
    }
    pullInProducers(tiledLinalgOpOr->op, untiledOperands, rewriter);
    return success();
  }
};
}  // namespace

namespace {
struct TileAndDistributeToWorkgroupsPass
    : public TileAndDistributeToWorkgroupsBase<
          TileAndDistributeToWorkgroupsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, IREE::Flow::FlowDialect,
                    IREE::HAL::HALDialect, linalg::LinalgDialect,
                    scf::SCFDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override;
};
}  // namespace

void TileAndDistributeToWorkgroupsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  FuncOp funcOp = getOperation();
  if (!isEntryPoint(funcOp)) return;

  SmallVector<Operation *> computeOps;
  SmallVector<LoopTilingAndDistributionInfo> tiledLoops;
  if (failed(getComputeOps(funcOp, computeOps, tiledLoops))) {
    return signalPassFailure();
  }
  if (!tiledLoops.empty()) {
    // The entry point already has distribution to workgroups. Do nothing.
    return;
  }
  if (computeOps.empty()) {
    // Ignore other operations.
    return;
  }

  // Get the tile sizes to use from lowering configuration if set.
  FailureOr<SmallVector<int64_t>> configTileSizes =
      getTileSizesFromLoweringConfig(computeOps, context);
  if (failed(configTileSizes)) {
    return signalPassFailure();
  }
  ArrayRef<int64_t> configTileSizesRef(configTileSizes.getValue());

  // Add a marker to the last operation in the list.
  auto marker = StringAttr::get(context, "__workgroup_tiling__");
  computeOps.back()->setAttr(linalg::LinalgTransforms::kLinalgTransformMarker,
                             marker);

  // Configure the linalg options.
  // Tile size selection function.
  auto tileSizeFn = [&](OpBuilder &builder,
                        Operation *op) -> SmallVector<Value, 4> {
    // Check if tile sizes are deduced from the configuration. If so use those.
    return llvm::to_vector<4>(
        llvm::map_range(configTileSizesRef, [&](int64_t ts) -> Value {
          return builder.create<arith::ConstantIndexOp>(op->getLoc(), ts);
        }));
  };

  auto linalgTilingOptions =
      linalg::LinalgTilingOptions()
          .setDistributionOptions(getIREELinalgLoopDistributionOptions())
          .setLoopType(linalg::LinalgTilingLoopType::Loops)
          .setTileSizeComputationFunction(tileSizeFn);

  RewritePatternSet patterns(context);
  patterns.insert<TileAndDistributeLinalgOpsPattern,
                  IREE::LinalgExt::TiledOpInterfaceTilingPattern>(
      context, linalgTilingOptions, linalg::LinalgTransformationFilter(marker));
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "--- After Tile + Distribute ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // Apply linalg tiling optimization patterns.
  RewritePatternSet canonicalizationPatterns(context);
  linalg::populateLinalgTilingCanonicalizationPatterns(
      canonicalizationPatterns);
  if (failed(applyPatternsAndFoldGreedily(
          funcOp, std::move(canonicalizationPatterns)))) {
    return signalPassFailure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "--- After Canonicalize ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // Rewrite destructive updates and ensure no remaining store remains to the
  // full output.

  // TODO(#...): Use of the destructive update rewrite is a hack! There needs to
  // be a way to generate loops as we need, and use the tiled op generation
  // implementation. This should be possible after moving everything to use the
  // `TilingInterface`.
  if (failed(rewriteLinalgDestructiveUpdates(funcOp))) {
    funcOp->emitError("Failed to rewrite destructive updates in:\n")
        << *funcOp.getOperation();
    return signalPassFailure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "--- After Rewriting destructive updates ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // After rewriting destructive updates, there might be uses of compute
  // operations only in `tensor.dim` ops. Resolve these.
  RewritePatternSet resolveDimOps(context);
  memref::populateResolveRankedShapeTypeResultDimsPatterns(resolveDimOps);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(resolveDimOps)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<FuncOp>>
createTileAndDistributeToWorkgroupsPass() {
  return std::make_unique<TileAndDistributeToWorkgroupsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
