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
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Common/DestructiveUpdateUtils.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/PartitionableLoopsInterface.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-tile-and-distribute-to-workgroups"

namespace mlir {
namespace iree_compiler {

//===---------------------------------------------------------------------===//
// Patterns and methods for tile and distribute of Linalg ops to workgroups.
//===---------------------------------------------------------------------===//

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

template <typename OpTy>
static Value buildHALWorkgroupInfoOp(OpBuilder &b, unsigned dim) {
  return b.template create<OpTy>(b.getInsertionPoint()->getLoc(), dim);
}

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

  // Add a marker to the last operation in the list.
  auto marker = StringAttr::get(context, "__workgroup_tiling__");
  computeOps.back()->setAttr(linalg::LinalgTransforms::kLinalgTransformMarker,
                             marker);

  // Configure the linalg options.
  // Distribute the ops using the flow workgroup ID/Count operations.
  static linalg::LinalgLoopDistributionOptions workgroupDistributionOptions = {
      [](OpBuilder &builder, Location loc, ArrayRef<Range> parallelLoopRanges) {
        auto numParallelDims = parallelLoopRanges.size();

        SmallVector<linalg::ProcInfo, 3> procInfo(numParallelDims);
        for (size_t dim = 0; dim < numParallelDims; ++dim) {
          procInfo[numParallelDims - dim - 1] = {
              buildHALWorkgroupInfoOp<IREE::HAL::InterfaceWorkgroupIDOp>(
                  builder, dim),
              buildHALWorkgroupInfoOp<IREE::HAL::InterfaceWorkgroupCountOp>(
                  builder, dim)};
        }
        return procInfo;
      },
      {linalg::DistributionMethod::Cyclic, linalg::DistributionMethod::Cyclic,
       linalg::DistributionMethod::Cyclic},
      DenseMap<StringRef,
               std::function<linalg::ProcInfo(OpBuilder &, Location)>>()};

  // Tile size selection function. Sets the tile size now to
  // hal.interface.workgroup.size op, with 0 for the innermost parallel loop
  // partitioned, 1 for the next outermost loop partitioned and so on.  Use the
  // workgroup size as a proxy for tile size here. At the flow level this
  // represents the "workload" per processors and is not necessarily tied to the
  // workgroup size.

  // TODO(#...): Refactor this pass to directly take the tile sizes from lower
  // configuration for the first level of tiling.
  auto tileSizeFn = [&](OpBuilder &builder,
                        Operation *op) -> SmallVector<Value, 4> {
    auto interfaceOp = dyn_cast<IREE::Flow::PartitionableLoopsInterface>(op);
    if (!interfaceOp) return {};
    SmallVector<unsigned> partitionedLoops =
        interfaceOp.getPartitionableLoops(kNumMaxParallelDims);
    if (partitionedLoops.empty()) return {};
    unsigned maxDepth = partitionedLoops.back() + 1;

    // Set all loops not partitioned to tile size 0. and those partitioned to
    // `flow.workgroup.size`.
    auto zero = builder.create<arith::ConstantIndexOp>(op->getLoc(), 0);
    SmallVector<Value, 4> useTileSizes(maxDepth, zero);
    llvm::DenseSet<unsigned> partitionedLoopsSet;
    partitionedLoopsSet.insert(partitionedLoops.begin(),
                               partitionedLoops.end());
    unsigned currFlowDim = 0;
    for (size_t dim = maxDepth; dim > 0; dim--) {
      if (partitionedLoopsSet.count(dim - 1)) {
        useTileSizes[dim - 1] =
            buildHALWorkgroupInfoOp<IREE::HAL::InterfaceWorkgroupSizeOp>(
                builder, currFlowDim++);
      }
    }
    return useTileSizes;
  };

  auto linalgTilingOptions =
      linalg::LinalgTilingOptions()
          .setDistributionOptions(workgroupDistributionOptions)
          .setLoopType(linalg::LinalgTilingLoopType::Loops)
          .setTileSizeComputationFunction(tileSizeFn);

  RewritePatternSet patterns(context);
  patterns.insert<TileAndDistributeLinalgOpsPattern,
                  IREE::LinalgExt::TiledOpInterfaceTilingPattern>(
      context, linalgTilingOptions, linalg::LinalgTransformationFilter(marker));
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }

  // Apply linalg tiling optimization patterns.
  RewritePatternSet canonicalizationPatterns(context);
  linalg::populateLinalgTilingCanonicalizationPatterns(
      canonicalizationPatterns);
  memref::populateResolveRankedShapeTypeResultDimsPatterns(
      canonicalizationPatterns);
  if (failed(applyPatternsAndFoldGreedily(
          funcOp, std::move(canonicalizationPatterns)))) {
    return signalPassFailure();
  }

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
}

std::unique_ptr<OperationPass<FuncOp>>
createTileAndDistributeToWorkgroupsPass() {
  return std::make_unique<TileAndDistributeToWorkgroupsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
