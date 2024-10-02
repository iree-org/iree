// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <functional>
#include <numeric>
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-gpu-fuse-and-hoist-parallel-loops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler::IREE::GPU {

#define GEN_PASS_DEF_FUSEANDHOISTPARALLELLOOPSPASS
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h.inc"

namespace {
struct FuseAndHoistParallelLoopsPass final
    : impl::FuseAndHoistParallelLoopsPassBase<FuseAndHoistParallelLoopsPass> {
  void runOnOperation() override;
};
} // namespace

static bool forallTripCountMatchesWorkgroupSize(scf::ForallOp forallOp,
                                                int64_t flatWorkgroupSize) {
  // True for lane mapped loops.
  if (forallOpHasMappingType<IREE::GPU::LaneIdAttr>(forallOp)) {
    return true;
  }

  // All other loops must be mapped to threads to compare. Also give up on
  // non-normalized loops.
  if (!forallOpHasMappingType<gpu::GPUThreadMappingAttr>(forallOp) ||
      !forallOp.isNormalized()) {
    return false;
  }

  int64_t tripCount = 1;
  for (OpFoldResult ub : forallOp.getMixedUpperBound()) {
    std::optional<int64_t> maybeConstantUb = getConstantIntValue(ub);
    if (!maybeConstantUb) {
      return false;
    }
    tripCount *= *maybeConstantUb;
  }
  return tripCount == flatWorkgroupSize;
}
struct FuseForalls final : OpRewritePattern<scf::ForallOp> {
  using OpRewritePattern::OpRewritePattern;
  FuseForalls(MLIRContext *ctx, int64_t flatWorkgroupSize, PatternBenefit b = 1)
      : OpRewritePattern<scf::ForallOp>(ctx, b),
        flatWorkgroupSize(flatWorkgroupSize) {}
  LogicalResult matchAndRewrite(scf::ForallOp producerForall,
                                PatternRewriter &rewriter) const override {
    if (!producerForall->hasOneUse()) {
      return rewriter.notifyMatchFailure(producerForall,
                                         "multi-use producer forall");
    }

    SmallVector<Operation *> consumerChain;
    Operation *currProducer = *producerForall->user_begin();
    while (currProducer && currProducer->hasOneUse()) {
      consumerChain.push_back(currProducer);
      if (!isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(currProducer)) {
        break;
      }
      currProducer = *currProducer->user_begin();
    }

    auto consumerForall = currProducer->getParentOfType<scf::ForallOp>();
    if (!consumerForall || !forallTripCountMatchesWorkgroupSize(
                               consumerForall, flatWorkgroupSize)) {
      return rewriter.notifyMatchFailure(
          producerForall,
          "no consumer forall with trip count matching workgroup size");
    }

    // TODO: Allow extracting multiple uses within the same consumer loop. Still
    // single producer single consumer loop, but multiple uses within the
    // consumer.
    if (!producerForall->hasOneUse()) {
      return failure();
    }

    return fuseForallIntoConsumer(rewriter, producerForall, consumerForall,
                                  consumerChain);
  }

private:
  int64_t flatWorkgroupSize;
};

struct FuseTilableDestinationProducers final : OpRewritePattern<scf::ForallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ForallOp forallOp,
                                PatternRewriter &rewriter) const override {
    TilingInterface tileableProducer;
    tensor::ExtractSliceOp sliceOp;
    for (auto iterArg : forallOp.getRegionIterArgs()) {
      for (auto user : iterArg.getUsers()) {
        sliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
        if (sliceOp) {
          break;
        }
      }
      if (!sliceOp) {
        continue;
      }
      tileableProducer = forallOp.getTiedLoopInit(iterArg)
                             ->get()
                             .getDefiningOp<TilingInterface>();
      if (tileableProducer) {
        break;
      }
    }
    if (!tileableProducer) {
      return failure();
    }

    SmallVector<LoopLikeOpInterface> loops = {forallOp};
    rewriter.startOpModification(forallOp);
    std::optional<scf::SCFFuseProducerOfSliceResult> fusionResult =
        mlir::scf::tileAndFuseProducerOfSlice(rewriter, sliceOp, loops);
    if (!fusionResult) {
      return failure();
    }
    rewriter.finalizeOpModification(forallOp);
    return success();
  }
};

struct FuseTilableSliceProducers final
    : OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    if (sliceOp->use_empty()) {
      return failure();
    }
    auto tilableProducer = sliceOp.getSource().getDefiningOp<TilingInterface>();
    if (!tilableProducer) {
      return failure();
    }

    auto parentForall = sliceOp->getParentOfType<scf::ForallOp>();
    if (!parentForall) {
      return failure();
    }

    SmallVector<LoopLikeOpInterface> loops = {parentForall};
    std::optional<scf::SCFFuseProducerOfSliceResult> fusionResult =
        mlir::scf::tileAndFuseProducerOfSlice(rewriter, sliceOp, loops);
    if (!fusionResult) {
      return failure();
    }
    return success();
  }
};

struct FuseTilableForallConsumers final
    : OpInterfaceRewritePattern<TilingInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(TilingInterface tilableOp,
                                PatternRewriter &rewriter) const override {
    // Currently consumer fusion requires DPS, and we don't want to fuse through
    // inits anyway.
    auto dpsOp = dyn_cast<DestinationStyleOpInterface>(*tilableOp);
    if (!dpsOp) {
      return failure();
    }

    tensor::ParallelInsertSliceOp producerSlice;
    for (auto operand : dpsOp.getDpsInputs()) {
      auto forallProducer = operand.getDefiningOp<scf::ForallOp>();
      if (!forallProducer) {
        continue;
      }
      Value iterArg = forallProducer.getTiedBlockArgument(
          forallProducer.getTiedOpOperand(cast<OpResult>(operand)));

      for (auto user : iterArg.getUsers()) {
        auto sliceOp = dyn_cast<tensor::ParallelInsertSliceOp>(user);
        if (sliceOp && sliceOp.getDest() == iterArg) {
          producerSlice = sliceOp;
          break;
        }
      }
      if (producerSlice) {
        break;
      }
    }

    if (!producerSlice) {
      return failure();
    }

    FailureOr<scf::SCFFuseConsumerOfSliceResult> fuseConsumerResults =
        scf::tileAndFuseConsumerOfSlice(rewriter, producerSlice);
    if (failed(fuseConsumerResults)) {
      return failure();
    }
    return success();
  }
};

void FuseAndHoistParallelLoopsPass::runOnOperation() {
  MLIRContext *context = &getContext();

  FunctionOpInterface funcOp = getOperation();

  // Try to get the flat workgroup size if possible.
  std::optional<int64_t> maybeFlatWorkgroupSize = std::nullopt;
  if (std::optional<SmallVector<int64_t>> workgroupSize =
          getWorkgroupSize(funcOp)) {
    maybeFlatWorkgroupSize =
        std::accumulate(workgroupSize->begin(), workgroupSize->end(), 1,
                        std::multiplies<int64_t>());
  }

  // First run the hoisting and fusion patterns.
  {
    RewritePatternSet patterns(context);
    // These two patterns are run to a fixed point, allowing fusion within
    // potentially nested loops, hoisting from said loops, and continued fusion.
    if (maybeFlatWorkgroupSize) {
      // Forall fusion requires knowing the workgroup size to verify the fusion
      // is valid.
      patterns.add<FuseForalls>(context, *maybeFlatWorkgroupSize,
                                /*benefit=*/1);
    }
    patterns.add<FuseTilableForallConsumers>(context);
    populateForallLoopHoistingPattern(patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  LDBG("After fusing and hoisting loops\n" << funcOp);

  // After hoisting parallel loops, try to fuse in any newly revealed consumers
  // and destinations.
  // TODO: Move the consumer fusion pattern to an explicit worklist rather than
  // using the GreedyPatternRewriter.
  {
    RewritePatternSet patterns(context);
    patterns.add<FuseTilableDestinationProducers>(context);
    patterns.add<FuseTilableForallConsumers>(context);
    tensor::populateFoldTensorEmptyPatterns(patterns);
    scf::ForallOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  LDBG("After fusing new consumers\n" << funcOp);

  // Finally try to do any new producer fusions.
  {
    RewritePatternSet patterns(context);
    patterns.add<FuseTilableDestinationProducers>(context);
    patterns.add<FuseTilableSliceProducers>(context);
    tensor::populateFoldTensorEmptyPatterns(patterns);
    scf::ForallOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  LDBG("After fusing new producers\n" << funcOp);
}

} // namespace mlir::iree_compiler::IREE::GPU
