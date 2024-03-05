// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/TileSizeSelection.h"
#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-unsupported-scalability-to-loops"
#define VEC_DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir::iree_compiler {

namespace {

class LLVMCPUUnsupportedScalabilityToLoopsPass
    : public LLVMCPUUnsupportedScalabilityToLoopsBase<
          LLVMCPUUnsupportedScalabilityToLoopsPass> {
public:
  using LLVMCPUUnsupportedScalabilityToLoopsBase::
      LLVMCPUUnsupportedScalabilityToLoopsBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, linalg::LinalgDialect, scf::SCFDialect>();
  }

  void runOnOperation() override;
};

static bool opKnownToSupport2DScalableVectorizationWithArmSME(Operation *op) {
  return isa<linalg::MatmulOp, linalg::MatmulTransposeAOp, linalg::FillOp>(op);
}

struct DropUnsupportedScalableDimsFromTilingInterfaceOps
    : public OpInterfaceRewritePattern<TilingInterface> {
  DropUnsupportedScalableDimsFromTilingInterfaceOps(MLIRContext *context,
                                                    bool assumeArmSME)
      : OpInterfaceRewritePattern(context), assumeArmSME(assumeArmSME) {}

  LogicalResult matchAndRewrite(TilingInterface op,
                                PatternRewriter &rewriter) const override {
    // Note: This rewrite is currently only required for ArmSME (which is the
    // only target that currently has some concept of 2D scalability).
    auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
    bool isArmSME = assumeArmSME || hasSMEFeature(targetAttr);
    if (!isArmSME || opKnownToSupport2DScalableVectorizationWithArmSME(op))
      return failure();

    auto loweringConfigAttr = getLoweringConfig(op);
    if (!loweringConfigAttr)
      return failure();

    auto tileSizes = loweringConfigAttr.getTileSizeVals();
    auto scalableFlags = loweringConfigAttr.getScalableTileFlagVals();

    // Drop scalable dimensions from leading tiling levels first (and leading
    // dimensions within tiling levels). This works out as dropping scalability
    // from leading dimensions of the vector type.
    int64_t firstTilingLevelWithScalableDims = -1;
    int64_t numScalableDims = 0;
    for (auto [level, levelScalableFlags] : llvm::enumerate(scalableFlags)) {
      numScalableDims += llvm::count(levelScalableFlags, true);
      if (numScalableDims > 0 && firstTilingLevelWithScalableDims == -1)
        firstTilingLevelWithScalableDims = level;
    }

    if (numScalableDims <= 1)
      return failure();

    auto levelTileSizes = tileSizes[firstTilingLevelWithScalableDims];
    auto levelScalableFlags = scalableFlags[firstTilingLevelWithScalableDims];

    SmallVector<int64_t> loopTileSizes;
    SmallVector<bool> newScalableFlags;
    for (auto [flag, size] : llvm::zip(levelScalableFlags, levelTileSizes)) {
      if (flag && numScalableDims >= 2) {
        --numScalableDims;
        loopTileSizes.push_back(size);
        newScalableFlags.push_back(false);
      } else {
        loopTileSizes.push_back(0);
        newScalableFlags.push_back(flag);
      }
    }

    // Re-tile the operation with some scalability dropped. This introduces
    // loops for previously scalable vector/tile sizes.
    scf::SCFTilingOptions options{};
    setSCFTileSizes(options, op, loopTileSizes, {});
    auto tilingResult =
        scf::tileUsingSCF(rewriter, cast<TilingInterface>(op), options);
    if (failed(tilingResult))
      return failure();

    // Update the lowering config of the new tiled operations.
    scalableFlags[firstTilingLevelWithScalableDims] = newScalableFlags;
    auto newLoweringConfig = IREE::Codegen::LoweringConfigAttr::get(
        getContext(), tileSizes, scalableFlags);
    for (auto *newOp : tilingResult->tiledOps) {
      if (isa<TilingInterface>(newOp))
        setLoweringConfig(newOp, newLoweringConfig);
    }

    rewriter.replaceOp(op, tilingResult->replacements);
    return success();
  };

private:
  bool assumeArmSME{false};
};

void LLVMCPUUnsupportedScalabilityToLoopsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<DropUnsupportedScalableDimsFromTilingInterfaceOps>(
      patterns.getContext(), assumeArmSME);

  if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                      std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUUnsupportedScalabilityToLoopsPass() {
  return std::make_unique<LLVMCPUUnsupportedScalabilityToLoopsPass>();
}

} // namespace mlir::iree_compiler
