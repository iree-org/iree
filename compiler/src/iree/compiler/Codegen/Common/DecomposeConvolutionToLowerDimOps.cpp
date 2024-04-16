// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/TileSizeSelection.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

namespace {

/// Computes a "decomposed" lowering config attribute for a conv OP
///
/// This method complements the patterns to decompose 2D convolutions into 1D
/// convs. Specifically, it will update the lowering config attached to a Conv
/// Op in a way that matches the "decomposition" patterns.
///
/// At the moment only Depthwise HWC convolutions are supported.
static FailureOr<IREE::Codegen::LoweringConfigAttr>
computeNewLCA(SmallVector<Operation *> computeOps, MLIRContext *context) {

  // 0.1 Double-check that there's only one convolution Op among the compute Ops
  // TODO: Make this hook work with multiple conv Ops
  int64_t numConvOps = llvm::count_if(computeOps, [](Operation *op) {
    return isa<linalg::ConvolutionOpInterface>(op);
  });
  assert(numConvOps == 1 && "Exactly 1 Linalg Conv Op is expected");

  // 0.2 ATM only 2D depthwise HWC convs are supported - check that that's the
  // one that's present.
  // TODO: Add support for other convs
  bool seen2DDepthwiseConvHWC = false;
  for (auto op : computeOps) {
    if (isa<linalg::DepthwiseConv2DNhwcHwcOp>(op))
      seen2DDepthwiseConvHWC = true;
  }
  if (!seen2DDepthwiseConvHWC)
    return failure();

  // 1. Get the current lowering config attached to the Conv Op
  FailureOr<IREE::Codegen::LoweringConfigAttr> loweringConfigAttr =
      getLoweringConfig(computeOps);
  if (failed(loweringConfigAttr))
    return failure();

  TilingConfig tc(loweringConfigAttr.value());

  // 2. Calculate new tiling levels
  // FIXME: This assumes that it's the H dim that's "folded". In practice, this
  // will depend on the tiling config.
  unsigned inputHDim = 1;
  unsigned filterHDim = 4;

  auto tilingLevels = loweringConfigAttr.value().getTilingLevels();
  SmallVector<IREE::Codegen::LoweringConfigTilingLevelAttr> newTilingLevelsList;
  for (auto level : tilingLevels) {
    SmallVector<int64_t> newSizes(level.getSizes());
    SmallVector<bool> newScalableFlags(level.getScalableFlags());

    newSizes.erase(newSizes.begin() + inputHDim);
    newSizes.erase(newSizes.begin() + filterHDim);
    if (newScalableFlags.size() > 0) {
      newScalableFlags.erase(newScalableFlags.begin() + inputHDim);
      newScalableFlags.erase(newScalableFlags.begin() + filterHDim);
    }

    SmallVector<int64_t> interchange = {};
    auto newLevel = IREE::Codegen::LoweringConfigTilingLevelAttr::get(
        context, newSizes, ArrayRef<int64_t>{}, newScalableFlags);
    newTilingLevelsList.push_back(newLevel);
  }

  // 3. Create and return new lowering config attribute
  auto newTilingLevels = IREE::Codegen::LoweringConfigTilingLevelsAttr::get(
      context, newTilingLevelsList);
  IREE::Codegen::LoweringConfigAttr newLCA =
      IREE::Codegen::LoweringConfigAttr::get(
          context, newTilingLevels,
          loweringConfigAttr.value().getNativeVectorSize());

  return newLCA;
}

class DecomposeConvolutionToLowerDimOpsPass
    : public DecomposeConvolutionToLowerDimOpsBase<
          DecomposeConvolutionToLowerDimOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, linalg::LinalgDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto funcOp = dyn_cast<func::FuncOp>(getOperation());
    auto computeOps = getComputeOps(funcOp);

    // 1. If there's exactly 1 conv in this function (most common case),
    // compute the "decomposed" version of its lowering config attribute.
    // TODO: Add support for cases with multiple convs per function
    int64_t numConvOps = llvm::count_if(computeOps, [](Operation *op) {
      return isa<linalg::ConvolutionOpInterface>(op);
    });

    if (numConvOps == 0) {
      return;
    }

    FailureOr<IREE::Codegen::LoweringConfigAttr> newLCA;
    if (numConvOps == 1) {
      newLCA = computeNewLCA(computeOps, context);
    }

    // 2. Run the patterns. This is the key part of this pass.
    RewritePatternSet patterns(context);
    linalg::populateDecomposeConvolutionPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }

    // 3. If there's exactly 1 conv in this function (most common case), attach
    // a "decomposed" lowering config to the newly decomposed conv.
    if (numConvOps == 1 && !failed(newLCA)) {
      auto computeOps = getComputeOps(funcOp);
      for (auto computeOp : computeOps) {
        if (isa<linalg::DepthwiseConv1DNwcWcOp>(computeOp))
          setLoweringConfig(computeOp, newLCA.value());
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createDecomposeConvolutionToLowerDimOpsPass() {
  return std::make_unique<DecomposeConvolutionToLowerDimOpsPass>();
}

} // namespace mlir::iree_compiler
