// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_DECOMPOSECONVOLUTIONTOLOWERDIMOPSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

static bool foldHDim(linalg::DepthwiseConv2DNhwcHwcOp convOp) {
  Value kernel = convOp.getInputs().back();
  Value output = convOp.getOutputs().front();

  auto kernelType = dyn_cast<RankedTensorType>(kernel.getType());
  auto outputType = dyn_cast<RankedTensorType>(output.getType());

  auto kernelShape = kernelType.getShape();
  auto outputShape = outputType.getShape();

  int64_t khSize = kernelShape[0];
  int64_t ohSize = outputShape[1];
  bool removeH = (khSize == 1 && ohSize == 1);

  return removeH;
}

/// Computes a "decomposed" lowering config attribute for a conv OP
///
/// This method complements the patterns to decompose 2D convolutions into 1D
/// convs. Specifically, it will update the lowering config attached to a Conv
/// Op in a way that matches the "decomposition" patterns.
///
/// At the moment only Depthwise HWC convolutions are supported.
static FailureOr<IREE::Codegen::LoweringConfigAttr>
computeDecomposedLoweringConfig(ArrayRef<Operation *> computeOps,
                                MLIRContext *context) {

  // 0.1 Double-check that there's only one convolution Op.
  // TODO: Make this hook work with multiple conv Ops
  assert(llvm::count_if(computeOps,
                        [](Operation *op) {
                          return isa<linalg::ConvolutionOpInterface>(op);
                        }) == 1 &&
         "Exactly 1 Linalg Conv Op is expected");

  // 1. Get the conv Op to update
  // ATM only 2D depthwise HWC convs are supported.
  // TODO: Add support for other convs
  linalg::DepthwiseConv2DNhwcHwcOp convOp;
  for (auto op : computeOps) {
    if (isa<linalg::DepthwiseConv2DNhwcHwcOp>(op)) {
      convOp = cast<linalg::DepthwiseConv2DNhwcHwcOp>(op);
      break;
    }
  }

  if (!convOp) {
    return failure();
  }

  // ATM only folding of the H dim is supported.
  // TODO: Add support for cases where the W dim is folded.
  if (!foldHDim(convOp))
    return failure();

  // 2. Get the current lowering config attached to the Conv Op.
  FailureOr<IREE::Codegen::LoweringConfigAttr> loweringConfigAttr =
      getFirstLoweringConfig<IREE::Codegen::LoweringConfigAttr>(computeOps);
  if (failed(loweringConfigAttr))
    return failure();

  // TODO: Either remove "interchange" from lowering_config or add support in
  // this pass.
  if (!loweringConfigAttr->isInterchangeEmpty())
    return failure();

  // 3. Calculate new tiling levels.
  // Note that this will basically erase the _H_ dims from the orignal lowering
  // config.
  auto dims = linalg::inferConvolutionDims(convOp);
  SmallVector<unsigned> hDimsToErase = {dims->outputImage[0],
                                        dims->filterLoop[0]};
  llvm::sort(hDimsToErase, [](auto x, auto y) { return x > y; });

  SmallVector<IREE::Codegen::LoweringConfigTilingLevelAttr> newTilingLevelsList;
  for (auto level : loweringConfigAttr.value().getTilingLevels()) {
    SmallVector<int64_t> newSizes(level.getSizes());
    SmallVector<bool> newScalableFlags(level.getScalableFlags());

    llvm::for_each(hDimsToErase, [&newSizes](unsigned idx) {
      newSizes.erase(newSizes.begin() + idx);
    });
    if (newScalableFlags.size() > 0) {
      llvm::for_each(hDimsToErase, [&newScalableFlags](unsigned idx) {
        newScalableFlags.erase(newScalableFlags.begin() + idx);
      });
    }

    auto newLevel = IREE::Codegen::LoweringConfigTilingLevelAttr::get(
        context, newSizes, /*interchange=*/{}, newScalableFlags);
    newTilingLevelsList.push_back(newLevel);
  }

  // 4. Create and return a new lowering config attribute.
  auto newTilingLevels = IREE::Codegen::LoweringConfigTilingLevelsAttr::get(
      context, newTilingLevelsList);
  return IREE::Codegen::LoweringConfigAttr::get(
      context, newTilingLevels,
      loweringConfigAttr.value().getNativeVectorSize());
}

class DecomposeConvolutionToLowerDimOpsPass
    : public impl::DecomposeConvolutionToLowerDimOpsPassBase<
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

    FailureOr<IREE::Codegen::LoweringConfigAttr> newLoweringConfig;
    if (numConvOps == 1) {
      newLoweringConfig = computeDecomposedLoweringConfig(computeOps, context);
    }

    // 2. Run the patterns. This is the key part of this pass.
    RewritePatternSet patterns(context);
    linalg::populateDecomposeConvolutionPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }

    // 3. If there's exactly 1 conv in this function (most common case), attach
    // a "decomposed" lowering config created earlier to the newly decomposed
    // conv Op.
    if (numConvOps == 1 && succeeded(newLoweringConfig)) {
      auto computeOps = getComputeOps(funcOp);
      for (auto computeOp : computeOps) {
        if (isa<linalg::DepthwiseConv1DNwcWcOp>(computeOp))
          setLoweringConfig(computeOp, newLoweringConfig.value());
      }
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
