// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUPADCONVSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

static LogicalResult padToStaticSizes(RewriterBase &rewriter,
                                      TilingInterface tilingInterfaceOp,
                                      SmallVector<OpFoldResult> paddingSizes) {
  SmallVector<Attribute> paddingValues;
  for (Value operand : tilingInterfaceOp.getOperation()->getOperands()) {
    paddingValues.push_back(
        rewriter.getZeroAttr(getElementTypeOrSelf(operand.getType())));
  }

  auto options = linalg::PadTilingInterfaceOptions()
                     .setPaddingSizes(paddingSizes)
                     .setPaddingValues(paddingValues)
                     .setPadToMultipleOf(true);

  SmallVector<tensor::PadOp> padOps;
  FailureOr<TilingInterface> maybePaddedOp =
      linalg::rewriteAsPaddedOp(rewriter, tilingInterfaceOp, options, padOps);
  if (failed(maybePaddedOp)) {
    return tilingInterfaceOp->emitOpError("failed to pad op");
  }

  return success();
}

struct GPUPadConvsPass final : impl::GPUPadConvsPassBase<GPUPadConvsPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    IRRewriter rewriter(funcOp);
    funcOp.walk([&](TilingInterface op) {
      auto linalgOp = dyn_cast<linalg::LinalgOp>(op.getOperation());
      if (!linalgOp || !linalg::isaConvolutionOpInterface(linalgOp)) {
        return;
      }

      auto loweringConfig =
          getLoweringConfig<IREE::GPU::LoweringConfigAttr>(op);
      if (!loweringConfig) {
        return;
      }

      // Get padding sizes from lowering_config.
      std::optional<SmallVector<int64_t>> paddingSizes =
          getPaddingList(loweringConfig, /*padConv*/ true);
      if (!paddingSizes) {
        return;
      }

      SmallVector<OpFoldResult> padSizes =
          getAsIndexOpFoldResult(rewriter.getContext(), paddingSizes.value());
      rewriter.setInsertionPoint(op);
      if (failed(padToStaticSizes(rewriter, op, padSizes))) {
        return signalPassFailure();
      }
    });
  }
};

} // namespace
} // namespace mlir::iree_compiler
