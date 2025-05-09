// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_SPIRVANNOTATEELIGIBLEFORWINOGRADPASS
#include "iree/compiler/Codegen/SPIRV/Passes.h.inc"

namespace {

static const char kWinogradAttr[] = "__winograd_conv";

template <typename ConvOp>
void annotateConvOps(FunctionOpInterface funcOp, MLIRContext *context) {
  Attribute attr = UnitAttr::get(context);
  funcOp.walk([&](ConvOp convOp) { convOp->setAttr(kWinogradAttr, attr); });
}

class SPIRVAnnotateEligibleForWinogradPass final
    : public impl::SPIRVAnnotateEligibleForWinogradPassBase<
          SPIRVAnnotateEligibleForWinogradPass> {
public:
  using impl::SPIRVAnnotateEligibleForWinogradPassBase<
      SPIRVAnnotateEligibleForWinogradPass>::
      SPIRVAnnotateEligibleForWinogradPassBase;

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp);
    if (!target) {
      funcOp.emitError("missing GPU target in #hal.executable.target");
      return signalPassFailure();
    }

    // For now only annotate if cooperative matrices are absent
    if (!target.getWgp().getMma().empty())
      return;

    MLIRContext *context = &getContext();
    annotateConvOps<linalg::Conv2DNhwcHwcfOp>(funcOp, context);
    annotateConvOps<linalg::Conv2DNchwFchwOp>(funcOp, context);
  }
};
} // namespace
} // namespace mlir::iree_compiler
