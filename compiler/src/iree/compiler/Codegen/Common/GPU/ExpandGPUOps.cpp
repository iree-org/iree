// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-expand-gpu-ops"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_EXPANDGPUOPSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

class ExpandGPUOpsPass final : public impl::ExpandGPUOpsPassBase<ExpandGPUOpsPass> {
private:
  // Apply AMD GPU targetting patterns
  bool forROCDL = false;

public:
  using impl::ExpandGPUOpsPassBase<ExpandGPUOpsPass>::ExpandGPUOpsPassBase;
  ExpandGPUOpsPass(bool forROCDL) : forROCDL(forROCDL) {}

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    std::optional<int> subgroupSize = getGPUSubgroupSize(funcOp);
    if (!subgroupSize) {
      funcOp->emitOpError("missing subgroup size");
      return signalPassFailure();
    }

    RewritePatternSet patterns(ctx);
    IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp);
    StringRef targetArch = target.getArch();
    auto maybeChipset = amdgpu::Chipset::parse(targetArch);
    if (succeeded(maybeChipset) && forROCDL) {
      populateGpuLowerSubgroupReduceToDPPPatterns(
          patterns, *subgroupSize, *maybeChipset, PatternBenefit(2));
      populateGpuLowerClusteredSubgroupReduceToDPPPatterns(
          patterns, *subgroupSize, *maybeChipset, PatternBenefit(2));
    }

    populateGpuBreakDownSubgroupReducePatterns(
        patterns, /* maxShuffleBitwidth=*/32, PatternBenefit(3));
    populateGpuLowerClusteredSubgroupReduceToShufflePatterns(
        patterns, *subgroupSize, /* shuffleBitwidth=*/32, PatternBenefit(1));
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  };
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createExpandGPUOpsPass(bool forROCDL) {
  return std::make_unique<ExpandGPUOpsPass>(forROCDL);
}

} // namespace mlir::iree_compiler
