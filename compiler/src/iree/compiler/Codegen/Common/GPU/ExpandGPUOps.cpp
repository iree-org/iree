// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
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

struct ExpandGPUOpsPass final : impl::ExpandGPUOpsPassBase<ExpandGPUOpsPass> {
  // Apply AMD GPU targeting patterns.
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    std::optional<int> subgroupSize = getGPUSubgroupSize(funcOp);
    if (!subgroupSize) {
      funcOp->emitOpError("missing subgroup size");
      return signalPassFailure();
    }

    RewritePatternSet patterns(ctx);
    auto execTarget = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
    IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp);
    StringRef targetArch = target.getArch();
    auto maybeChipset = amdgpu::Chipset::parse(targetArch);
    if (succeeded(maybeChipset) && isROCMBackend(execTarget)) {
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

} // namespace mlir::iree_compiler
