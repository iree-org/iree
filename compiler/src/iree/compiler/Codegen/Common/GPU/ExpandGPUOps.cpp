// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
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
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    std::optional<int> subgroupSize = getGPUSubgroupSize(funcOp);
    if (!subgroupSize) {
      funcOp->emitOpError("missing subgroup size");
      return signalPassFailure();
    }

    RewritePatternSet patterns(ctx);
    populateGpuBreakDownSubgroupReducePatterns(
        patterns, /* maxShuffleBitwidth=*/32, PatternBenefit(2));
    populateGpuLowerClusteredSubgroupReduceToShufflePatterns(
        patterns, *subgroupSize, /* shuffleBitwidth=*/32, PatternBenefit(1));
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  };
};

} // namespace

} // namespace mlir::iree_compiler
