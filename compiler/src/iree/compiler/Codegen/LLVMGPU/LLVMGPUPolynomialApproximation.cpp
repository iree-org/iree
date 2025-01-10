// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUPOLYNOMIALAPPROXIMATIONPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

struct LLVMGPUPolynomialApproximationPass final
    : impl::LLVMGPUPolynomialApproximationPassBase<
          LLVMGPUPolynomialApproximationPass> {
  void runOnOperation() override {
    RewritePatternSet mathPatterns(&getContext());
    // TODO(lialan): Handle these functions efficiently in ROCDL/NVVM
    // conversion passes. This expansion is likely suboptimal.
    populateExpandPowFPattern(mathPatterns);
    populateExpandFPowIPattern(mathPatterns);

    walkAndApplyPatterns(getOperation(), std::move(mathPatterns));
  }
};

} // namespace

} // namespace mlir::iree_compiler
