// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmgpu-vector-flattening"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUVECTORFLATTENINGPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

struct LLVMGPUVectorFlatteningPass final
    : impl::LLVMGPUVectorFlatteningPassBase<LLVMGPUVectorFlatteningPass> {

  void runOnOperation() override {
    mlir::FunctionOpInterface funcOp = getOperation();
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    vector::populateVectorMultiReductionFlatteningPatterns(
        patterns, vector::VectorMultiReductionLowering::InnerReduction);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace mlir::iree_compiler
