// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmgpu-vector-multi-reduction-lowering"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUVECTORMULTIREDUCTIONLOWERINGPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

/// Converts 1D vector.multi_reduction directly to vector.reduction.
///
/// Example:
/// ```mlir
/// // Before
/// %r = vector.multi_reduction <add>, %v, %acc [0] : vector<Nxf32> to f32
///
/// // After
/// %r = vector.reduction <add>, %v, %acc : vector<Nxf32> into f32
/// ```
struct OneDimMultiReductionToReduction
    : public vector::MaskableOpRewritePattern<vector::MultiDimReductionOp> {
  using MaskableOpRewritePattern::MaskableOpRewritePattern;

  FailureOr<Value>
  matchAndRewriteMaskableOp(vector::MultiDimReductionOp multiReductionOp,
                            vector::MaskingOpInterface maskingOp,
                            PatternRewriter &rewriter) const override {
    auto srcRank = multiReductionOp.getSourceVectorType().getRank();
    if (srcRank != 1) {
      return failure();
    }

    if (!multiReductionOp.isReducedDim(0)) {
      return failure();
    }

    auto loc = multiReductionOp.getLoc();
    Value mask = maskingOp ? maskingOp.getMask() : Value();

    Operation *reductionOp = vector::ReductionOp::create(
        rewriter, loc, multiReductionOp.getKind(), multiReductionOp.getSource(),
        multiReductionOp.getAcc());

    if (mask) {
      reductionOp = mlir::vector::maskOperation(rewriter, reductionOp, mask);
    }

    return reductionOp->getResult(0);
  }
};

} // namespace

struct LLVMGPUVectorMultiReductionLoweringPass final
    : impl::LLVMGPUVectorMultiReductionLoweringPassBase<
          LLVMGPUVectorMultiReductionLoweringPass> {

  void runOnOperation() override {
    mlir::FunctionOpInterface funcOp = getOperation();
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<OneDimMultiReductionToReduction>(ctx);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace mlir::iree_compiler
