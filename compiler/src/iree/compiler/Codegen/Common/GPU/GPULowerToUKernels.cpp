// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/UKernelOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPULOWERTOUKERNELSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

/// Matches generic that represent argmax and check if
/// we have the ukernel that matches it shape constraint, and types.
/// If we do, then we convert into iree_codegen.ukernel.argmax operation,
/// that is later lowered into a call to the microkernel.
static FailureOr<IREE::Codegen::UKernelOpInterface>
matchArgmaxDAGForUKernel(RewriterBase &rewriter, linalg::GenericOp op) {
  Value input = op.getDpsInputOperand(0)->get();
  auto inputType = cast<ShapedType>(input.getType());
  Value index = op.getDpsInitOperand(1)->get();
  auto indexType = cast<ShapedType>(index.getType());
  std::string suffix;
  llvm::raw_string_ostream(suffix)
      << inputType.getElementType() << indexType.getElementType();
  auto loweringConfig = getLoweringConfig<IREE::GPU::LoweringConfigAttr>(op);
  if (!loweringConfig) {
    return rewriter.notifyMatchFailure(op, "no lowering_config on this op");
  }
  IREE::GPU::UKernelConfigAttr ukernelAttr =
      IREE::GPU::getUkernelSpec(loweringConfig);
  if (!ukernelAttr) {
    return rewriter.notifyMatchFailure(op, "no ukernel selected for this op");
  }

  Location loc = op.getLoc();
  // Currently only support 1D reduction, where reduc is on fastest dim.
  // Tiling argmax ukernel is also set to enforce this structure.
  const int kReductionDim = op.getNumLoops() - 1;
  Value reductionDimSize =
      rewriter.create<tensor::DimOp>(loc, input, kReductionDim);
  auto genericMicroKernelOp = rewriter.create<IREE::Codegen::UKernelGenericOp>(
      loc, indexType, ukernelAttr.getName(), ValueRange{input}, index,
      ValueRange{reductionDimSize}, ukernelAttr.getDefAttrs(),
      /*strided_outer_dims=*/rewriter.getIndexAttr(0));
  return cast<IREE::Codegen::UKernelOpInterface>(
      genericMicroKernelOp.getOperation());
}

struct LowerArgmaxToUKernelPattern : OpRewritePattern<linalg::GenericOp> {
  LowerArgmaxToUKernelPattern(MLIRContext *context)
      : OpRewritePattern<linalg::GenericOp>(context) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(isArgmaxOp(op))) {
      return failure();
    }
    FailureOr<IREE::Codegen::UKernelOpInterface> ukernelOp =
        matchArgmaxDAGForUKernel(rewriter, op);
    if (failed(ukernelOp)) {
      return rewriter.notifyMatchFailure(
          op, "failed to find microkernel op to replace with");
    }
    rewriter.replaceAllUsesWith(op.getResults()[1],
                                ukernelOp.value()->getResults());
    return success();
  }
};

struct GPULowerToUKernelsPass final
    : impl::GPULowerToUKernelsPassBase<GPULowerToUKernelsPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    // Enabling a lowering of an op to a microkernel is a trade-off between the
    // potential performance advantage of a microkernel over pure code
    // generation for that op, and the potential benefits of fusions. Indeed,
    // once an op lowered into a microkernel, it will never be fused at any MLIR
    // level. Since microkernels are linked as bitcode, they will still undergo
    // LTO-like optimization in their calling contexts, but we shouldn't expect
    // this to achieve similar results as fusing structured ops.

    // These patterns are unconditionally enabled, because we have strong
    // evidence that it is difficult for codegen to consistently approach
    // microkernels performance, and that consideration overrides the benefit of
    // fusions for these ops.
    patterns.insert<LowerArgmaxToUKernelPattern>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
