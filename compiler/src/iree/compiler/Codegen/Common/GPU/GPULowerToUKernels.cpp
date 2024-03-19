// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/PassDetail.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/UKernelOps.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {
class GPULowerToUKernelsPass
    : public GPULowerToUKernelsBase<GPULowerToUKernelsPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Codegen::IREECodegenDialect>();
  }

  void runOnOperation() override;
};
} // namespace

/// Holds a function name and attributes.
struct FnNameAndDefAttrs {
  std::string name;
  SmallVector<NamedAttribute> defAttrs;
};

/// Returns the function name and attributes to use for a ukernel with given
/// `ukernelName` on the target described by `targetAttr`.
static FnNameAndDefAttrs
getFnNameAndDefAttrs(const char *ukernelName, std::string &typeSuffixID,
                     RewriterBase &rewriter,
                     IREE::HAL::ExecutableTargetAttr targetAttr) {
  FnNameAndDefAttrs result;
  if (isROCMBackend(targetAttr)) {
    result.name =
        std::string("__iree_uk_rocm_") + ukernelName + "_" + typeSuffixID;
    result.defAttrs.emplace_back(rewriter.getStringAttr("vm.import.module"),
                                 rewriter.getStringAttr("rocm"));
  }
  return result;
}

/// Matches generic that represent argmax and check if
/// we have the ukernel that matches it shape constraint, and types.
/// If we do, then we convert into iree_codegen.ukernel.argmax operation,
/// that is later lowered into a call to the microkernel.
static FailureOr<IREE::Codegen::UKernelOpInterface>
matchArgmaxDAGForUKernel(RewriterBase &rewriter, linalg::GenericOp op) {
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
  const char ukernelName[] = "argmax";
  if (!hasUkernel(targetAttr, ukernelName) ||
      !hasUkernelSupportedGpuArch(targetAttr)) {
    return failure();
  }

  // Currently only support argmax where parallel dims are 1.
  // Tiling pipeline is also set to tile all parallel dims to 1, and
  // reduction dim to be size of whole reduction problem. Which allow
  // this constraint to be true for a lot of argmax variances.
  // TODO: Support multi-row or grid-strided argmax ukernel.
  SmallVector<int64_t, 4> bounds = op.getStaticLoopRanges();
  SmallVector<unsigned> parallelDims;
  op.getParallelDims(parallelDims);
  int64_t parallelSize = 1;
  for (int64_t dim : parallelDims) {
    if (ShapedType::isDynamic(bounds[dim])) {
      return failure();
    }
    parallelSize *= bounds[dim];
  }
  if (parallelSize != 1)
    return failure();

  // Get value/input type.
  Value input = op.getDpsInputOperand(0)->get();
  auto inputType = llvm::cast<ShapedType>(input.getType());
  Type inputElemType = inputType.getElementType();
  // Only support f16 and f32 values.
  if (!inputElemType.isF16() && !inputElemType.isF32()) {
    return failure();
  }

  // Get index type.
  Value index = op.getDpsInitOperand(1)->get();
  auto indexType = llvm::cast<ShapedType>(index.getType());
  Type indexElemType = indexType.getElementType();
  // Only support i32 and i64 index.
  if (!indexElemType.isInteger(32) && !indexElemType.isInteger(64)) {
    return failure();
  }

  std::string typeSuffixID = "";
  if (inputElemType.isF16() && indexElemType.isInteger(32)) {
    typeSuffixID = "F16I32";
  } else if (inputElemType.isF16() && indexElemType.isInteger(64)) {
    typeSuffixID = "F16I64";
  } else if (inputElemType.isF32() && indexElemType.isInteger(32)) {
    typeSuffixID = "F32I32";
  } else if (inputElemType.isF32() && indexElemType.isInteger(64)) {
    typeSuffixID = "F32I64";
  } else {
    return rewriter.notifyMatchFailure(
        op, "unsupported combination of element types");
  }

  Location loc = op.getLoc();
  // Currently only support 1D reduction, where reduc is on fastest dim.
  // Tiling argmax ukernel is also set to enforce this structure.
  const int kReductionDim = op.getNumLoops() - 1;
  Value reductionDimSize =
      rewriter.create<tensor::DimOp>(loc, input, kReductionDim);
  auto fn =
      getFnNameAndDefAttrs(ukernelName, typeSuffixID, rewriter, targetAttr);
  auto genericMicroKernelOp = rewriter.create<IREE::Codegen::UKernelGenericOp>(
      loc, indexType, fn.name, ValueRange{input}, index,
      ValueRange{reductionDimSize},
      /*fn_def_attrs=*/rewriter.getDictionaryAttr(fn.defAttrs),
      /*strided_outer_dims=*/rewriter.getIndexAttr(0));
  return cast<IREE::Codegen::UKernelOpInterface>(
      genericMicroKernelOp.getOperation());
}

namespace {

using TargetPredicate = std::function<bool(IREE::HAL::ExecutableTargetAttr)>;

struct LowerArgmaxToUKernelPattern : OpRewritePattern<linalg::GenericOp> {
  LowerArgmaxToUKernelPattern(MLIRContext *context,
                              TargetPredicate targetPredicate)
      : OpRewritePattern<linalg::GenericOp>(context),
        targetPredicate(targetPredicate) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (targetPredicate &&
        !targetPredicate(IREE::HAL::ExecutableTargetAttr::lookup(op))) {
      return failure();
    }
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

  TargetPredicate targetPredicate;
};

} // namespace

void GPULowerToUKernelsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  // Enabling a lowering of an op to a microkernel is a trade-off between the
  // potential performance advantage of a microkernel over pure code generation
  // for that op, and the potential benefits of fusions. Indeed, once an op
  // lowered into a microkernel, it will never be fused at any MLIR level.
  // Since microkernels are linked as bitcode, they will still undergo LTO-like
  // optimization in their calling contexts, but we shouldn't expect this to
  // achieve similar results as fusing structured ops.

  // These patterns are unconditionally enabled, because we have strong evidence
  // that it is difficult for codegen to consistently approach microkernels
  // performance, and that consideration overrides the benefit of fusions for
  // these ops.
  patterns.insert<LowerArgmaxToUKernelPattern>(context, isROCMBackend);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<>> createGPULowerToUKernelsPass() {
  return std::make_unique<GPULowerToUKernelsPass>();
}

} // namespace iree_compiler
} // namespace mlir
