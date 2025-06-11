// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-rematerialize-parallel-ops"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_REMATERIALIZEPARALLELOPSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

static bool isScalarOrTensorOfSizeOne(Type t) {
  if (auto tensorType = dyn_cast<RankedTensorType>(t)) {
    return tensorType.hasStaticShape() && tensorType.getNumElements() == 1;
  }
  return t.isIntOrIndexOrFloat();
}

///  This function checks whether the `genericOp` has any external captures,
///  i.e., whether it uses any values that are defined outside of its body.
///  %10 = linalg.generic {indexing_maps = [#map, #map],
///          iterator_types = ["parallel", "parallel"]}
///         ins(%5 : tensor<4096x64xi64>) outs(%9 : tensor<4096x64xf16>) {
///          ^bb0(%in: i64, %out: f16):
///            %14 = linalg.index 0 : index
///            %15 = arith.index_cast %in : i64 to index
///            %extracted = tensor.extract %4[%14, %15] : tensor<4096x64xf16>
///            linalg.yield %extracted : f16
///           } -> tensor<4096x64xf16>
///  Here %4 is an external capture used via tensor.extract inside
///  linalg.generic hence the above `genericOp` has an external capture.
static bool hasExternalCapture(linalg::GenericOp genericOp) {
  Block &body = genericOp.getRegion().front();
  for (Operation &op : body.getOperations()) {
    for (Value operand : op.getOperands()) {
      if (auto bArg = dyn_cast<BlockArgument>(operand)) {
        // Check whether the operand lies in the same block.
        if (bArg.getOwner() == &body) {
          continue;
        }
        return true;
      }
      Operation *defOp = operand.getDefiningOp();
      // Scalar constant is allowed.
      if (defOp && defOp->hasTrait<mlir::OpTrait::ConstantLike>()) {
        Type type = operand.getType();
        if (type.isIntOrFloat() || type.isIndex()) {
          continue;
        }
      }
      // If defining op is not inside the block, it’s an external value.
      if (!defOp || defOp->getBlock() != &body) {
        return true;
      }
    }
  }
  return false; // All operands are locally defined or block arguments.
}

/// Rematerialize all parallel elementwise operations into its users within a
/// `flow.dispatch.region`.
struct RematerializeParallelOpsPattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    // Avoid doing this for scalar operations.
    auto isScalarValue = [](Value v) {
      return isScalarOrTensorOfSizeOne(v.getType());
    };
    if (llvm::all_of(genericOp.getOperands(), isScalarValue) &&
        llvm::all_of(genericOp.getResults(), isScalarValue)) {
      return failure();
    }

    // Find the first operand that is defined by another generic op on tensors.
    for (OpOperand &opOperand : genericOp->getOpOperands()) {
      if (!linalg::areElementwiseOpsFusable(&opOperand)) {
        continue;
      }
      auto producer = opOperand.get().getDefiningOp<linalg::GenericOp>();
      if (producer && hasExternalCapture(producer)) {
        continue;
      }
      FailureOr<linalg::ElementwiseOpFusionResult> fusionResult =
          linalg::fuseElementwiseOps(rewriter, &opOperand);
      if (succeeded(fusionResult)) {
        auto replacements = fusionResult->fusedOp->getResults().take_back(
            genericOp.getNumResults());
        // Copy over any non native attributes for the operation.
        auto prunedAttributeList = linalg::getPrunedAttributeList(genericOp);
        fusionResult->fusedOp->setAttrs(prunedAttributeList);
        rewriter.replaceOp(genericOp, replacements);
        return success();
      }
    }
    return failure();
  }
};

struct RematerializeParallelOpsPass final
    : impl::RematerializeParallelOpsPassBase<RematerializeParallelOpsPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    RewritePatternSet fusionPatterns(funcOp.getContext());
    fusionPatterns.insert<RematerializeParallelOpsPattern>(funcOp.getContext());
    linalg::populateEraseUnusedOperandsAndResultsPatterns(fusionPatterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(fusionPatterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
