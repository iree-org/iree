// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree/compiler/Codegen/Common/VectorLayoutAnalysis.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Rewrite/PatternApplicator.h"

namespace mlir::iree_compiler {

using namespace mlir::iree_compiler::IREE::VectorExt;
using VectorValue = TypedValue<VectorType>;

namespace {

struct DistributeConstants final : OpDistributionPattern<arith::ConstantOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(arith::ConstantOp constantOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    Value constantResult = constantOp.getResult();
    if (!isa<VectorType>(constantResult.getType()))
      return failure();
    auto constant = cast<VectorValue>(constantResult);

    // Only handle splat values for now.
    auto attr = dyn_cast<SplatElementsAttr>(constantOp.getValue());
    if (!attr)
      return failure();

    VectorLayoutInterface layout = signature.results[0];

    // Replace the original op with the distributed op.
    Type elementType = constant.getType().getElementType();
    auto vectorType = VectorType::get(
        layout.getDistributedShape(constant.getType()), elementType);
    Operation *distirbutedOp = rewriter.create<arith::ConstantOp>(
        constantOp.getLoc(), vectorType,
        SplatElementsAttr::get(vectorType, attr.getSplatValue<Attribute>()));
    replaceOpWithDistributedValues(rewriter, constantOp,
                                   distirbutedOp->getResult(0));
    return success();
  }
};

template <typename OpTy>
struct DistributeElementwise final : OpDistributionPattern<OpTy> {
  using OpDistributionPattern<OpTy>::OpDistributionPattern;

  LogicalResult matchAndRewrite(OpTy op, DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    // Get the distributed operands.
    SmallVector<Value> operands;
    for (auto [operand, opLayout] :
         llvm::zip(op->getOperands(), signature.operands)) {
      if (auto vectorOperand = dyn_cast<VectorValue>(operand)) {
        operand = DistributionPattern::getDistributed(rewriter, vectorOperand,
                                                      opLayout);
      }
      operands.push_back(operand);
    }

    // Get the new distributed vector types for the operation.
    SmallVector<Type> resultTypes;
    for (auto [result, resLayout] :
         llvm::zip(op->getResults(), signature.results)) {
      Type resultType = result.getType();

      // Distribute vector result types.
      if (auto vectorResult = dyn_cast<VectorValue>(result)) {
        resultType = VectorType::get(
            resLayout.getDistributedShape(vectorResult.getType()),
            vectorResult.getType().getElementType());
      }
      resultTypes.push_back(resultType);
    }

    // Replace the original op with the distributed op.
    Operation *distributedOp = rewriter.create(
        op->getLoc(), op->getName().getIdentifier(), operands, resultTypes);
    DistributionPattern::replaceOpWithDistributedValues(
        rewriter, op, distributedOp->getResults());
    return success();
  }
};

}; // namespace

void populateGPUDistributionPatterns(RewritePatternSet &patterns) {
  patterns.add<DistributeConstants, DistributeElementwise<arith::MulIOp>,
               DistributeElementwise<arith::MulFOp>,
               DistributeElementwise<arith::AddIOp>,
               DistributeElementwise<arith::AddFOp>>(patterns.getContext());
}

}; // namespace mlir::iree_compiler
