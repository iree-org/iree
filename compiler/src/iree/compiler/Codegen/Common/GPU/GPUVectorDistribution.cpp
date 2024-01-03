// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/Common/GPU/VectorLayoutProvider.h"
#include "iree/compiler/Codegen/Common/VectorLayoutAnalysis.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-vector-distribution"

using namespace mlir::iree_compiler::IREE::VectorExt;

namespace mlir::iree_compiler {

TypedValue<VectorType> getDistributed(RewriterBase &rewriter,
                                      TypedValue<VectorType> value,
                                      LayoutProvider &provider) {
  // If this is a result of a "to_simd" op, use the source value of it.
  if (auto toSIMD = value.getDefiningOp<IREE::VectorExt::ToSIMDOp>()) {
    value = cast<TypedValue<VectorType>>(toSIMD.getInput());
    return value;
  }
  // Create a "to_simt" op to convert the value to the distributed layout.
  SmallVector<int64_t> distributedShape = provider.getDistributedShape(value);
  VectorType distributedType =
      VectorType::get(distributedShape, value.getType().getElementType());
  auto toSIMT = rewriter.create<IREE::VectorExt::ToSIMTOp>(
      value.getLoc(), distributedType, value);
  return toSIMT.getResult();
}

void replaceOpWithDistributedValues(RewriterBase &rewriter, Operation *op,
                                    LayoutProvider &provider,
                                    ValueRange values) {
  // Replace all OpResults with the given values.
  SmallVector<Value> replacements;
  for (OpResult opResult : op->getOpResults()) {
    Value replacement = values[opResult.getResultNumber()];
    // If this value is a vector type, it must be converted back to simd.
    if (isa<VectorType>(replacement.getType())) {
      auto oldResult = cast<TypedValue<VectorType>>(opResult);
      // Create a toSIMD op to convert the value back to the simd.
      rewriter.setInsertionPointAfterValue(oldResult);
      auto toSIMD = rewriter.create<IREE::VectorExt::ToSIMDOp>(
          oldResult.getLoc(), oldResult.getType(), replacement);
      // Clone the layout to the new value.
      provider.getAnalysis().cloneLayoutInformationToNewValue(
          oldResult, toSIMD.getResult());
      // Add to replacements.
      replacement = toSIMD.getResult();
    }
    replacements.push_back(replacement);
  }

  rewriter.replaceOp(op, replacements);
}

class DistributeConstants : public OpRewritePattern<arith::ConstantOp> {
public:
  DistributeConstants(MLIRContext *context, VectorLayoutAnalysis &analysis,
                      LayoutProvider &provider, PatternBenefit benefit = 1)
      : OpRewritePattern<arith::ConstantOp>(context, benefit),
        analysis(analysis), provider(provider) {}

  LogicalResult matchAndRewrite(arith::ConstantOp constantOp,
                                PatternRewriter &rewriter) const override {
    Value constantResult = constantOp.getResult();
    if (!isa<VectorType>(constantResult.getType()))
      return failure();
    auto constant = cast<TypedValue<VectorType>>(constantResult);
    auto attr = llvm::cast<DenseElementsAttr>(constantOp.getValue());

    // Only handle splat values for now.
    if (!attr.isSplat())
      return failure();

    // Replace the original op with the distributed op.
    Type elementType = constant.getType().getElementType();
    auto vectorType =
        VectorType::get(provider.getDistributedShape(constant), elementType);
    replaceOpWithNewDistributedOp<arith::ConstantOp>(
        provider, rewriter, constantOp, vectorType,
        DenseElementsAttr::get(vectorType, attr.getSplatValue<APFloat>()));
    return success();
  }

private:
  VectorLayoutAnalysis &analysis;
  LayoutProvider &provider;
};

class DistributeElementwise
    : public OpTraitRewritePattern<OpTrait::Elementwise> {
public:
  DistributeElementwise(MLIRContext *context, VectorLayoutAnalysis &analysis,
                        LayoutProvider &provider, PatternBenefit benefit = 1)
      : OpTraitRewritePattern<OpTrait::Elementwise>(context, benefit),
        analysis(analysis), provider(provider) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!OpTrait::hasElementwiseMappableTraits(op))
      return failure();

    // Get the distributed operands.
    SmallVector<Value> operands;
    for (Value operand : op->getOperands()) {
      if (auto vectorOperand = dyn_cast<TypedValue<VectorType>>(operand)) {
        operand = getDistributed(rewriter, vectorOperand, provider);
      }
      operands.push_back(operand);
    }

    // Get the new distributed vector types for the operation.
    SmallVector<Type> resultTypes;
    for (Value result : op->getResults()) {
      if (auto vectorResult = dyn_cast<TypedValue<VectorType>>(result)) {
        // Distribute vector result types.
        auto newType =
            VectorType::get(provider.getDistributedShape(vectorResult),
                            vectorResult.getType().getElementType());
        resultTypes.push_back(newType);
      } else {
        resultTypes.push_back(result.getType());
      }
    }

    // Replace the original op with the distributed op.
    Operation *distributedOp =
        rewriter.create(op->getLoc(), op->getName().getIdentifier(), operands,
                        resultTypes, op->getAttrs());
    replaceOpWithDistributedValues(rewriter, op, provider,
                                   distributedOp->getResults());
    return success();
  }

private:
  VectorLayoutAnalysis &analysis;
  LayoutProvider &provider;
};

VectorDistribution::VectorDistribution(func::FuncOp root,
                                       VectorLayoutAnalysis &analysis,
                                       LayoutProvider &provider)
    : root(root), analysis(analysis), provider(provider) {
  provider.setAnchorOps();
  if (failed(analysis.run()))
    return;
  LLVM_DEBUG(llvm::dbgs() << "Layout Analysis Completed Successfully :\n");
  LLVM_DEBUG(analysis.print(llvm::dbgs()));
}

/// A rewriter for the pattern rewriting driver.
class VectorDistributionRewriter : public PatternRewriter,
                                   RewriterBase::Listener {
public:
  VectorDistributionRewriter(MLIRContext *ctx,
                             DenseSet<Operation *> &opsToErase,
                             SmallVector<Operation *> &worklist)
      : PatternRewriter(context, this), opsToErase(opsToErase),
        worklist(worklist) {}

  void notifyOperationRemoved(Operation *op) override { opsToErase.insert(op); }

private:
  // Reference to operations to be erased and the worklist of the driver.
  DenseSet<Operation *> &opsToErase;
  SmallVector<Operation *> &worklist;
};

static bool canDistribute(Operation *op, VectorLayoutAnalysis &analysis) {
  bool needsDistribution = false;
  // Check if this operation has any operands with a vector type. If so,
  // then they need to have a layout.
  for (Value operand : op->getOperands()) {
    if (isa<VectorType>(operand.getType())) {
      needsDistribution = true;
      if (!analysis.getLayout<Attribute>(operand)) {
        return false;
      }
    }
  }

  // Check if this operation has any results with a vector type. If so,
  // then they need to have a layout.
  for (OpResult result : op->getResults()) {
    if (isa<VectorType>(result.getType())) {
      needsDistribution = true;
      if (!analysis.getLayout<Attribute>(result)) {
        return false;
      }
    }
  }

  return needsDistribution;
}

static LogicalResult applyVectorDistributionDriver(
    Operation *root, const FrozenRewritePatternSet &patterns,
    VectorLayoutAnalysis &analysis, LayoutProvider &provider) {

  DenseSet<Operation *> opsToErase;
  SmallVector<Operation *> worklist;

  VectorDistributionRewriter rewriter(root->getContext(), opsToErase, worklist);
  PatternApplicator applicator(patterns);
  applicator.applyDefaultCostModel();

  // Collect all the operations to be distributed.
  root->walk([&](Operation *op) {
    if (canDistribute(op, analysis))
      worklist.push_back(op);
  });

  for (Operation *op : worklist) {
    if (opsToErase.contains(op))
      continue;

    if (failed(applicator.matchAndRewrite(op, rewriter))) {
      return failure();
    }
  }

  return success();
}

LogicalResult VectorDistribution::distribute() {
  RewritePatternSet patterns(root.getContext());
  patterns.add<DistributeConstants, DistributeElementwise>(root.getContext(),
                                                           analysis, provider);
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  return applyVectorDistributionDriver(root, frozenPatterns, analysis,
                                       provider);
}

} // namespace mlir::iree_compiler
