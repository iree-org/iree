// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
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

using VectorValue = TypedValue<VectorType>;

VectorValue getDistributed(RewriterBase &rewriter, VectorValue value,
                           VectorLayoutOptions &options) {
  // If this is a result of a "to_simd" op, use the source value of it.
  if (auto toSIMD = value.getDefiningOp<IREE::VectorExt::ToSIMDOp>()) {
    return cast<VectorValue>(toSIMD.getInput());
  }
  // Create a "to_simt" op to convert the value to the distributed layout.
  SmallVector<int64_t> distributedShape = options.getDistributedShape(value);
  VectorType distributedType =
      VectorType::get(distributedShape, value.getType().getElementType());
  auto toSIMT = rewriter.create<IREE::VectorExt::ToSIMTOp>(
      value.getLoc(), distributedType, value);
  return toSIMT.getResult();
}

void replaceOpWithDistributedValues(RewriterBase &rewriter, Operation *op,
                                    VectorLayoutOptions &options,
                                    ValueRange values) {
  // Replace all OpResults with the given values.
  SmallVector<Value> replacements;
  for (auto [opResult, replacement] :
       llvm::zip_equal(op->getOpResults(), values)) {
    // If this value is a vector type, it must be converted back to simd.
    if (isa<VectorType>(replacement.getType())) {
      auto oldResult = cast<VectorValue>(opResult);
      // Create a toSIMD op to convert the value back to the simd.
      rewriter.setInsertionPointAfterValue(oldResult);
      Value toSIMD = rewriter.create<IREE::VectorExt::ToSIMDOp>(
          oldResult.getLoc(), oldResult.getType(), replacement);
      // Clone the layout to the new value.
      options.getAnalysis().cloneLayoutInformationToNewValue(oldResult, toSIMD);
      // Add to replacements.
      replacement = toSIMD;
    }
    replacements.push_back(replacement);
  }

  rewriter.replaceOp(op, replacements);
}

class DistributeConstants : public OpRewritePattern<arith::ConstantOp> {
public:
  DistributeConstants(MLIRContext *context, VectorLayoutAnalysis &analysis,
                      VectorLayoutOptions &options, PatternBenefit benefit = 1)
      : OpRewritePattern<arith::ConstantOp>(context, benefit),
        analysis(analysis), options(options) {}

  LogicalResult matchAndRewrite(arith::ConstantOp constantOp,
                                PatternRewriter &rewriter) const override {
    Value constantResult = constantOp.getResult();
    if (!isa<VectorType>(constantResult.getType()))
      return failure();
    auto constant = cast<VectorValue>(constantResult);

    // Only handle splat values for now.
    auto attr = dyn_cast<DenseElementsAttr>(constantOp.getValue());
    if (!attr)
      return failure();
    if (!attr.isSplat())
      return failure();

    // Replace the original op with the distributed op.
    Type elementType = constant.getType().getElementType();
    auto vectorType =
        VectorType::get(options.getDistributedShape(constant), elementType);
    replaceOpWithNewDistributedOp<arith::ConstantOp>(
        options, rewriter, constantOp, vectorType,
        SplatElementsAttr::get(vectorType, attr.getSplatValue<Attribute>()));
    return success();
  }

private:
  VectorLayoutAnalysis &analysis;
  VectorLayoutOptions &options;
};

class DistributeElementwise
    : public OpTraitRewritePattern<OpTrait::Elementwise> {
public:
  DistributeElementwise(MLIRContext *context, VectorLayoutAnalysis &analysis,
                        VectorLayoutOptions &options,
                        PatternBenefit benefit = 1)
      : OpTraitRewritePattern<OpTrait::Elementwise>(context, benefit),
        analysis(analysis), options(options) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!OpTrait::hasElementwiseMappableTraits(op))
      return failure();

    // Get the distributed operands.
    SmallVector<Value> operands;
    for (Value operand : op->getOperands()) {
      if (auto vectorOperand = dyn_cast<VectorValue>(operand)) {
        operand = getDistributed(rewriter, vectorOperand, options);
      }
      operands.push_back(operand);
    }

    // Get the new distributed vector types for the operation.
    SmallVector<Type> resultTypes;
    for (Value result : op->getResults()) {
      Type resultType = result.getType();

      // Distribute vector result types.
      if (auto vectorResult = dyn_cast<VectorValue>(result)) {
        resultType = VectorType::get(options.getDistributedShape(vectorResult),
                                     vectorResult.getType().getElementType());
      }
      resultTypes.push_back(resultType);
    }

    // Replace the original op with the distributed op.
    Operation *distributedOp =
        rewriter.create(op->getLoc(), op->getName().getIdentifier(), operands,
                        resultTypes, op->getAttrs());
    replaceOpWithDistributedValues(rewriter, op, options,
                                   distributedOp->getResults());
    return success();
  }

private:
  VectorLayoutAnalysis &analysis;
  VectorLayoutOptions &options;
};

/// A rewriter for the pattern rewriting driver.
class VectorDistributionRewriter : public PatternRewriter,
                                   RewriterBase::Listener {
public:
  VectorDistributionRewriter(MLIRContext *ctx, DenseSet<Operation *> &erasedOps)
      : PatternRewriter(context, this) {}
};

static bool canDistribute(Operation *op, VectorLayoutAnalysis &analysis) {
  auto values = llvm::to_vector_of<Value>(op->getOperands());
  llvm::append_range(values, op->getResults());

  // Check if all operands and results of this operation have a layout.
  return llvm::all_of(values, [&](Value value) -> bool {
    if (auto vectorValue = dyn_cast<VectorValue>(value)) {
      return analysis.getLayout<Attribute>(vectorValue) != nullptr;
    }
    return false;
  });
}

static void
debugPrintUniqueOperationNames(SmallVectorImpl<Operation *> &worklist,
                               DenseSet<Operation *> &exclude) {
  DenseSet<StringRef> uniqueNames;
  for (Operation *op : worklist) {
    if (exclude.contains(op))
      continue;
    uniqueNames.insert(op->getName().getStringRef());
  }

  for (StringRef name : uniqueNames) {
    llvm::dbgs().indent(2) << "* " << name << "\n";
  }
  LLVM_DEBUG(llvm::dbgs() << "\n");
}

static void applyVectorDistribution(Operation *root,
                                    const FrozenRewritePatternSet &patterns,
                                    VectorLayoutAnalysis &analysis,
                                    VectorLayoutOptions &options) {

  DenseSet<Operation *> erasedOps;
  SmallVector<Operation *> worklist;

  VectorDistributionRewriter rewriter(root->getContext(), erasedOps);
  PatternApplicator applicator(patterns);
  applicator.applyDefaultCostModel();

  // Collect all the operations to be distributed.
  LLVM_DEBUG(llvm::dbgs() << "Collecting operations to be distributed\n");
  root->walk([&](Operation *op) {
    if (canDistribute(op, analysis)) {
      worklist.push_back(op);
    }
  });
  LLVM_DEBUG(debugPrintUniqueOperationNames(worklist, erasedOps));

  // Note that the pattern application here never runs on a newly created
  // operation. It always runs on an existing operation. This ensures that no
  // invalidated state of the analysis is ever used.
  for (Operation *op : worklist) {
    if (erasedOps.contains(op))
      continue;

    LLVM_DEBUG(llvm::dbgs() << "Distributing: ");
    LLVM_DEBUG(op->print(llvm::dbgs(), OpPrintingFlags().skipRegions()));
    LLVM_DEBUG(llvm::dbgs() << "\n");
    if (failed(applicator.matchAndRewrite(op, rewriter))) {
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << ": Failed to distribute operation:\n");
      continue;
    }

    LLVM_DEBUG(llvm::dbgs().indent(2)
               << ": Successfully distributed operation:\n");
  }

  LLVM_DEBUG(llvm::dbgs() << "\nDistribution complete:\n\n");
  LLVM_DEBUG(llvm::dbgs() << "Operation names of ops not distributed:\n");
  LLVM_DEBUG(debugPrintUniqueOperationNames(worklist, erasedOps));
}

void distributeVectorOps(Operation *root, VectorLayoutOptions &options) {
  // Run the analysis and determine the layouts.
  VectorLayoutAnalysis analysis(root);
  options.setAnchorOps();
  if (failed(analysis.run()))
    return;
  LLVM_DEBUG(llvm::dbgs() << "Layout Analysis Completed Successfully :\n");
  LLVM_DEBUG(analysis.print(llvm::dbgs()));
  LLVM_DEBUG(llvm::dbgs() << "\n\n");

  // Run the distribution patterns.
  RewritePatternSet patterns(root->getContext());
  patterns.add<DistributeConstants, DistributeElementwise>(root->getContext(),
                                                           analysis, options);
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  return applyVectorDistributionDriver(root, frozenPatterns, analysis, options);
}

} // namespace mlir::iree_compiler
