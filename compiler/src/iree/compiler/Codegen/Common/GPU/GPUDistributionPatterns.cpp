// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpDefinition.h"
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
    auto constant = dyn_cast<VectorValue>(constantOp.getResult());
    if (!constant)
      return failure();

    // Only handle splat values for now.
    auto attr = dyn_cast<SplatElementsAttr>(constantOp.getValue());
    if (!attr)
      return failure();

    VectorLayoutInterface layout = signature[constant];

    // Replace the original op with the distributed op.
    Type elementType = constant.getType().getElementType();
    auto vectorType =
        VectorType::get(layout.getDistributedShape(), elementType);
    auto distributedOp = rewriter.create<arith::ConstantOp>(
        constantOp.getLoc(), vectorType,
        SplatElementsAttr::get(vectorType, attr.getSplatValue<Attribute>()));
    replaceOpWithDistributedValues(rewriter, constantOp,
                                   distributedOp->getResult(0));
    return success();
  }
};

struct DistributePoison final : OpDistributionPattern<ub::PoisonOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(ub::PoisonOp poisonOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {

    auto poisonVal = dyn_cast<VectorValue>(poisonOp.getResult());
    if (!poisonVal)
      return failure();

    SmallVector<int64_t> distributedShape =
        signature[poisonVal].getDistributedShape();

    Type elementType = poisonVal.getType().getElementType();
    auto vectorType = VectorType::get(distributedShape, elementType);
    auto distributedOp =
        ub::PoisonOp::create(rewriter, poisonVal.getLoc(), vectorType);
    replaceOpWithDistributedValues(rewriter, poisonOp,
                                   distributedOp->getResult(0));
    return success();
  }
};

struct DistributeElementwise final
    : OpTraitDistributionPattern<OpTrait::Elementwise> {
  using OpTraitDistributionPattern::OpTraitDistributionPattern;

  LogicalResult matchAndRewrite(Operation *op, DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    // Check if this operation has elementwise mappable traits. This is
    // more restricted than only having elementwise trait.
    if (!OpTrait::hasElementwiseMappableTraits(op)) {
      return failure();
    }

    // Get the distributed operands.
    SmallVector<Value> operands;
    for (Value operand : op->getOperands()) {
      if (auto vectorOperand = dyn_cast<VectorValue>(operand)) {
        operand = DistributionPattern::getDistributed(rewriter, vectorOperand,
                                                      signature[vectorOperand]);
      }
      operands.push_back(operand);
    }

    // Get the new distributed vector types for the operation.
    SmallVector<Type> resultTypes;
    for (Value result : op->getResults()) {
      Type resultType = result.getType();

      // Distribute vector result types.
      if (auto vectorResult = dyn_cast<VectorValue>(result)) {
        VectorLayoutInterface resLayout = signature[vectorResult];
        resultType = VectorType::get(resLayout.getDistributedShape(),
                                     vectorResult.getType().getElementType());
      }
      resultTypes.push_back(resultType);
    }

    // Replace the original op with the distributed op.
    Operation *distributedOp = mlir::clone(rewriter, op, resultTypes, operands);

    DistributionPattern::replaceOpWithDistributedValues(
        rewriter, op, distributedOp->getResults());
    return success();
  }
};

/// Given a projected permutation, get a reduced permutation, i.e. without
/// the projected dimensions.
static SmallVector<int64_t>
getReducedPermutation(AffineMap permutationMap,
                      llvm::SmallBitVector &unusedDims) {
  assert(permutationMap.isProjectedPermutation() &&
         "permutation map should be a projected permutation.");
  // TODO: The permutation map may also have broadcasting. Currently, we do not
  // handle it. This can be fixed by adding a "BROADCAST" dimension in the
  // layout.

  unusedDims.clear();
  unusedDims.resize(permutationMap.getNumDims(), true);

  for (AffineExpr dimExpr : permutationMap.getResults()) {
    int64_t pos = cast<AffineDimExpr>(dimExpr).getPosition();
    unusedDims[pos] = false;
  }

  SmallVector<int64_t> permutation;
  permutation.reserve(permutationMap.getNumResults());

  AffineMap reducedMap = compressUnusedDims(permutationMap);
  for (AffineExpr dimExpr : reducedMap.getResults()) {
    int64_t pos = cast<AffineDimExpr>(dimExpr).getPosition();
    permutation.push_back(pos);
  }

  return permutation;
}

struct DistributeScfFor final : OpDistributionPattern<scf::ForOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    Block *oldLoopBody = forOp.getBody();

    if (forOp.getInitArgs().empty()) {
      return failure();
    }

    // The new vector init_args of the loop.
    SmallVector<Value> newInitArgs;
    for (Value initArg : forOp.getInitArgs()) {
      if (auto vectorInitArg = dyn_cast<VectorValue>(initArg)) {
        initArg =
            getDistributed(rewriter, vectorInitArg, signature[vectorInitArg]);
      }
      newInitArgs.push_back(initArg);
    }

    auto newForOp = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newInitArgs);
    newForOp->setAttrs(forOp->getAttrs());
    Block *loopBody = newForOp.getBody();

    // Set up new iter_args. The loop body uses SIMD, so wrap the SIMD iter_args
    // of the new loop op into ToSIMDOps.
    rewriter.setInsertionPointToStart(loopBody);
    SmallVector<Value> iterArgs = getBbArgsReplacements(
        rewriter, newForOp.getRegionIterArgs(), forOp.getInitArgs());
    iterArgs.insert(iterArgs.begin(), newForOp.getInductionVar());

    // Move loop body to new loop.
    rewriter.mergeBlocks(oldLoopBody, loopBody, iterArgs);

    if (failed(distributeYield(rewriter, newForOp))) {
      return failure();
    }

    // Repleace loop results.
    replaceOpWithDistributedValues(rewriter, forOp, newForOp.getResults());
    return success();
  }

  LogicalResult distributeYield(PatternRewriter &rewriter,
                                scf::ForOp forOp) const {
    scf::YieldOp yieldOp =
        llvm::cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    std::optional<DistributionSignature> maybeSignature =
        getOpSignature(yieldOp);
    if (!maybeSignature) {
      return failure();
    }
    DistributionSignature signature = *maybeSignature;

    // Get the distributed operands.
    SmallVector<Value> operands;
    for (Value operand : yieldOp->getOperands()) {
      if (auto vectorOperand = dyn_cast<VectorValue>(operand)) {
        operand = DistributionPattern::getDistributed(rewriter, vectorOperand,
                                                      signature[vectorOperand]);
      }
      operands.push_back(operand);
    }

    // Since this operation has no results, we can directly replace it using
    // the standard API.
    auto distributedYieldOp =
        rewriter.create<scf::YieldOp>(yieldOp.getLoc(), operands);
    rewriter.replaceOp(yieldOp, distributedYieldOp);
    return success();
  }

  /// Helper function for loop distribution. Given a list of bbArgs of the new
  /// (distributed) loop op, wrap the distributed vector args (now distributed)
  /// into ToSIMDOps, so that the block body can be moved over to the new op.
  SmallVector<Value> getBbArgsReplacements(RewriterBase &rewriter,
                                           Block::BlockArgListType bbArgs,
                                           ValueRange oldInits) const {
    SmallVector<Value> replacements;
    for (auto [bbArg, oldInit] : llvm::zip_equal(bbArgs, oldInits)) {
      Value val = bbArg;
      if (auto oldVectorInit = dyn_cast<VectorValue>(oldInit)) {
        val = rewriter.create<IREE::VectorExt::ToSIMDOp>(
            oldVectorInit.getLoc(), oldVectorInit.getType(), val);
      }
      replacements.push_back(val);
    }
    return replacements;
  }
};

struct DistributeTrivialLayoutConversions final
    : OpDistributionPattern<IREE::VectorExt::ToLayoutOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(IREE::VectorExt::ToLayoutOp toLayoutOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    auto input = cast<VectorValue>(toLayoutOp.getInput());
    auto output = cast<VectorValue>(toLayoutOp.getOutput());
    VectorLayoutInterface currentLayout = signature[input];
    VectorLayoutInterface targetLayout = signature[output];

    if (!currentLayout) {
      return rewriter.notifyMatchFailure(toLayoutOp, "No layout set on input");
    }

    if (!targetLayout) {
      return rewriter.notifyMatchFailure(toLayoutOp, "No layout set on output");
    }

    if (currentLayout != targetLayout) {
      return rewriter.notifyMatchFailure(toLayoutOp,
                                         "Non-trivial layout conversion.");
    }

    rewriter.replaceOp(toLayoutOp, toLayoutOp.getOperand());
    return success();
  }
};

struct DistributeGather final : OpDistributionPattern<vector::GatherOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(vector::GatherOp gatherOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    VectorValue result = gatherOp.getResult();
    VectorValue indexVec = gatherOp.getIndexVec();
    VectorValue mask = gatherOp.getMask();
    VectorValue passThru = gatherOp.getPassThru();

    VectorLayoutInterface resultLayout = signature[result];
    VectorLayoutInterface indicesLayout = signature[indexVec];
    VectorLayoutInterface maskLayout = signature[mask];
    VectorLayoutInterface passThruLayout = signature[passThru];

    if (!resultLayout) {
      return rewriter.notifyMatchFailure(gatherOp,
                                         "result does not have layout");
    }
    if (!indicesLayout) {
      return rewriter.notifyMatchFailure(gatherOp,
                                         "indices does not have layout");
    }
    if (!maskLayout) {
      return rewriter.notifyMatchFailure(gatherOp, "mask does not have layout");
    }
    if (!passThruLayout) {
      return rewriter.notifyMatchFailure(gatherOp,
                                         "passThru does not have layout");
    }

    SmallVector<int64_t> distributedShape = resultLayout.getDistributedShape();
    Type elementType = result.getType().getElementType();
    VectorType distributedType = VectorType::get(distributedShape, elementType);

    // Simply distribute all operands and results.
    VectorValue distributed = rewriter.create<vector::GatherOp>(
        gatherOp.getLoc(), distributedType, gatherOp.getBase(),
        gatherOp.getIndices(),
        getDistributed(rewriter, indexVec, indicesLayout),
        getDistributed(rewriter, mask, maskLayout),
        getDistributed(rewriter, passThru, passThruLayout));

    replaceOpWithDistributedValues(rewriter, gatherOp, distributed);
    return success();
  }
};

/// Distribute a 0-rank vector to scalar vector.extract conversion.
struct DistributeTrivialExtract final
    : OpDistributionPattern<vector::ExtractOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(vector::ExtractOp extractOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    if (extractOp.getSourceVectorType().getRank() != 0) {
      return rewriter.notifyMatchFailure(
          extractOp, "Only 0-rank vector extractions supported");
    }

    VectorValue source = extractOp.getVector();
    VectorLayoutInterface sourceLayout = signature[source];

    Value distributed = rewriter.create<vector::ExtractOp>(
        extractOp.getLoc(), getDistributed(rewriter, source, sourceLayout),
        ArrayRef<int64_t>{});

    replaceOpWithDistributedValues(rewriter, extractOp, distributed);

    return success();
  }
};

} // namespace

void populateGPUDistributionPatterns(RewritePatternSet &patterns) {
  patterns.add<DistributeConstants, DistributePoison, DistributeScfFor,
               DistributeTrivialExtract>(patterns.getContext());
  // Elementwise patterns.
  patterns.add<DistributeElementwise>(patterns.getContext());
  patterns.add<DistributeTrivialLayoutConversions>(patterns.getContext());
  // Gather patterns.
  patterns.add<DistributeGather>(patterns.getContext());
}

}; // namespace mlir::iree_compiler
