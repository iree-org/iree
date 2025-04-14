// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUVECTORLOWERINGPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

//====---------------------------------------------------------------------===//
// Patterns for late vector op lowering.
//====---------------------------------------------------------------------===//

namespace {

// TODO(newling)
//
// Convert as many ops as possible to shape_casts.
//
// broadcast
// extract
// extract_strided_slice
// shape_cast
// transpose
//
// See example in vector_lowering.mlir (transpose -> extract where both are just
// shape_casts).

struct PromoteContractOperands final
    : public vector::MaskableOpRewritePattern<vector::ContractionOp> {
  using MaskableOpRewritePattern::MaskableOpRewritePattern;

  FailureOr<Value>
  matchAndRewriteMaskableOp(vector::ContractionOp contractOp,
                            vector::MaskingOpInterface maskOp,
                            PatternRewriter &rewriter) const override {
    Type operandElType = getElementTypeOrSelf(contractOp.getLhsType());
    Type resultElType = getElementTypeOrSelf(contractOp.getResultType());

    if (operandElType == resultElType) {
      return failure();
    }

    Location loc = contractOp.getLoc();
    Value lhs =
        promoteToElementType(loc, rewriter, contractOp.getLhs(), resultElType);
    Value rhs =
        promoteToElementType(loc, rewriter, contractOp.getRhs(), resultElType);

    auto replacement = rewriter.create<vector::ContractionOp>(
        loc, lhs, rhs, contractOp.getAcc(), contractOp.getIndexingMaps(),
        contractOp.getIteratorTypes());

    if (!maskOp) {
      return replacement.getResult();
    }
    auto maskedOp = vector::maskOperation(
        rewriter, replacement, maskOp.getMask(), maskOp.getPassthru());
    return maskedOp->getResult(0);
  }

  Value promoteToElementType(Location loc, RewriterBase &rewriter, Value v,
                             Type dstElementType) const {
    Type elementType = getElementTypeOrSelf(v.getType());
    if (elementType == dstElementType)
      return v;

    // vector.contract only allows extension on operands.
    assert(elementType.getIntOrFloatBitWidth() <=
               dstElementType.getIntOrFloatBitWidth() &&
           "vector.contract does not allow truncation of operands");

    Type promotedType = dstElementType;
    if (auto vecType = dyn_cast<VectorType>(v.getType()))
      promotedType = vecType.clone(promotedType);

    if (isa<FloatType>(dstElementType))
      return rewriter.create<arith::ExtFOp>(loc, promotedType, v);
    // For integer types, vector.contract only supports signless integer types
    // and promotion happens via sign extension.
    return rewriter.create<arith::ExtSIOp>(loc, promotedType, v);
  }
};

SmallVector<int64_t> getWithLeadingOnes(VectorType vectorType) {
  SmallVector<int64_t> nativeSize(vectorType.getRank(), 1);
  if (vectorType.getRank() > 0) {
    nativeSize.back() = vectorType.getShape().back();
  }
  return nativeSize;
}

std::optional<SmallVector<int64_t>> getNativeVectorShape(Operation *op) {

  // Example: elementwise operation `vector<10x4xf32> to vector<10x4xf16>`
  // will be unrolled to 10 elementwise operations on vectors of shape 1x4.
  if (OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1) {
    if (auto vecType = llvm::dyn_cast<VectorType>(op->getResultTypes()[0])) {
      return getWithLeadingOnes(vecType);
    }
  }

  // Unroll vector.transpose in all but the inner-most dimension of result.
  // Example: A transpose `vector<2x4xf32> to vector<4x2xf32>` results in 4
  // extract->insert_strided_slice pairs.
  //
  // An alternative is to use `populateVectorTransposeLoweringPatterns`
  // which always creates scalar extract-insert pairs.
  //
  // TODO(newling) reconsider the optimal strategy for this.
  if (auto transposeOp = llvm::dyn_cast<vector::TransposeOp>(op)) {
    return getWithLeadingOnes(transposeOp.getResultVectorType());
  }
  return std::nullopt;
}

struct LLVMGPUVectorLoweringPass final
    : impl::LLVMGPUVectorLoweringPassBase<LLVMGPUVectorLoweringPass> {
  using impl::LLVMGPUVectorLoweringPassBase<
      LLVMGPUVectorLoweringPass>::LLVMGPUVectorLoweringPassBase;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<vector::VectorDialect>();
    registry.insert<scf::SCFDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();

    auto addVectorCanonicalizationPatterns = [](RewritePatternSet &patterns) {
      MLIRContext *ctx = patterns.getContext();
      vector::BroadcastOp::getCanonicalizationPatterns(patterns, ctx);
      vector::ExtractOp::getCanonicalizationPatterns(patterns, ctx);
      vector::ExtractStridedSliceOp::getCanonicalizationPatterns(patterns, ctx);
      vector::InsertOp::getCanonicalizationPatterns(patterns, ctx);
      vector::InsertStridedSliceOp::getCanonicalizationPatterns(patterns, ctx);
      vector::ShapeCastOp::getCanonicalizationPatterns(patterns, ctx);
      vector::TransposeOp::getCanonicalizationPatterns(patterns, ctx);
    };

    MLIRContext *context = funcOp.getContext();

    // Remove permutation_map, replace with explict broadcast and transpose ops
    // (which we immediately try to canonicalize away).
    {
      RewritePatternSet patterns(context);
      vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
      addVectorCanonicalizationPatterns(patterns);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // Conversions and unrolls.
    {
      RewritePatternSet patterns(context);
      // Unroll broadcast, leaving rank-1 broadcast.
      vector::populateVectorBroadcastLoweringPatterns(patterns);
      // Unroll gather, leaving rank-1 gathers.
      vector::populateVectorGatherLoweringPatterns(patterns);
      // Unroll create_mask, leaving rank-1 create_masks.
      vector::populateVectorMaskOpLoweringPatterns(patterns);
      // contract to fma.
      vector::populateVectorContractLoweringPatterns(
          patterns, vector::VectorContractLowering::OuterProduct);
      patterns.add<PromoteContractOperands>(context);
      // multi_reduce to arith ops.
      vector::populateVectorMultiReductionLoweringPatterns(
          patterns, vector::VectorMultiReductionLowering::InnerParallel);
      // Remaining vector and other dialect ops have unrolling/lowering handled
      // by 'generic' vector unrolling.
      auto opts = vector::UnrollVectorOptions().setNativeShapeFn(
          [=](auto op) { return getNativeVectorShape(op); });
      vector::populateVectorUnrollPatterns(patterns, opts);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // transfer_read -> load and transfer_write -> store.
    {
      RewritePatternSet patterns(context);
      VectorTransferToSCFOptions vectorToSCFOptions;
      vectorToSCFOptions.enableFullUnroll();
      populateVectorToSCFConversionPatterns(patterns, vectorToSCFOptions);
      vector::populateVectorTransferLoweringPatterns(patterns);
      memref::populateFoldMemRefAliasOpPatterns(patterns);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // Canonicalize.
    {
      RewritePatternSet patterns(context);
      addVectorCanonicalizationPatterns(patterns);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // Less desirable unrolls, delayed till here in case previous
    // canonicalization can eliminate them.
    {
      RewritePatternSet patterns(context);
      // shape_cast to extract-like and insert-like ops.
      vector::populateVectorShapeCastLoweringPatterns(patterns);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // Canonicalize.
    {
      RewritePatternSet patterns(context);
      addVectorCanonicalizationPatterns(patterns);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
