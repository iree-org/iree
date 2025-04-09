// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUVECTORLOWERINGPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

//====---------------------------------------------------------------------===//
// Patterns for late vector op lowering.
//====---------------------------------------------------------------------===//

namespace {

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

// Example of unrolling with native shape chosen based on the result's final
// dimension. In this case the result is a 4x2 vector, so the 'native shape' is
// 1x2.
//
// clang-format off
//   %transposed = vector.transpose %cst, [1,0] : vector<2x4xf32> to vector<4x2xf32>
// clang-format on
//
//  becomes
//
// clang-format off
//   %cst = arith.constant dense<0.000000e+00> : vector<4x2xf32>
//   %0 = vector.extract_strided_slice %arg0 {offsets = [0, 0], sizes = [2, 1], strides = [1, 1]} : vector<2x4xf32> to vector<2x1xf32>
//   %1 = vector.transpose %0, [1, 0] : vector<2x1xf32> to vector<1x2xf32>
//   %2 = vector.insert_strided_slice %1, %cst {offsets = [0, 0], strides = [1, 1]} : vector<1x2xf32> into vector<4x2xf32>
//   %3 = vector.extract_strided_slice %arg0 {offsets = [0, 1], sizes = [2, 1], strides = [1, 1]} : vector<2x4xf32> to vector<2x1xf32>
//   %4 = vector.transpose %3, [1, 0] : vector<2x1xf32> to vector<1x2xf32>
//   %5 = vector.insert_strided_slice %4, %2 {offsets = [1, 0], strides = [1, 1]} : vector<1x2xf32> into vector<4x2xf32>
//   [...]
// clang-format on
//
//
// TODO(newling) Further analysis can improve this. For example flattening
// out unit dimensions and canonicalizations might help. Example:
//
// clang-format off
//   %t = vector.transpose %cst, [0,2,1] : vector<10x1x4xf32> to vector<10x4x1xf32>
// clang-format on
//
// should ideally not result in 40 extract-insert pairs of single f32 elements.
SmallVector<int64_t> getNativeVectorShapeImpl(vector::TransposeOp op) {
  return getWithLeadingOnes(op.getResultVectorType());
}

SmallVector<int64_t> getNativeVectorShapeImpl(vector::GatherOp op) {
  return getWithLeadingOnes(op.getVectorType());
}

SmallVector<int64_t> getNativeVectorShapeImpl(vector::ReductionOp op) {
  return getWithLeadingOnes(op.getSourceVectorType());
}

// As we do unrolling after lowering vector operations, it's not necessary to
// provide an unroll shape for most ops (as compared to SPIRV).
//
// Unrolling here assumes that the input has already been tiled such that the
// inner-most dimensions of vectors are the optimal hardware vector size to
// operate on.
std::optional<SmallVector<int64_t>> getNativeVectorShape(Operation *op) {

  // Unroll shape for elementwise operations. Example:
  //
  // clang-format off
  //   %2 = arith.truncf %1 : vector<10x4xf32> to vector<10x4xf16>
  // clang-format on
  //
  // will be unrolled to 10 arith.truncf operations on vectors of shape 1x4.
  if (OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1) {
    if (auto vecType = llvm::dyn_cast<VectorType>(op->getResultTypes()[0])) {
      return getWithLeadingOnes(vecType);
    }
  }

  return TypeSwitch<Operation *, std::optional<SmallVector<int64_t>>>(op)
      .Case<vector::ReductionOp, vector::TransposeOp, vector::GatherOp>(
          [](auto typedOp) { return getNativeVectorShapeImpl(typedOp); })
      .Default([](Operation *) { return std::nullopt; });
}

// Duplicate of
// https://github.com/llvm/llvm-project/pull/133988
// Pattern to rewrite a ShapeCast(PoisonOp) -> PoisonOp.
class ShapeCastPoisonFolder final
    : public OpRewritePattern<vector::ShapeCastOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::ShapeCastOp cast,
                                PatternRewriter &rewriter) const override {
    if (!cast.getSource().getDefiningOp<ub::PoisonOp>())
      return failure();
    rewriter.replaceOpWithNewOp<ub::PoisonOp>(cast, cast.getType());
    return success();
  }
};

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

    MLIRContext *context = funcOp.getContext();

    {
      // Lower high level vector operations like contract or multidim reduce ops
      // to lower level vector ops.
      RewritePatternSet contractLoweringPatterns(context);
      auto options =
          vector::VectorTransformsOptions().setVectorTransformsOptions(
              vector::VectorContractLowering::OuterProduct);
      vector::populateVectorTransferPermutationMapLoweringPatterns(
          contractLoweringPatterns);
      vector::TransposeOp::getCanonicalizationPatterns(contractLoweringPatterns,
                                                       context);
      vector::populateVectorBroadcastLoweringPatterns(contractLoweringPatterns);
      vector::populateVectorContractLoweringPatterns(
          contractLoweringPatterns, options.vectorContractLowering);
      contractLoweringPatterns.add<PromoteContractOperands>(context);
      vector::populateVectorGatherLoweringPatterns(contractLoweringPatterns);
      vector::populateVectorMaskOpLoweringPatterns(contractLoweringPatterns);
      vector::populateVectorShapeCastLoweringPatterns(contractLoweringPatterns);
      vector::populateVectorMultiReductionLoweringPatterns(
          contractLoweringPatterns,
          vector::VectorMultiReductionLowering::InnerParallel);
      if (failed(applyPatternsGreedily(funcOp,
                                       std::move(contractLoweringPatterns)))) {
        return signalPassFailure();
      }
    }

    RewritePatternSet vectorToLoopsPatterns(context);
    VectorTransferToSCFOptions vectorToSCFOptions;
    vectorToSCFOptions.enableFullUnroll();
    populateVectorToSCFConversionPatterns(vectorToLoopsPatterns,
                                          vectorToSCFOptions);
    memref::populateFoldMemRefAliasOpPatterns(vectorToLoopsPatterns);
    vector::populateVectorTransferLoweringPatterns(vectorToLoopsPatterns);
    if (failed(
            applyPatternsGreedily(funcOp, std::move(vectorToLoopsPatterns)))) {
      return signalPassFailure();
    }

    if (!unroll && flatten) {
      funcOp->emitOpError("can only flatten if unrolled, cannot have ")
          << "unroll=false and flatten=true";
      return signalPassFailure();
    }

    if (!unroll) {
      return;
    }

    // Canonicalize in preparation for unrolling.
    {
      RewritePatternSet patterns(context);
      vector::InsertOp::getCanonicalizationPatterns(patterns, context);
      vector::ExtractOp::getCanonicalizationPatterns(patterns, context);
      vector::BroadcastOp::getCanonicalizationPatterns(patterns, context);
      vector::ShapeCastOp::getCanonicalizationPatterns(patterns, context);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // Unroll vector operations to the optimal size for a single thread to
    // operate on.
    {
      RewritePatternSet patterns(context);
      auto opts = vector::UnrollVectorOptions().setNativeShapeFn(
          [=](auto op) { return getNativeVectorShape(op); });
      vector::populateVectorUnrollPatterns(patterns, opts);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    if (!flatten) {
      return;
    }

    // Now lower vector transpose ops. TODO(newling) unify the approach
    // of lowering transpose, there are 2 odd things about needing to
    // do it here: 1) why not lowered with the other ops? 2) Why do we
    // need to unroll it before lowering it?
    //
    // This converts transpose ops into extract and insert pairs, in preparation
    // of further transformations to canonicalize/cancel.
    {
      RewritePatternSet patterns(context);
      auto options =
          vector::VectorTransformsOptions().setVectorTransposeLowering(
              vector::VectorTransposeLowering::EltWise);
      vector::populateVectorTransposeLoweringPatterns(
          patterns, options.vectorTransposeLowering);
      vector::populateVectorShapeCastLoweringPatterns(patterns);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // Cast away leading 1 dimensions, and related canonicalizers.
    {
      RewritePatternSet patterns(context);
      vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
      vector::populateVectorInsertExtractStridedSliceDecompositionPatterns(
          patterns);
      vector::ExtractOp::getCanonicalizationPatterns(patterns, context);
      vector::BroadcastOp::getCanonicalizationPatterns(patterns, context);
      vector::ShapeCastOp::getCanonicalizationPatterns(patterns, context);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // Reduce rank of ops with vector.shape_casts, and hope that they cancel
    // with each other.
    {
      RewritePatternSet patterns(context);
      // extract(shape_cast(x)) gets 'folded' to extract(x), which can prevent
      // shape_cast cancelling, so disabling folding in this set of patterns.
      GreedyRewriteConfig config;
      config.fold = false;
      populateFlattenVectorExtractInsertPatterns(patterns);
      patterns.insert<ShapeCastPoisonFolder>(context);
      populateForOpInductionVarShapePatterns(patterns);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
        return signalPassFailure();
      }
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
