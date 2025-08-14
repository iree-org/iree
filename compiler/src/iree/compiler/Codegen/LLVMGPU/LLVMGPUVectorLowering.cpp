// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/AMDGPU/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/PatternMatch.h"
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

/// true: vector
/// false: vector
/// pred: i1
///
/// select(pred, true, false) -> broadcast(pred)
/// select(pred, false, true) -> broadcast(not(pred))
///
/// Ideally, this would be a canonicalization pattern on arith::SelectOp, but
/// we cannot have arith depending on vector. Also, it would implicitly force
/// users only using arith and vector dialect to use vector dialect. Since
/// upstream does not have a mechanism of registering canonicalization without
/// adding dependencies like this, we manually add it where it is needed.
struct FoldI1SelectToBroadcast final : OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp selectOp,
                                PatternRewriter &rewriter) const override {
    auto vecType = dyn_cast<VectorType>(selectOp.getType());
    if (!vecType || !vecType.getElementType().isInteger(1)) {
      return failure();
    }

    // Vector conditionals do not need broadcast and are already handled by
    // the arith.select folder.
    Value pred = selectOp.getCondition();
    if (isa<VectorType>(pred.getType())) {
      return failure();
    }

    std::optional<int64_t> trueInt =
        getConstantIntValue(selectOp.getTrueValue());
    std::optional<int64_t> falseInt =
        getConstantIntValue(selectOp.getFalseValue());
    if (!trueInt || !falseInt) {
      return failure();
    }

    // Redundant selects are already handled by arith.select canonicalizations.
    if (trueInt.value() == falseInt.value()) {
      return failure();
    }

    // The only remaining possibilities are:
    //
    // select(pred, true, false)
    // select(pred, false, true)

    // select(pred, false, true) -> select(not(pred), true, false)
    if (trueInt.value() == 0) {
      // TODO: flip the condition here to handle through the existing path.
      return failure();
    }

    /// select(pred, true, false) -> broadcast(pred)
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(
        selectOp, vecType.clone(rewriter.getI1Type()), pred);
    return success();
  }
};

// Set FMF to fold arith.mulf + arith.addf -> math.fma and thus enable
// optimizations w.r.t vector.multi_reduction(fma).
struct SetMulAddFMF final : OpRewritePattern<vector::MultiDimReductionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp redOp,
                                PatternRewriter &rewriter) const override {
    if (redOp.getKind() != vector::CombiningKind::ADD) {
      return failure();
    }

    Value src = redOp.getSource();
    auto addOp = src.getDefiningOp<arith::AddFOp>();
    if (!addOp) {
      return failure();
    }

    auto mulOp = addOp.getLhs().getDefiningOp<arith::MulFOp>();
    if (!mulOp) {
      mulOp = addOp.getRhs().getDefiningOp<arith::MulFOp>();
    }

    if (!mulOp) {
      return failure();
    }

    constexpr auto none = arith::FastMathFlags::none;
    if (mulOp.getFastmath() != none || addOp.getFastmath() != none) {
      return failure();
    }

    constexpr auto contract = arith::FastMathFlags::contract;
    rewriter.modifyOpInPlace(mulOp, [&] { mulOp.setFastmath(contract); });
    rewriter.modifyOpInPlace(addOp, [&] { addOp.setFastmath(contract); });

    return success();
  }
};

// TODO: scale to more than 2 rows
// struct LowerMultiReductionFMAToContract final : OpRewritePattern<vector::MultiDimReductionOp> {
//   using OpRewritePattern::OpRewritePattern;

//   LogicalResult matchAndRewrite(vector::MultiDimReductionOp redOp,
//                                 PatternRewriter &rewriter) const override {
//     if (redOp.getKind() != vector::CombiningKind::ADD) {
//       return failure();
//     }

//     Value src = redOp.getSource();
//     auto fmaOp = src.getDefiningOp<math::FmaOp>();
//     if (!fmaOp) {
//       return failure();
//     }

//     Location loc = redOp.getLoc();
//     Value lhs = fmaOp.getOperand(0);
//     Value rhs = fmaOp.getOperand(1);
//     Value acc = redOp.getAcc();

//     auto srcType = dyn_cast<VectorType>(fmaOp.getResult().getType());
//     auto resType = dyn_cast<VectorType>(redOp.getResult().getType());
//     resType.dump();

//     ArrayRef<int64_t> reductionDims = redOp.getReductionDims();
//     SmallVector<int64_t> parallelDims;

//     // this can definitely be made better
//     for (int64_t i{}; i < srcType.getRank(); i++) {
//       if (!llvm::is_contained(reductionDims, i)) {
//         parallelDims.push_back(i);
//       }
//     }

//     int64_t parallelSize = 1;
//     int64_t reductionSize = 1;

//     for (int64_t dim : parallelDims) {
//       // 2 * 1
//       parallelSize *= srcType.getDimSize(dim);
//     }
//     for (int64_t dim : reductionDims) {
//       // 8
//       reductionSize *= srcType.getDimSize(dim);
//     }

//     // 2 x 8 x f32
//     auto flat2DType = VectorType::get({parallelSize, reductionSize},
//                                       srcType.getElementType());
//     // 2 x f32
//     auto flat1DType = VectorType::get({parallelSize}, srcType.getElementType());

//     Value flatLhs = rewriter.create<vector::ShapeCastOp>(loc, flat2DType, lhs);
//     Value flatRhs = rewriter.create<vector::ShapeCastOp>(loc, flat2DType, rhs);
//     Value flatAcc = rewriter.create<vector::ShapeCastOp>(loc, flat1DType, acc);

//     MLIRContext *ctx = rewriter.getContext();

//     AffineExpr d0 = getAffineDimExpr(0, ctx); // parallel dimension
//     AffineExpr d1 = getAffineDimExpr(1, ctx); // reduction dimension

//     auto lhsMap = AffineMap::get(2, 0, {d0, d1}, ctx);
//     auto rhsMap = AffineMap::get(2, 0, {d0, d1}, ctx);
//     auto accMap = AffineMap::get(2, 0, {d0}, ctx);

//     // Step 6: Create the contract operation
//     auto indexingMaps =
//         rewriter.getAffineMapArrayAttr({lhsMap, rhsMap, accMap});
//     auto iteratorTypes =
//         rewriter.getArrayAttr({rewriter.getStringAttr("parallel"),
//                                rewriter.getStringAttr("reduction")});

//     auto contractOp = rewriter.create<vector::ContractionOp>(
//         loc, flatLhs, flatRhs, flatAcc, indexingMaps, iteratorTypes,
//         vector::CombiningKindAttr::get(ctx, vector::CombiningKind::ADD));

//     // Step 7: Shape cast result back to original shape
//     // Example: vector<2xf32> -> vector<2x1x1xf32>
//     Value result = rewriter.create<vector::ShapeCastOp>(loc, resType,
//                                                         contractOp.getResult());

//     // Step 8: Add attribute to mark this as FMA-derived
//     // (so later lowering knows to preserve FMA)
//     contractOp->setAttr("fma_chain", rewriter.getUnitAttr());

//     rewriter.replaceOp(redOp, result);
//     return success();
//   }
// };

struct MultiReduceFMAToContract final
    : OpRewritePattern<vector::MultiDimReductionOp> {
  using OpRewritePattern::OpRewritePattern;

  static bool isZeroSplat(Value v) {
    // Accept only explicit, foldable zeros.
    if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
      Attribute attr = cst.getValue();
      // Vector/array: dense splat 0.0
      if (auto dense = dyn_cast<DenseElementsAttr>(attr)) {
        if (dense.isSplat()) {
          if (auto ft = dyn_cast<FloatType>(dense.getElementType())) {
            return dense.getSplatValue<APFloat>().isZero();
          }
        }
        return false;
      }
      // Scalar float 0.0
      if (auto fa = dyn_cast<FloatAttr>(attr))
        return fa.getValue().isZero();
    }
    return false;
  }

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp reduceOp,
                                PatternRewriter &rewriter) const override {
    if (reduceOp.getKind() != vector::CombiningKind::ADD) {
      return failure();
    }

    auto fma = reduceOp.getSource().getDefiningOp<math::FmaOp>();
    if (!fma)
      return failure();

    Value a = fma.getOperand(0);
    Value b = fma.getOperand(1);
    Value c = fma.getOperand(2);
    auto aVT = dyn_cast<VectorType>(a.getType());
    auto bVT = dyn_cast<VectorType>(b.getType());
    auto srcVT = dyn_cast<VectorType>(fma.getResult().getType());
    auto resVT = dyn_cast<VectorType>(reduceOp.getResult().getType());
    if (!aVT || !bVT || !srcVT || !resVT) {
      return failure();
    }
    if (aVT != bVT || aVT != srcVT) {
      return failure();
    }
    if (!isa<FloatType>(srcVT.getElementType())) {
      return failure();
    }
    if (reduceOp.getAcc().getType() != resVT) {
      return failure();
    }

    // TODO: support c != 0
    if (!isZeroSplat(c)) {
      return failure();
    }

    // Build iterator types and indexing maps:
    //  A: id, B: id, C: projection of non-reduction dims; iterator per dim.
    SmallVector<bool> mask = reduceOp.getReductionMask();
    if (mask.size() != srcVT.getRank()) {
      return failure();
    }

    auto srcMap = rewriter.getMultiDimIdentityMap(mask.size());

    SmallVector<AffineExpr> projExprs;
    SmallVector<vector::IteratorType> iters;
    projExprs.reserve(mask.size());
    iters.reserve(mask.size());
    for (int i = 0, e = mask.size(); i < e; ++i) {
      if (!mask[i]) {
        iters.push_back(vector::IteratorType::parallel);
        projExprs.push_back(rewriter.getAffineDimExpr(i));
      } else {
        iters.push_back(vector::IteratorType::reduction);
      }
    }

    auto dstMap = AffineMap::get(mask.size(), /*symbols=*/0, projExprs,
                                 reduceOp.getContext());
    auto itersAttr = rewriter.getArrayAttr(llvm::to_vector(
        llvm::map_range(iters, [&](vector::IteratorType t) -> Attribute {
          return vector::IteratorTypeAttr::get(rewriter.getContext(), t);
        })));
    auto mapsAttr = rewriter.getAffineMapArrayAttr({srcMap, srcMap, dstMap});

    rewriter.replaceOpWithNewOp<vector::ContractionOp>(
        reduceOp, a, b, reduceOp.getAcc(), mapsAttr, itersAttr);
    return success();
  }
};

struct LLVMGPUVectorLoweringPass final
    : impl::LLVMGPUVectorLoweringPassBase<LLVMGPUVectorLoweringPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<vector::VectorDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<math::MathDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    // Uplift arith ops to math.fma before lowering high level vector ops.
    {
      RewritePatternSet fmaPatterns(ctx);
      fmaPatterns.add<SetMulAddFMF>(ctx, PatternBenefit(2));
      populateUpliftToFMAPatterns(fmaPatterns);
      fmaPatterns.add<MultiReduceFMAToContract>(ctx);
      if (failed(applyPatternsGreedily(funcOp, std::move(fmaPatterns)))) {
        return signalPassFailure();
      }
    }

    // {
    //   // Lower high level vector operations like contract or multidim reduce ops
    //   // to lower level vector ops.
    //   RewritePatternSet contractLoweringPatterns(funcOp.getContext());
    //   auto options =
    //       vector::VectorTransformsOptions().setVectorTransformsOptions(
    //           vector::VectorContractLowering::OuterProduct);
    //   vector::populateVectorTransferPermutationMapLoweringPatterns(
    //       contractLoweringPatterns);
    //   vector::TransposeOp::getCanonicalizationPatterns(contractLoweringPatterns,
    //                                                    funcOp.getContext());
    //   vector::populateVectorBroadcastLoweringPatterns(contractLoweringPatterns);
    //   vector::populateVectorContractLoweringPatterns(
    //       contractLoweringPatterns, options.vectorContractLowering);
    //   contractLoweringPatterns.add<PromoteContractOperands>(
    //       funcOp->getContext());
    //   vector::populateVectorGatherLoweringPatterns(contractLoweringPatterns);
    //   vector::populateVectorMaskOpLoweringPatterns(contractLoweringPatterns);
    //   vector::populateVectorShapeCastLoweringPatterns(contractLoweringPatterns);
    //   vector::populateVectorMultiReductionLoweringPatterns(
    //       contractLoweringPatterns,
    //       vector::VectorMultiReductionLowering::InnerReduction);
    //   if (failed(applyPatternsGreedily(funcOp,
    //                                    std::move(contractLoweringPatterns)))) {
    //     return signalPassFailure();
    //   }
    // }

    {
      RewritePatternSet vectorToLoopsPatterns(&getContext());
      VectorTransferToSCFOptions vectorToSCFOptions;
      vectorToSCFOptions.enableFullUnroll();
      populateVectorToSCFConversionPatterns(vectorToLoopsPatterns,
                                            vectorToSCFOptions);
      memref::populateFoldMemRefAliasOpPatterns(vectorToLoopsPatterns);
      vector::populateVectorTransferLoweringPatterns(vectorToLoopsPatterns);
      if (failed(applyPatternsGreedily(funcOp,
                                       std::move(vectorToLoopsPatterns)))) {
        return signalPassFailure();
      }
    }

    // Cleanup canonicalization for masking and other basic vector operations.
    {
      RewritePatternSet patterns(ctx);
      vector::CreateMaskOp::getCanonicalizationPatterns(patterns, ctx);
      vector::ConstantMaskOp::getCanonicalizationPatterns(patterns, ctx);
      vector::ExtractOp::getCanonicalizationPatterns(patterns, ctx);
      vector::BroadcastOp::getCanonicalizationPatterns(patterns, ctx);
      arith::SelectOp::getCanonicalizationPatterns(patterns, ctx);
      patterns.add<FoldI1SelectToBroadcast>(ctx);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
