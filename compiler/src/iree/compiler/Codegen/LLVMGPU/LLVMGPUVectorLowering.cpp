// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "llvm/ADT/SmallBitVector.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/AMDGPU/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
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

struct ContractAddToChainFMA final : OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getKind() != vector::CombiningKind::ADD) {
      return failure();
    }

    auto lhsVecType = dyn_cast<VectorType>(op.getLhsType());
    auto rhsVecType = dyn_cast<VectorType>(op.getRhsType());
    if (!lhsVecType || !rhsVecType || !lhsVecType.hasStaticShape() ||
        !rhsVecType.hasStaticShape()) {
      return failure();
    }

    Type elemTy = lhsVecType.getElementType();
    if (!isa<Float32Type, Float16Type>(elemTy)) {
      return failure();
    }

    auto accVecType = dyn_cast<VectorType>(op.getAccType());
    if (accVecType && !accVecType.hasStaticShape()) {
      return failure();
    }

    SmallVector<int64_t> redDims, parDims;
    getReductionAndParallelLoopDims(op.getIteratorTypes(), redDims, parDims);
    if (redDims.empty()) {
      return failure();
    }

    // indices: [reduction..., parallel...].
    SmallVector<int64_t> indices;
    indices.append(redDims.begin(), redDims.end());
    indices.append(parDims.begin(), parDims.end());

    SmallVector<AffineMap, 4> maps = op.getIndexingMapsArray();
    SmallVector<int64_t> lhsPerm =
        getPermutationFromIndexingMap(maps[0], indices);
    SmallVector<int64_t> rhsPerm =
        getPermutationFromIndexingMap(maps[1], indices);
    SmallVector<int64_t> accPerm;
    if (accVecType) {
      accPerm = getPermutationFromIndexingMap(maps[2], parDims);
    }

    Location loc = op.getLoc();

    // Transpose operands to [red..., par...].
    Value lhs = op.getLhs();
    if (!isIdentityPermutation(lhsPerm)) {
      lhs =
          vector::TransposeOp::create(rewriter, loc, lhs, lhsPerm).getResult();
    }

    Value rhs = op.getRhs();
    if (!isIdentityPermutation(rhsPerm)) {
      rhs =
          vector::TransposeOp::create(rewriter, loc, rhs, rhsPerm).getResult();
    }

    // Shape-cast operands to 2D {reduction_size, parallel_size}.
    const unsigned numRed = redDims.size();
    VectorType vecType = cast<VectorType>(lhs.getType());
    int64_t redSize = productOfDims(vecType, 0, numRed);
    int64_t parSize = productOfDims(vecType, numRed, vecType.getRank());
    VectorType flattened2DType = VectorType::get({redSize, parSize}, elemTy);
    Value lhs2D =
        rewriter.create<vector::ShapeCastOp>(loc, flattened2DType, lhs);
    Value rhs2D =
        rewriter.create<vector::ShapeCastOp>(loc, flattened2DType, rhs);

    Value flattenedAcc;
    VectorType parVecType = VectorType::get({parSize}, elemTy);
    if (accVecType) {
      Value acc = op.getAcc();
      if (!isIdentityPermutation(accPerm)) {
        acc = vector::TransposeOp::create(rewriter, loc, acc, accPerm);
      }
      flattenedAcc = rewriter.create<vector::ShapeCastOp>(loc, parVecType, acc);
    } else {
      flattenedAcc =
          rewriter.create<vector::SplatOp>(loc, parVecType, op.getAcc());
    }

    const int64_t chunkSize = 2;
    Value resultFlat = buildFMAChain(rewriter, loc, lhs2D, rhs2D, flattenedAcc,
                                     redSize, parSize, chunkSize);

    // Restore result shape/types
    Value result;
    if (accVecType) {
      const bool needTranspose = !isIdentityPermutation(accPerm);
      VectorType accTransposedType =
          needTranspose ? permutedType(accVecType, accPerm) : accVecType;

      // Unflatten
      Value reshaped = rewriter.create<vector::ShapeCastOp>(
          loc, accTransposedType, resultFlat);

      result = needTranspose ? rewriter.create<vector::TransposeOp>(
                                   loc, accVecType, reshaped, invert(accPerm))
                             : reshaped;
    } else {
      result = rewriter.create<vector::ExtractOp>(loc, resultFlat,
                                                  ArrayRef<int64_t>{0});
    }

    rewriter.replaceOp(op, result);
    return success();
  }

private:
  static SmallVector<int64_t> invert(ArrayRef<int64_t> p) {
    SmallVector<int64_t> inv(p.size());
    for (int64_t i = 0, e = p.size(); i < e; ++i) {
      inv[p[i]] = i;
    }
    return inv;
  }

  static void getReductionAndParallelLoopDims(ArrayAttr iters,
                                              SmallVectorImpl<int64_t> &red,
                                              SmallVectorImpl<int64_t> &par) {
    for (auto [idx, attr] : llvm::enumerate(iters)) {
      if (vector::isReductionIterator(attr)) {
        red.push_back(idx);
      } else {
        par.push_back(idx);
      }
    }
  }

  /// Constructs a permutation for vector.transpose from an affine map and a
  /// reordered list of dimension indices ([red ..., par ...])
  ///
  /// Example:
  ///   map: (d0, d1, d2) -> (d0, d2, d1)
  ///   indices: [2, 0, 1]
  ///
  ///   Step 1: Build dim-to-result mapping from the map
  ///           dimToRes = [0, 2, 1]
  ///
  ///   Step 2: Walk new indices in order to build permutation
  ///           indices[0]=2 -> dimToRes[2]=1 -> perm[0]=1
  ///           indices[1]=0 -> dimToRes[0]=0 -> perm[1]=0
  ///           indices[2]=1 -> dimToRes[1]=2 -> perm[2]=2
  ///
  ///   Result: perm = [1, 0, 2]
  static SmallVector<int64_t>
  getPermutationFromIndexingMap(AffineMap map, ArrayRef<int64_t> indices) {
    SmallVector<int64_t> dimToRes(map.getNumDims(), -1);
    for (int res = 0; res < map.getNumResults(); ++res) {
      dimToRes[map.getDimPosition(res)] = res;
    }

    SmallVector<int64_t> perm;
    for (int64_t i : indices) {
      int64_t operandDim = dimToRes[i];
      if (operandDim >= 0) {
        perm.push_back(operandDim);
      }
    }

    return perm;
  }

  static int64_t productOfDims(VectorType vt, unsigned lo, unsigned hi) {
    int64_t p = 1;
    for (unsigned i = lo; i < hi; ++i) {
      p *= vt.getDimSize(i);
    }
    return p;
  }

  static bool isIdentityPermutation(ArrayRef<int64_t> p) {
    for (int64_t i = 0, e = p.size(); i < e; ++i) {
      if (p[i] != i) {
        return false;
      }
    }
    return true;
  }

  static VectorType permutedType(VectorType vt, ArrayRef<int64_t> perm) {
    SmallVector<int64_t> shape;
    for (int64_t i : perm) {
      shape.push_back(vt.getDimSize(i));
    }
    return VectorType::get(shape, vt.getElementType());
  }

  static Value processFMAChunk(PatternRewriter &rewriter, Location loc,
                               Value lhsRow, Value rhsRow, Value acc,
                               int64_t offset, int64_t chunkSize) {
    int64_t stride = 1;

    Value a = rewriter.create<vector::ExtractStridedSliceOp>(
        loc, lhsRow, offset, chunkSize, stride);
    Value b = rewriter.create<vector::ExtractStridedSliceOp>(
        loc, rhsRow, offset, chunkSize, stride);
    Value c = rewriter.create<vector::ExtractStridedSliceOp>(loc, acc, offset,
                                                             chunkSize, stride);

    Value fma = rewriter.create<math::FmaOp>(loc, a, b, c);

    return rewriter.create<vector::InsertStridedSliceOp>(loc, fma, acc, offset,
                                                         stride);
  }

  // Core: produce resultFlat := fold_k fma(lhs2D[k,*], rhs2D[k,*], accFlat)
  // using vector<2xf32> chunks, chaining from inner-most FMA (backwards).
  static Value buildFMAChain(PatternRewriter &rewriter, Location loc,
                             Value lhs2D, Value rhs2D, Value accFlat, int64_t K,
                             int64_t P, int64_t chunkSize) {
    Value current = accFlat;

    for (int64_t k = K - 1; k >= 0; --k) {
      Value lhsRow =
          rewriter.create<vector::ExtractOp>(loc, lhs2D, ArrayRef<int64_t>{k});
      Value rhsRow =
          rewriter.create<vector::ExtractOp>(loc, rhs2D, ArrayRef<int64_t>{k});

      // Process full chunks
      int64_t p = 0;
      for (; p + chunkSize <= P; p += chunkSize) {
        current = processFMAChunk(rewriter, loc, lhsRow, rhsRow, current, p,
                                  chunkSize);
      }

      // Process any remaining scalars
      int64_t tail = P - p;
      if (tail > 0) {
        current =
            processFMAChunk(rewriter, loc, lhsRow, rhsRow, current, p, tail);
      }
    }
    return current;
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

    {
      RewritePatternSet patterns(ctx);
      vector::populateVectorReductionToContractPatterns(patterns);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    {
      RewritePatternSet patterns(ctx);
      patterns.add<ContractAddToChainFMA>(ctx);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
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
      // vector::populateVectorContractLoweringPatterns(
          // contractLoweringPatterns, options.vectorContractLowering);
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
