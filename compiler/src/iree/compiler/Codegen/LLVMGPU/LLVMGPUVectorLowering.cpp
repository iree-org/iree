// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/AMDGPU/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
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

    auto replacement = vector::ContractionOp::create(
        rewriter, loc, lhs, rhs, contractOp.getAcc(),
        contractOp.getIndexingMaps(), contractOp.getIteratorTypes());

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
      return arith::ExtFOp::create(rewriter, loc, promotedType, v);
    // For integer types, vector.contract only supports signless integer types
    // and promotion happens via sign extension.
    return arith::ExtSIOp::create(rewriter, loc, promotedType, v);
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

struct ContractToChainFMA final : OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: Add a rewrite to support relevant contractions nested in
    // vector.mask
    if (op.isMasked()) {
      return failure();
    }

    if (op.getKind() != vector::CombiningKind::ADD) {
      return failure();
    }

    VectorType lhsVecType = op.getLhsType();
    VectorType rhsVecType = op.getRhsType();
    if (lhsVecType.isScalable() || rhsVecType.isScalable()) {
      return failure();
    }

    Type elemTy = lhsVecType.getElementType();
    if (!isa<Float32Type, Float16Type>(elemTy)) {
      return failure();
    }

    auto accVecType = dyn_cast<VectorType>(op.getAccType());
    if (accVecType && accVecType.isScalable()) {
      return failure();
    }

    SmallVector<int64_t> redDims, parDims;
    getReductionAndParallelLoopDims(op.getIteratorTypes(), redDims, parDims);
    if (redDims.empty()) {
      return failure();
    }

    // New indices: [reduction..., parallel...].
    SmallVector<int64_t> indices;
    indices.reserve(redDims.size() + parDims.size());
    llvm::append_range(indices, redDims);
    llvm::append_range(indices, parDims);

    SmallVector<AffineMap, 4> maps = op.getIndexingMapsArray();

    // We only lower contracts where both LHS and RHS carry the same set of
    // parallel iterators. Order may differ, but no parallel dim
    // may be dropped on either side. This excludes matmul-like cases and any
    // contract where parallel sizes would differ between operands.
    if (!verifyParallelDimsInMap(parDims, maps[0]) ||
        !verifyParallelDimsInMap(parDims, maps[1])) {
      return failure();
    }

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
      lhs = vector::TransposeOp::create(rewriter, loc, lhs, lhsPerm);
    }

    Value rhs = op.getRhs();
    if (!isIdentityPermutation(rhsPerm)) {
      rhs = vector::TransposeOp::create(rewriter, loc, rhs, rhsPerm);
    }

    const size_t numRed = redDims.size();
    auto lhsTransposedVecType = cast<VectorType>(lhs.getType());
    int64_t lhsRedSize = productOfDims(lhsTransposedVecType, 0, numRed);
    int64_t lhsParSize = productOfDims(lhsTransposedVecType, numRed,
                                       lhsTransposedVecType.getRank());

    // Shape-cast operands to 2D {reduction_size, parallel_size}.
    int64_t redSize = lhsRedSize;
    int64_t parSize = lhsParSize;
    VectorType flattened2DType = VectorType::get({redSize, parSize}, elemTy);
    Value lhs2D =
        vector::ShapeCastOp::create(rewriter, loc, flattened2DType, lhs);
    Value rhs2D =
        vector::ShapeCastOp::create(rewriter, loc, flattened2DType, rhs);

    Value flattenedAcc;
    auto flatAccVecType = VectorType::get({parSize}, elemTy);
    VectorType preFlattenVecType = accVecType;

    if (accVecType) {
      Value acc = op.getAcc();

      if (!isIdentityPermutation(accPerm)) {
        acc = vector::TransposeOp::create(rewriter, loc, acc, accPerm);
        preFlattenVecType = cast<VectorType>(acc.getType());
      }

      flattenedAcc =
          vector::ShapeCastOp::create(rewriter, loc, flatAccVecType, acc);
    } else {
      flattenedAcc = vector::BroadcastOp::create(rewriter, loc, flatAccVecType,
                                                 op.getAcc());
    }

    // Form chain in 2-element chunks that we can map to one packed FMA.
    constexpr int64_t chunkSize = 2;
    Value resultFlat = buildFMAChain(rewriter, loc, lhs2D, rhs2D, flattenedAcc,
                                     redSize, parSize, chunkSize);

    // Restore result to original form.
    Value result;
    if (accVecType) {
      Value reshaped = vector::ShapeCastOp::create(
          rewriter, loc, preFlattenVecType, resultFlat);

      if (!isIdentityPermutation(accPerm)) {
        result = vector::TransposeOp::create(rewriter, loc, accVecType,
                                             reshaped, invert(accPerm));
      } else {
        result = reshaped;
      }

    } else {
      result = vector::ExtractOp::create(rewriter, loc, resultFlat, 0);
    }

    rewriter.replaceOp(op, result);
    return success();
  }

private:
  static bool verifyParallelDimsInMap(ArrayRef<int64_t> parallelDims,
                                      AffineMap map) {
    llvm::SmallSetVector<int64_t, 8> usedDims;
    map.walkExprs([&](AffineExpr expr) {
      if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
        usedDims.insert(dimExpr.getPosition());
      }
    });

    return llvm::all_of(parallelDims, [&](int64_t parDim) {
      return usedDims.contains(parDim);
    });
  }

  static SmallVector<int64_t> invert(ArrayRef<int64_t> perm) {
    SmallVector<int64_t> inv(perm.size());
    for (int64_t i = 0; i < perm.size(); ++i) {
      inv[perm[i]] = i;
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
  /// reordered list of dimension.
  ///
  /// Example:
  ///   map: (d0, d1, d2) -> (d0, d2, d1)
  ///   iterator_types = ["parallel","parallel","reduction"]
  //    ==> new dim order: [2, 0, 1]
  ///
  ///   Step 1: Build dim-to-result mapping from the map.
  ///           dimToRes = [0, 2, 1] i.e {0: 0, 1: 2, 2: 1}
  ///
  ///   Step 2: Walk new dimension order in order to build permutation.
  ///           indices[0]=2 -> dimToRes[2]=1
  ///           indices[1]=0 -> dimToRes[0]=0
  ///           indices[2]=1 -> dimToRes[1]=2
  ///
  ///   Result: perm = [1, 0, 2]
  static SmallVector<int64_t>
  getPermutationFromIndexingMap(AffineMap map, ArrayRef<int64_t> indices) {
    SmallVector<int64_t> dimToRes(map.getNumDims());
    for (int res = 0, e = map.getNumResults(); res != e; ++res) {
      dimToRes[map.getDimPosition(res)] = res;
    }

    return to_vector(
        llvm::map_range(indices, [&](int64_t i) { return dimToRes[i]; }));
  }

  static int64_t productOfDims(VectorType vt, unsigned lo, unsigned hi) {
    int64_t p = 1;
    for (unsigned i = lo; i < hi; ++i) {
      p *= vt.getDimSize(i);
    }
    return p;
  }

  static bool isIdentityPermutation(ArrayRef<int64_t> perm) {
    return llvm::all_of(llvm::enumerate(perm),
                        [](auto p) { return p.value() == p.index(); });
  }

  static Value processFMAChunk(PatternRewriter &rewriter, Location loc,
                               Value lhsRow, Value rhsRow, Value acc,
                               int64_t offset, int64_t chunkSize) {
    int64_t stride = 1;

    Value a = vector::ExtractStridedSliceOp::create(rewriter, loc, lhsRow,
                                                    offset, chunkSize, stride);
    Value b = vector::ExtractStridedSliceOp::create(rewriter, loc, rhsRow,
                                                    offset, chunkSize, stride);
    Value c = vector::ExtractStridedSliceOp::create(rewriter, loc, acc, offset,
                                                    chunkSize, stride);

    Value fma = math::FmaOp::create(rewriter, loc, a, b, c);

    return vector::InsertStridedSliceOp::create(rewriter, loc, fma, acc, offset,
                                                stride);
  }

  static Value buildFMAChain(PatternRewriter &rewriter, Location loc,
                             Value lhs2D, Value rhs2D, Value accFlat, int64_t K,
                             int64_t P, int64_t chunkSize) {
    Value current = accFlat;

    for (int64_t k = K - 1; k >= 0; --k) {
      Value lhsRow = vector::ExtractOp::create(rewriter, loc, lhs2D, k);
      Value rhsRow = vector::ExtractOp::create(rewriter, loc, rhs2D, k);

      // Process full chunks.
      int64_t p = 0;
      for (; p + chunkSize <= P; p += chunkSize) {
        current = processFMAChunk(rewriter, loc, lhsRow, rhsRow, current, p,
                                  chunkSize);
      }

      // Process any remaining scalars.
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
      // Lower high level vector operations like contract or multidim reduce ops
      // to lower level vector ops.
      RewritePatternSet contractLoweringPatterns(funcOp.getContext());
      auto options =
          vector::VectorTransformsOptions().setVectorTransformsOptions(
              vector::VectorContractLowering::OuterProduct);
      vector::populateVectorTransferPermutationMapLoweringPatterns(
          contractLoweringPatterns);
      vector::TransposeOp::getCanonicalizationPatterns(contractLoweringPatterns,
                                                       funcOp.getContext());
      vector::populateVectorBroadcastLoweringPatterns(contractLoweringPatterns);
      vector::populateVectorContractLoweringPatterns(
          contractLoweringPatterns, options.vectorContractLowering);
      contractLoweringPatterns.add<PromoteContractOperands>(
          funcOp->getContext());
      contractLoweringPatterns.add<ContractToChainFMA>(funcOp->getContext(),
                                                       PatternBenefit(2));
      vector::populateVectorGatherLoweringPatterns(contractLoweringPatterns);
      vector::populateVectorMaskOpLoweringPatterns(contractLoweringPatterns);
      vector::populateVectorShapeCastLoweringPatterns(contractLoweringPatterns);
      vector::populateVectorMultiReductionLoweringPatterns(
          contractLoweringPatterns,
          vector::VectorMultiReductionLowering::InnerReduction);
      if (failed(applyPatternsGreedily(funcOp,
                                       std::move(contractLoweringPatterns)))) {
        return signalPassFailure();
      }
    }

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
