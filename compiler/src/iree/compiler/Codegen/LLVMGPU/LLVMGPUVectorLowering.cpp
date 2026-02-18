// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Transforms.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/AMDGPU/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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
    if (elementType == dstElementType) {
      return v;
    }

    // vector.contract only allows extension on operands.
    assert(elementType.getIntOrFloatBitWidth() <=
               dstElementType.getIntOrFloatBitWidth() &&
           "vector.contract does not allow truncation of operands");

    Type promotedType = dstElementType;
    if (auto vecType = dyn_cast<VectorType>(v.getType())) {
      promotedType = vecType.clone(promotedType);
    }

    if (isa<FloatType>(dstElementType)) {
      return arith::ExtFOp::create(rewriter, loc, promotedType, v);
    }
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
  using Base::Base;

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
  using Base::Base;

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

// Rewrites vector.contracts into a chain of math.fma ops when possible.
// Starting from the innermost position of the reduction dimension,
// the lowering emits a single nested FMA chain as follows:
// fma(a0 ,b0, fma(a1, b1, fma(a2, b2, fma(a3, b3, acc))))
// where ai and bi are the elements extracted from lhs and rhs vectors
// respectively along the reduction dimension.
//
// Example:
// ```mlir
// #map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// #map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
// vector.contract
//{
//    indexing_maps = [#map, #map, #map1],
//    iterator_types = ["parallel", "parallel", "reduction"],
//    kind = #vector.kind<add>
// }
// %arg0, %arg1, %cst : vector<2x1x8xf16>, vector<2x1x8xf16> into
// vector<2x1xf16>
// ```
//
// ==>
// <Extract lhs/rhs along reduction dim> then:
// ```mlir
// %34 = math.fma %32, %33, %cst : vector<2xf16>
// %37 = math.fma %35, %36, %34 : vector<2xf16>
// %40 = math.fma %38, %39, %37 : vector<2xf16>
// %43 = math.fma %41, %42, %40 : vector<2xf16>
// %45 = math.fma %44, %45, %43 : vector<2xf16>
// %49 = math.fma %47, %48, %46 : vector<2xf16>
// %52 = math.fma %50, %51, %49 : vector<2xf16>
// %55 = math.fma %53, %54, %52 : vector<2xf16>
// ```
//
// Previously, contracts of the same form lowered to elementwise multiplies
// followed by a vector.reduce. This lowering elides the need to reduce the
// result of the elementwise operations separately and instead accumulates
// directly result via FMAs, offering more profitable instruction level
// scheduling on GPUs.
struct ContractToChainFMA final : OpRewritePattern<vector::ContractionOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: Add a rewrite to support relevant contractions nested in
    // vector.mask.
    if (op.isMasked() || op.getKind() != vector::CombiningKind::ADD) {
      return failure();
    }

    VectorType lhsVecType = op.getLhsType();
    VectorType rhsVecType = op.getRhsType();
    if (lhsVecType.isScalable() || rhsVecType.isScalable()) {
      return failure();
    }

    auto resultVecType = dyn_cast<VectorType>(op.getResultType());
    if (!resultVecType || resultVecType.isScalable()) {
      return failure();
    }

    auto maybeAccVecType = dyn_cast<VectorType>(op.getAccType());
    if (maybeAccVecType && maybeAccVecType.isScalable()) {
      return failure();
    }

    if (!isa<FloatType>(lhsVecType.getElementType())) {
      return failure();
    }

    SmallVector<int64_t> redDims, parDims;
    getReductionAndParallelLoopDims(op.getIteratorTypes(), redDims, parDims);
    if (redDims.empty()) {
      return failure();
    }

    auto elemType = getElementTypeOrSelf(op.getAccType());

    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    if (lhsVecType.getElementType() != elemType) {
      Type promotedType = lhsVecType.clone(elemType);
      lhs = arith::ExtFOp::create(rewriter, loc, promotedType, lhs);
      lhsVecType = cast<VectorType>(lhs.getType());
    }

    if (rhsVecType.getElementType() != elemType) {
      Type promotedType = rhsVecType.clone(elemType);
      rhs = arith::ExtFOp::create(rewriter, loc, promotedType, rhs);
      rhsVecType = cast<VectorType>(rhs.getType());
    }

    // New indices: [reduction..., parallel...].
    auto indices = llvm::to_vector(llvm::concat<int64_t>(redDims, parDims));

    ArrayRef<int64_t> lhsShape = lhsVecType.getShape();
    ArrayRef<int64_t> rhsShape = rhsVecType.getShape();
    SmallVector<AffineMap, 4> maps = op.getIndexingMapsArray();
    AffineMap lhsMap = maps[0];
    AffineMap rhsMap = maps[1];
    AffineMap accMap = maps[2];

    // Broadcast operands for missing parallel dimensions.
    unsigned numParallelDims = accMap.getNumResults();

    SmallVector<int64_t> lhsTranspose, rhsTranspose;
    lhs = broadcastMissingDims(
        rewriter, loc, lhsMap, accMap, op.getIteratorTypes(), numParallelDims,
        resultVecType, lhs, lhsShape, elemType, lhsTranspose);
    rhs = broadcastMissingDims(
        rewriter, loc, rhsMap, accMap, op.getIteratorTypes(), numParallelDims,
        resultVecType, rhs, rhsShape, elemType, rhsTranspose);

    // Apply transposes to get [reduction..., parallel...] layout.
    lhs = vector::TransposeOp::create(rewriter, loc, lhs, lhsTranspose);
    rhs = vector::TransposeOp::create(rewriter, loc, rhs, rhsTranspose);

    SmallVector<int64_t> accPerm;
    if (maybeAccVecType) {
      accPerm = getPermutationFromIndexingMap(maps[2], parDims);
    }

    const size_t numRed = redDims.size();
    auto lhsTransposedVecType = cast<VectorType>(lhs.getType());
    int64_t lhsRedSize = productOfDims(lhsTransposedVecType, 0, numRed);
    int64_t lhsParSize = productOfDims(lhsTransposedVecType, numRed,
                                       lhsTransposedVecType.getRank());

    // Shape-cast operands to 2D {reduction_size, parallel_size}.
    int64_t redSize = lhsRedSize;
    int64_t parSize = lhsParSize;
    auto flattened2DType = VectorType::get({redSize, parSize}, elemType);
    Value lhs2D =
        vector::ShapeCastOp::create(rewriter, loc, flattened2DType, lhs);
    Value rhs2D =
        vector::ShapeCastOp::create(rewriter, loc, flattened2DType, rhs);

    Value flattenedAcc;
    auto flatAccVecType = VectorType::get({parSize}, elemType);
    VectorType preFlattenVecType = maybeAccVecType;

    if (maybeAccVecType) {
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

    Value resultFlat =
        buildFMAChain(rewriter, loc, lhs2D, rhs2D, flattenedAcc, redSize);

    // Restore result to original form.
    Value result;
    if (maybeAccVecType) {
      Value reshaped = vector::ShapeCastOp::create(
          rewriter, loc, preFlattenVecType, resultFlat);

      if (!isIdentityPermutation(accPerm)) {
        result = vector::TransposeOp::create(rewriter, loc, maybeAccVecType,
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
  static Value broadcastMissingDims(
      PatternRewriter &rewriter, Location loc, AffineMap operandMap,
      AffineMap accMap, ArrayAttr iteratorTypes, unsigned numParallelDims,
      VectorType resultType, Value operand, ArrayRef<int64_t> operandShape,
      Type elemType, SmallVectorImpl<int64_t> &transpose) {
    SmallVector<int64_t> reductionDims =
        getReductionIndex(operandMap, iteratorTypes);

    unsigned numDimToBroadcast =
        numParallelDims - (operandMap.getNumResults() - reductionDims.size());

    SmallVector<int64_t> broadcastDims;

    for (int64_t dim : reductionDims) {
      transpose.push_back(numDimToBroadcast + dim);
    }

    for (unsigned i = 0; i < numParallelDims; ++i) {
      unsigned iterDim = accMap.getDimPosition(i);

      std::optional<unsigned> opDim = getDimPosition(operandMap, iterDim);
      if (opDim) {
        transpose.push_back(numDimToBroadcast + *opDim);
      } else {
        broadcastDims.push_back(resultType.getDimSize(i));
        transpose.push_back(broadcastDims.size() - 1);
      }
    }

    Value result = operand;
    if (!broadcastDims.empty()) {
      llvm::append_range(broadcastDims, operandShape);
      auto expandedType = VectorType::get(broadcastDims, elemType);
      result = vector::BroadcastOp::create(rewriter, loc, expandedType, result);
    }

    return result;
  }

  static std::optional<unsigned> getDimPosition(AffineMap map, unsigned dim) {
    for (unsigned i = 0, e = map.getNumResults(); i < e; i++) {
      if (map.getDimPosition(i) == dim) {
        return i;
      }
    }
    return std::nullopt;
  }

  static SmallVector<int64_t> getReductionIndex(AffineMap map,
                                                ArrayAttr iteratorTypes) {
    SmallVector<int64_t> dimsIdx;
    for (unsigned i = 0, e = map.getNumResults(); i < e; i++) {
      if (vector::isReductionIterator(iteratorTypes[map.getDimPosition(i)])) {
        dimsIdx.push_back(i);
      }
    }
    return dimsIdx;
  }

  static SmallVector<int64_t> invert(ArrayRef<int64_t> perm) {
    SmallVector<int64_t> inv(perm.size());
    for (auto [i, p] : llvm::enumerate(perm)) {
      inv[p] = i;
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

    return map_to_vector(indices, [&](int64_t i) { return dimToRes[i]; });
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

  static Value buildFMAChain(PatternRewriter &rewriter, Location loc,
                             Value lhs2D, Value rhs2D, Value accFlat,
                             int64_t K) {
    Value current = accFlat;

    for (int64_t k = K - 1; k >= 0; --k) {
      Value a = vector::ExtractOp::create(rewriter, loc, lhs2D, k);
      Value b = vector::ExtractOp::create(rewriter, loc, rhs2D, k);
      current = math::FmaOp::create(rewriter, loc, a, b, current);
    }
    return current;
  }
};

struct UnrollElementwiseOps final : public RewritePattern {
  UnrollElementwiseOps(MLIRContext *context, PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!OpTrait::hasElementwiseMappableTraits(op) ||
        op->getNumResults() != 1) {
      return failure();
    }

    Location loc = op->getLoc();
    VectorType dstVecTy = dyn_cast<VectorType>(op->getResult(0).getType());
    if (!dstVecTy || dstVecTy.getRank() <= 1) {
      return failure();
    }
    ArrayRef<int64_t> originalSize = dstVecTy.getShape();

    Value result = ub::PoisonOp::create(rewriter, loc, dstVecTy);
    auto subVecTy =
        VectorType::get({originalSize.back()}, dstVecTy.getElementType());

    SmallVector<int64_t> tileShape(dstVecTy.getRank() - 1, 1);
    for (SmallVector<int64_t> offsets :
         StaticTileOffsetRange(originalSize.drop_back(), tileShape)) {
      // Extract from each operand.
      SmallVector<Value> operands;
      for (Value val : op->getOperands()) {
        // Extract subvector if the operand is a vector. This is to handle
        // things like arith.select which take a scalar conditional but are
        // otherwise elementwise.
        if (isa<VectorType>(val.getType())) {
          val = vector::ExtractOp::create(rewriter, loc, val, offsets);
        }
        operands.push_back(val);
      }

      Operation *clonedOp = clone(rewriter, op, subVecTy, operands);
      Value subResult = clonedOp->getResult(0);
      result =
          vector::InsertOp::create(rewriter, loc, subResult, result, offsets);
    }

    rewriter.replaceOp(op, result);
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
    mlir::FunctionOpInterface funcOp = getOperation();
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
      vector::populateVectorMultiReductionReorderAndExpandPatterns(
          contractLoweringPatterns,
          vector::VectorMultiReductionLowering::InnerReduction);
      vector::populateVectorMultiReductionFlatteningPatterns(
          contractLoweringPatterns,
          vector::VectorMultiReductionLowering::InnerReduction);
      vector::populateVectorMultiReductionUnrollingPatterns(
          contractLoweringPatterns,
          vector::VectorMultiReductionLowering::InnerReduction);
      contractLoweringPatterns.add<UnrollElementwiseOps>(funcOp->getContext());
      // Unroll transfer_gather ops to rank 1 and lower contiguous ones to
      // vector.transfer_read.
      IREE::VectorExt::populateVectorTransferGatherLoweringPatterns(
          contractLoweringPatterns);
      IREE::VectorExt::TransferGatherOp::getCanonicalizationPatterns(
          contractLoweringPatterns, ctx);
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
