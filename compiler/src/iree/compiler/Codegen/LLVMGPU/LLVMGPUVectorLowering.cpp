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
    : vector::MaskableOpRewritePattern<vector::ContractionOp> {
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

// Transposes the operands of a vector.contract so that LHS and RHS have
// [reduction..., parallel...] physical layout, and the acc (if vector) has
// parallel dims in the canonical order. Emits vector.broadcast for missing
// parallel dims on LHS/RHS. Produces a new vector.contract with identity-like
// indexing maps over the normalized iteration space
// [reduction..., parallel...].
//
// Example:
// ```mlir
// #map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// #map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
// vector.contract {
//    indexing_maps = [#map, #map, #map1],
//    iterator_types = ["parallel", "parallel", "reduction"],
//    kind = #vector.kind<add>
// } %arg0, %arg1, %cst : vector<2x1x8xf16>, vector<2x1x8xf16>
//                         into vector<2x1xf16>
// ```
//
// ==>
//
// ```mlir
// %lhs_t = vector.transpose %arg0, [2, 0, 1]
//     : vector<2x1x8xf16> to vector<8x2x1xf16>
// %rhs_t = vector.transpose %arg1, [2, 0, 1]
//     : vector<2x1x8xf16> to vector<8x2x1xf16>
// #new_lhs = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// #new_acc = affine_map<(d0, d1, d2) -> (d1, d2)>
// %result = vector.contract {
//    indexing_maps = [#new_lhs, #new_lhs, #new_acc],
//    iterator_types = ["reduction", "parallel", "parallel"],
//    kind = #vector.kind<add>
// } %lhs_t, %rhs_t, %cst : vector<8x2x1xf16>, vector<8x2x1xf16>
//                           into vector<2x1xf16>
// ```
struct TransposeContractOperands final
    : OpRewritePattern<vector::ContractionOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override {
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

    SmallVector<int64_t> redDims, parDims;
    getReductionAndParallelLoopDims(op.getIteratorTypes(), redDims, parDims);
    if (redDims.empty()) {
      return failure();
    }

    SmallVector<AffineMap, 4> maps = op.getIndexingMapsArray();
    AffineMap lhsMap = maps[0];
    AffineMap rhsMap = maps[1];
    AffineMap accMap = maps[2];

    unsigned numRedDims = redDims.size();
    unsigned numParDims = parDims.size();
    unsigned numDims = numRedDims + numParDims;
    MLIRContext *ctx = op.getContext();

    // Build the target identity-like maps for the new iteration order
    // [reduction..., parallel...].
    AffineMap newLhsMap = AffineMap::getMultiDimIdentityMap(numDims, ctx);
    AffineMap newRhsMap = newLhsMap;
    SmallVector<AffineExpr> accExprs;
    for (unsigned i = 0; i < numParDims; ++i) {
      accExprs.push_back(getAffineDimExpr(numRedDims + i, ctx));
    }
    AffineMap newAccMap = AffineMap::get(numDims, 0, accExprs, ctx);

    SmallVector<Attribute> newIterTypes;
    for (unsigned i = 0; i < numRedDims; ++i) {
      newIterTypes.push_back(
          vector::IteratorTypeAttr::get(ctx, vector::IteratorType::reduction));
    }
    for (unsigned i = 0; i < numParDims; ++i) {
      newIterTypes.push_back(
          vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel));
    }
    auto newIterTypesAttr = ArrayAttr::get(ctx, newIterTypes);

    // Bail out if already in normalized form.
    if (maps[0] == newLhsMap && maps[1] == newRhsMap && maps[2] == newAccMap &&
        op.getIteratorTypes() == newIterTypesAttr) {
      return failure();
    }

    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    // Promote operands to the accumulator's element type *before* broadcast
    // and transpose, matching the emission order of the old monolithic
    // ContractToChainFMA. This order matters because translateModuleToLLVMIR
    // is sensitive to use-list ordering in the in-memory IR.
    auto accElemType = getElementTypeOrSelf(op.getAccType());
    if (isa<FloatType>(lhsVecType.getElementType()) &&
        lhsVecType.getElementType() != accElemType) {
      Type promotedType = lhsVecType.clone(accElemType);
      lhs = arith::ExtFOp::create(rewriter, loc, promotedType, lhs);
      lhsVecType = cast<VectorType>(lhs.getType());
    }
    if (isa<FloatType>(rhsVecType.getElementType()) &&
        rhsVecType.getElementType() != accElemType) {
      Type promotedType = rhsVecType.clone(accElemType);
      rhs = arith::ExtFOp::create(rewriter, loc, promotedType, rhs);
      rhsVecType = cast<VectorType>(rhs.getType());
    }

    ArrayRef<int64_t> lhsShape = lhsVecType.getShape();
    ArrayRef<int64_t> rhsShape = rhsVecType.getShape();

    SmallVector<int64_t> lhsTranspose, rhsTranspose;
    lhs = broadcastMissingDims(rewriter, loc, lhsMap, accMap,
                               op.getIteratorTypes(), numParDims, resultVecType,
                               lhs, lhsShape, accElemType, lhsTranspose);
    rhs = broadcastMissingDims(rewriter, loc, rhsMap, accMap,
                               op.getIteratorTypes(), numParDims, resultVecType,
                               rhs, rhsShape, accElemType, rhsTranspose);

    lhs = vector::TransposeOp::create(rewriter, loc, lhs, lhsTranspose);
    rhs = vector::TransposeOp::create(rewriter, loc, rhs, rhsTranspose);

    // Transpose acc to match the canonical parallel dim order if needed.
    Value acc = op.getAcc();
    SmallVector<int64_t> accPerm;
    bool accTransposed = false;
    if (maybeAccVecType) {
      accPerm = getPermutationFromIndexingMap(accMap, parDims);
      if (!isIdentityPermutation(accPerm)) {
        acc = vector::TransposeOp::create(rewriter, loc, acc, accPerm);
        accTransposed = true;
      }
    }

    auto newContract = vector::ContractionOp::create(
        rewriter, loc, lhs, rhs, acc,
        rewriter.getAffineMapArrayAttr({newLhsMap, newRhsMap, newAccMap}),
        newIterTypesAttr, op.getKind());

    // Undo the acc transpose on the result if we applied one.
    Value result = newContract.getResult();
    if (accTransposed) {
      result = vector::TransposeOp::create(rewriter, loc, resultVecType, result,
                                           invert(accPerm));
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

  static SmallVector<int64_t>
  getPermutationFromIndexingMap(AffineMap map, ArrayRef<int64_t> indices) {
    SmallVector<int64_t> dimToRes(map.getNumDims());
    for (int res = 0, e = map.getNumResults(); res != e; ++res) {
      dimToRes[map.getDimPosition(res)] = res;
    }

    return map_to_vector(indices, [&](int64_t i) { return dimToRes[i]; });
  }

  static bool isIdentityPermutation(ArrayRef<int64_t> perm) {
    return llvm::all_of(llvm::enumerate(perm),
                        [](auto p) { return p.value() == p.index(); });
  }
};

// Rewrites vector.contracts that are already in normalized
// [reduction..., parallel...] form into a chain of math.fma ops.
// Expects LHS/RHS to have identity indexing maps and iterator types ordered
// as [reduction..., parallel...]. The TransposeContractOperands pattern
// should run first to normalize contracts into this form.
//
// Starting from the innermost position of the reduction dimension,
// the lowering emits a single nested FMA chain as follows:
// fma(a0 ,b0, fma(a1, b1, fma(a2, b2, fma(a3, b3, acc))))
// where ai and bi are the elements extracted from lhs and rhs vectors
// respectively along the reduction dimension.
//
// Example (after TransposeContractOperands has run):
// ```mlir
// #map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// #map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
// vector.contract {
//    indexing_maps = [#map, #map, #map1],
//    iterator_types = ["reduction", "parallel", "parallel"],
//    kind = #vector.kind<add>
// } %lhs, %rhs, %cst : vector<8x2x1xf16>, vector<8x2x1xf16>
//                       into vector<2x1xf16>
// ```
//
// ==>
// ```mlir
// %34 = math.fma %32, %33, %cst : vector<2xf16>
// %37 = math.fma %35, %36, %34 : vector<2xf16>
// ...
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

    // Check that the contract is in normalized [reduction..., parallel...] form
    // with identity maps on LHS/RHS.
    ArrayAttr iteratorTypes = op.getIteratorTypes();
    unsigned numDims = iteratorTypes.size();
    unsigned numRedDims = 0;
    for (unsigned i = 0; i < numDims; ++i) {
      if (vector::isReductionIterator(iteratorTypes[i])) {
        if (i != numRedDims) {
          return failure();
        }
        ++numRedDims;
      }
    }
    if (numRedDims == 0) {
      return failure();
    }

    SmallVector<AffineMap, 4> maps = op.getIndexingMapsArray();
    AffineMap identityMap =
        AffineMap::getMultiDimIdentityMap(numDims, op.getContext());
    if (maps[0] != identityMap || maps[1] != identityMap) {
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

    int64_t redSize = productOfDims(lhsVecType, 0, numRedDims);
    int64_t parSize =
        productOfDims(lhsVecType, numRedDims, lhsVecType.getRank());

    auto flattened2DType = VectorType::get({redSize, parSize}, elemType);
    Value lhs2D =
        vector::ShapeCastOp::create(rewriter, loc, flattened2DType, lhs);
    Value rhs2D =
        vector::ShapeCastOp::create(rewriter, loc, flattened2DType, rhs);

    Value flattenedAcc;
    auto flatAccVecType = VectorType::get({parSize}, elemType);

    if (maybeAccVecType) {
      flattenedAcc = vector::ShapeCastOp::create(rewriter, loc, flatAccVecType,
                                                 op.getAcc());
    } else {
      flattenedAcc = vector::BroadcastOp::create(rewriter, loc, flatAccVecType,
                                                 op.getAcc());
    }

    Value resultFlat =
        buildFMAChain(rewriter, loc, lhs2D, rhs2D, flattenedAcc, redSize);

    Value result;
    if (maybeAccVecType) {
      result = vector::ShapeCastOp::create(rewriter, loc, maybeAccVecType,
                                           resultFlat);
    } else {
      result = vector::ExtractOp::create(rewriter, loc, resultFlat, 0);
    }

    rewriter.replaceOp(op, result);
    return success();
  }

private:
  static int64_t productOfDims(VectorType vt, unsigned lo, unsigned hi) {
    int64_t p = 1;
    for (unsigned i = lo; i < hi; ++i) {
      p *= vt.getDimSize(i);
    }
    return p;
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
      contractLoweringPatterns.add<TransposeContractOperands>(
          funcOp->getContext(), PatternBenefit(2));
      contractLoweringPatterns.add<ContractToChainFMA>(funcOp->getContext(),
                                                       PatternBenefit(2));
      vector::populateVectorGatherLoweringPatterns(contractLoweringPatterns);
      vector::populateVectorMaskOpLoweringPatterns(contractLoweringPatterns);
      vector::populateVectorShapeCastLoweringPatterns(contractLoweringPatterns);
      vector::populateVectorMultiReductionReorderPatterns(
          contractLoweringPatterns,
          vector::VectorMultiReductionLowering::InnerReduction);
      vector::populateVectorMultiReductionFlatteningPatterns(
          contractLoweringPatterns,
          vector::VectorMultiReductionLowering::InnerReduction);
      vector::populateVectorMultiReductionUnrollingPatterns(
          contractLoweringPatterns,
          vector::VectorMultiReductionLowering::InnerReduction);
      // Unroll transfer_gather ops to rank 1 and lower contiguous ones to
      // vector.transfer_read.
      IREE::VectorExt::populateVectorTransferGatherScatterLoweringPatterns(
          contractLoweringPatterns);
      IREE::VectorExt::TransferGatherOp::getCanonicalizationPatterns(
          contractLoweringPatterns, ctx);
      IREE::VectorExt::TransferScatterOp::getCanonicalizationPatterns(
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
  }
};
} // namespace
} // namespace mlir::iree_compiler
