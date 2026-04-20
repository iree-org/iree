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
    if (op.isMasked()) {
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

    ArrayRef<int64_t> lhsShape = lhsVecType.getShape();
    ArrayRef<int64_t> rhsShape = rhsVecType.getShape();

    // Broadcast operands for missing parallel dimensions, then transpose to
    // [reduction..., parallel...] layout. Each operand uses its own element
    // type for the broadcast (important for mixed-precision contracts).
    SmallVector<int64_t> lhsTranspose, rhsTranspose;
    lhs = broadcastMissingDims(rewriter, loc, lhsMap, accMap,
                               op.getIteratorTypes(), numParDims, resultVecType,
                               lhs, lhsShape, lhsVecType.getElementType(),
                               lhsTranspose);
    rhs = broadcastMissingDims(rewriter, loc, rhsMap, accMap,
                               op.getIteratorTypes(), numParDims, resultVecType,
                               rhs, rhsShape, rhsVecType.getElementType(),
                               rhsTranspose);

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

// Flattens a vector.contract in normalized [reduction..., parallel...] form
// (with identity maps on LHS/RHS) to a 2D contract: one reduction dim and
// one parallel dim. Produces shape_casts around a new vector.contract.
//
// Matches contracts where numDims > 2 and the layout is already normalized
// (identity maps, [reduction..., parallel...] iterator order). Bails out
// if the contract is already 2D or not in the expected form.
//
// Example:
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
//
// ```mlir
// %lhs_flat = vector.shape_cast %lhs
//     : vector<8x2x1xf16> to vector<8x2xf16>
// %rhs_flat = vector.shape_cast %rhs
//     : vector<8x2x1xf16> to vector<8x2xf16>
// %acc_flat = vector.shape_cast %cst
//     : vector<2x1xf16> to vector<2xf16>
// #map2 = affine_map<(d0, d1) -> (d0, d1)>
// #map3 = affine_map<(d0, d1) -> (d1)>
// %result_flat = vector.contract {
//    indexing_maps = [#map2, #map2, #map3],
//    iterator_types = ["reduction", "parallel"],
//    kind = #vector.kind<add>
// } %lhs_flat, %rhs_flat, %acc_flat
//     : vector<8x2xf16>, vector<8x2xf16> into vector<2xf16>
// %result = vector.shape_cast %result_flat
//     : vector<2xf16> to vector<2x1xf16>
// ```
struct FlattenContractOperands final : OpRewritePattern<vector::ContractionOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override {
    if (op.isMasked()) {
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
    unsigned numParDims = numDims - numRedDims;

    // Already 2D (one reduction + one parallel) — nothing to flatten.
    if (numRedDims <= 1 && numParDims <= 1) {
      return failure();
    }

    SmallVector<AffineMap, 4> maps = op.getIndexingMapsArray();
    AffineMap identityMap =
        AffineMap::getMultiDimIdentityMap(numDims, op.getContext());
    if (maps[0] != identityMap || maps[1] != identityMap) {
      return failure();
    }

    Location loc = op.getLoc();
    MLIRContext *ctx = op.getContext();

    int64_t redSize = productOfDims(lhsVecType, 0, numRedDims);
    int64_t parSize = productOfDims(lhsVecType, numRedDims, numDims);

    // Shape-cast LHS/RHS to 2D {redSize, parSize}.
    auto lhsElemType = lhsVecType.getElementType();
    auto rhsElemType = rhsVecType.getElementType();
    auto flat2DLhsType = VectorType::get({redSize, parSize}, lhsElemType);
    auto flat2DRhsType = VectorType::get({redSize, parSize}, rhsElemType);
    Value lhsFlat =
        vector::ShapeCastOp::create(rewriter, loc, flat2DLhsType, op.getLhs());
    Value rhsFlat =
        vector::ShapeCastOp::create(rewriter, loc, flat2DRhsType, op.getRhs());

    // Shape-cast or broadcast acc to 1D {parSize}.
    auto accElemType = getElementTypeOrSelf(op.getAccType());
    auto flatAccType = VectorType::get({parSize}, accElemType);
    Value accFlat;
    if (maybeAccVecType) {
      accFlat =
          vector::ShapeCastOp::create(rewriter, loc, flatAccType, op.getAcc());
    } else {
      accFlat =
          vector::BroadcastOp::create(rewriter, loc, flatAccType, op.getAcc());
    }

    // Build 2D contract: (d0=reduction, d1=parallel).
    AffineMap newLhsMap = AffineMap::getMultiDimIdentityMap(2, ctx);
    AffineMap newRhsMap = newLhsMap;
    AffineMap newAccMap = AffineMap::get(2, 0, {getAffineDimExpr(1, ctx)}, ctx);

    SmallVector<Attribute> newIterTypes = {
        vector::IteratorTypeAttr::get(ctx, vector::IteratorType::reduction),
        vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel),
    };

    auto newContract = vector::ContractionOp::create(
        rewriter, loc, lhsFlat, rhsFlat, accFlat,
        rewriter.getAffineMapArrayAttr({newLhsMap, newRhsMap, newAccMap}),
        ArrayAttr::get(ctx, newIterTypes), op.getKind());

    // Shape-cast result back to original type.
    Value result = newContract.getResult();
    if (maybeAccVecType) {
      result =
          vector::ShapeCastOp::create(rewriter, loc, maybeAccVecType, result);
    } else {
      result = vector::ExtractOp::create(rewriter, loc, result,
                                         SmallVector<int64_t>{0});
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
};

// Rewrites a 2D vector.contract (one reduction dim, one parallel dim) in
// normalized form into a chain of math.fma ops. Expects the contract to
// have identity maps and iterator types ["reduction", "parallel"].
// The TransposeContractOperands and FlattenContractOperands patterns should
// run first to bring contracts into this form.
//
// Starting from the innermost position of the reduction dimension,
// the lowering emits a single nested FMA chain as follows:
// fma(a0 ,b0, fma(a1, b1, fma(a2, b2, fma(a3, b3, acc))))
// where ai and bi are the elements extracted from lhs and rhs vectors
// respectively along the reduction dimension.
//
// Example (after Transpose + Flatten):
// ```mlir
// #map = affine_map<(d0, d1) -> (d0, d1)>
// #map1 = affine_map<(d0, d1) -> (d1)>
// vector.contract {
//    indexing_maps = [#map, #map, #map1],
//    iterator_types = ["reduction", "parallel"],
//    kind = #vector.kind<add>
// } %lhs, %rhs, %acc : vector<8x2xf16>, vector<8x2xf16>
//                       into vector<2xf16>
// ```
//
// ==>
// ```mlir
// %34 = math.fma %32, %33, %acc : vector<2xf16>
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

    if (!isa<FloatType>(lhsVecType.getElementType())) {
      return failure();
    }

    // Expect exactly 2D: ["reduction", "parallel"] with identity LHS/RHS maps.
    ArrayAttr iteratorTypes = op.getIteratorTypes();
    if (iteratorTypes.size() != 2) {
      return failure();
    }
    if (!vector::isReductionIterator(iteratorTypes[0]) ||
        vector::isReductionIterator(iteratorTypes[1])) {
      return failure();
    }

    SmallVector<AffineMap, 4> maps = op.getIndexingMapsArray();
    AffineMap identityMap =
        AffineMap::getMultiDimIdentityMap(2, op.getContext());
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
    }

    if (rhsVecType.getElementType() != elemType) {
      Type promotedType = rhsVecType.clone(elemType);
      rhs = arith::ExtFOp::create(rewriter, loc, promotedType, rhs);
    }

    int64_t K = lhsVecType.getDimSize(0);

    Value acc;
    auto maybeAccVecType = dyn_cast<VectorType>(op.getAccType());
    if (maybeAccVecType) {
      acc = op.getAcc();
    } else {
      auto flatAccType = VectorType::get({lhsVecType.getDimSize(1)}, elemType);
      acc =
          vector::BroadcastOp::create(rewriter, loc, flatAccType, op.getAcc());
    }

    Value result = buildFMAChain(rewriter, loc, lhs, rhs, acc, K);

    if (!maybeAccVecType) {
      result = vector::ExtractOp::create(rewriter, loc, result, 0);
    }

    rewriter.replaceOp(op, result);
    return success();
  }

private:
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

struct UnrollElementwiseOps final : RewritePattern {
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
      contractLoweringPatterns.add<TransposeContractOperands>(
          funcOp->getContext(), PatternBenefit(2));
      contractLoweringPatterns.add<FlattenContractOperands>(
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
      contractLoweringPatterns.add<UnrollElementwiseOps>(funcOp->getContext());
      // Unroll transfer_gather/scatter ops to rank 1 and lower contiguous ones
      // to vector.transfer_read/write.
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
