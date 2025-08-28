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

struct ContractAddToChainFMA final : OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getKind() != vector::CombiningKind::ADD) {
      return failure();
    }

    // Static vector types only.
    auto lhsVT = dyn_cast<VectorType>(op.getLhsType());
    auto rhsVT = dyn_cast<VectorType>(op.getRhsType());
    if (!lhsVT || !rhsVT || !lhsVT.hasStaticShape() ||
        !rhsVT.hasStaticShape()) {
      return failure();
    }

    // Floats only
    Type elemTy = lhsVT.getElementType();
    if (!isa<FloatType>(elemTy)) {
      return failure();
    }

    // Acc/result type may be vector or scalar.
    auto accVT = dyn_cast<VectorType>(op.getAccType());
    if (accVT && !accVT.hasStaticShape()) {
      return failure();
    }

    // Iterator partition: exactly one reduction dim.
    // how come only one reduction dimension is allowed?
    SmallVector<int64_t> redDims, parDims;
    getReductionAndParallelLoopDims(op.getIteratorTypes(), redDims, parDims);
    if (redDims.empty()) {
      return failure();
    }

    // Loop order: [reduction..., parallel...].
    SmallVector<int64_t> loopOrder;
    loopOrder.append(redDims.begin(), redDims.end());
    loopOrder.append(parDims.begin(), parDims.end());

    // Indexing maps -> operandDimToLoopDim.
    SmallVector<AffineMap, 3> maps = op.getIndexingMapsArray();
    SmallVector<unsigned, 4> lhsDimToLoop, rhsDimToLoop, accDimToLoop;
    // Check for broadcasts (we don't support them)
    if (!isProjectedPermutation(maps[0], lhsDimToLoop) ||
        !isProjectedPermutation(maps[1], rhsDimToLoop)) {
      return failure();
    }

    if (accVT) {
      if (!isProjectedPermutation(maps[2], accDimToLoop)) {
        return failure();
      }
      // Accumulator must not reference reduction dims
      for (unsigned loop : accDimToLoop) {
        if (llvm::is_contained(redDims, loop)) {
          return failure();
        }
      }
    }

    // Build per-operand permutations that order vector dims as loopOrder.
    SmallVector<int64_t> lhsPerm, rhsPerm, accPerm;
    if (failed(buildOperandPermutation(lhsDimToLoop, loopOrder, lhsPerm))) {
      return failure();
    }
    if (failed(buildOperandPermutation(rhsDimToLoop, loopOrder, rhsPerm))) {
      return failure();
    }
    if (accVT) {
      // Build acc permutation w.r.t *parallel* dims order only: remove
      // reduction dims from loopOrder.
      SmallVector<int64_t> parOrder =
          parDims; // already in loopOrder’s tail order
      if (failed(buildOperandPermutation(accDimToLoop, parOrder, accPerm))) {
        return failure();
      }
    }

    Location loc = op.getLoc();

    // Transpose operands to [red..., par...].
    Value lhsTransposed =
        transposeIfNeeded(rewriter, loc, op.getLhs(), lhsVT, lhsPerm, elemTy);
    VectorType lhsTransposedType = lhsVT;
    if (!isIdentityPermutation(lhsPerm)) {
      SmallVector<int64_t> newShape = permuteShape(lhsVT.getShape(), lhsPerm);
      lhsTransposedType = VectorType::get(newShape, elemTy);
    }

    Value rhsTransposed =
        transposeIfNeeded(rewriter, loc, op.getRhs(), rhsVT, rhsPerm, elemTy);
    VectorType rhsTransposedType = rhsVT;
    if (!isIdentityPermutation(rhsPerm)) {
      SmallVector<int64_t> newShape = permuteShape(rhsVT.getShape(), rhsPerm);
      rhsTransposedType = VectorType::get(newShape, elemTy);
    }

    // Check the first `numRed` dims agree between LHS/RHS; same for parallel
    // product size.
    const unsigned numRedDims = redDims.size();
    int64_t lhsRedSize = productOfDims(lhsTransposedType, 0, numRedDims);
    int64_t rhsRedSize = productOfDims(rhsTransposedType, 0, numRedDims);
    if (lhsRedSize != rhsRedSize) {
      return failure();
    }

    int64_t lhsParSize = productOfDims(lhsTransposedType, numRedDims,
                                       lhsTransposedType.getRank());
    int64_t rhsParSize = productOfDims(rhsTransposedType, numRedDims,
                                       rhsTransposedType.getRank());
    if (lhsParSize != rhsParSize) {
      return failure();
    }
    int64_t redSize = lhsRedSize;
    int64_t parSize = lhsParSize;

    // Shape-cast operands to 2D {K, P}.
    VectorType flattened2DType = VectorType::get({redSize, parSize}, elemTy);
    Value lhs2D = rewriter.create<vector::ShapeCastOp>(loc, flattened2DType,
                                                       lhsTransposed);
    Value rhs2D = rewriter.create<vector::ShapeCastOp>(loc, flattened2DType,
                                                       rhsTransposed);

    // Prepare accumulator as {P}.
    Value flattenedAcc;
    VectorType parVT = VectorType::get({parSize}, elemTy);

    if (accVT) {
      Value acc = op.getAcc();
      if (!isIdentityPermutation(accPerm)) {
        auto newShape = permuteShape(accVT.getShape(), accPerm);
        auto accTransposedType = VectorType::get(newShape, elemTy);
        acc = rewriter.create<vector::TransposeOp>(loc, accTransposedType, acc,
                                                   accPerm);
      }
      flattenedAcc = rewriter.create<vector::ShapeCastOp>(loc, parVT, acc);
    } else {
      // Scalar accumulator: splat to vector
      flattenedAcc = rewriter.create<vector::SplatOp>(loc, parVT, op.getAcc());
    }

    // Build FMA chain: resultFlat : vector<Pxf32>
    const int64_t chunkSize = 2;
    Value resultFlat = buildFMAChain2D(
        rewriter, loc, lhs2D, rhs2D, flattenedAcc, redSize, parSize, chunkSize);

    // Restore result shape/types to match op.getAccType()/op.getType().
    Value resultVal;
    if (accVT) {
      // Unflatten to the (possibly transposed) acc shape, then
      // inverse-transpose. First, compute the intermediate (transposed) shape.
      VectorType accTType = accVT;
      if (!isIdentityPermutation(accPerm)) {
        SmallVector<int64_t> newShape = permuteShape(accVT.getShape(), accPerm);
        accTType = VectorType::get(newShape, elemTy);
      }

      Value reshaped =
          rewriter.create<vector::ShapeCastOp>(loc, accTType, resultFlat);
      if (!isIdentityPermutation(accPerm)) {
        // Inverse permutation.
        SmallVector<int64_t> inv(accPerm.size(), 0);
        for (int64_t i = 0, e = accPerm.size(); i < e; ++i) {
          inv[accPerm[i]] = i;
        }

        resultVal =
            rewriter.create<vector::TransposeOp>(loc, accVT, reshaped, inv);
      } else {
        resultVal = reshaped; // already accVT shape
      }
    } else {
      // Scalar result: P must be 1; extract element 0.
      if (parSize != 1) {
        return failure(); // not a dot if P>1
      }
      resultVal = rewriter.create<vector::ExtractOp>(loc, resultFlat,
                                                     ArrayRef<int64_t>{0});
    }

    rewriter.replaceOp(op, resultVal);
    return success();
  }

private:
  static SmallVector<int64_t> permuteShape(ArrayRef<int64_t> shape,
                                           ArrayRef<int64_t> perm) {
    SmallVector<int64_t> out(shape.size());
    for (auto [idx, val] : llvm::enumerate(perm)) {
      out[idx] = shape[val];
    }
    return out;
  }

  // Extract iterator classification.
  static void getReductionAndParallelLoopDims(ArrayAttr iters,
                                              SmallVectorImpl<int64_t> &red,
                                              SmallVectorImpl<int64_t> &par) {
    for (auto [idx, attr] : llvm::enumerate(iters)) {
      auto it_type = llvm::cast<vector::IteratorTypeAttr>(attr).getValue();
      if (it_type == vector::IteratorType::reduction) {
        red.push_back(idx);
      } else {
        par.push_back(idx);
      }
    }
  }

  // Returns true iff `map` is a projected permutation of loop dims (pure dim
  // exprs, no symbols). Fills `operandDimToLoopDim[j] = d` where j is the j-th
  // vector dim of this operand.

  // map of opernaddim index (e.g 4) to loop dim index (d0)
  static bool isProjectedPermutation(AffineMap map,
                                     SmallVectorImpl<unsigned> &dimMapping) {
    if (!map || map.getNumSymbols() != 0) {
      return false;
    }

    dimMapping.resize(map.getNumResults());
    llvm::SmallBitVector seen(map.getNumDims());

    for (auto [i, expr] : llvm::enumerate(map.getResults())) {
      auto dim = dyn_cast<AffineDimExpr>(expr);
      if (!dim) {
        return false;
      }
      unsigned pos = dim.getPosition();
      if (seen[pos]) {
        return false;
      }
      seen[pos] = true;
      dimMapping[i] = pos;
    }

    return true;
  }

  // Build a permutation that orders operand vector dims to follow `loopOrder`.
  // `operandDimToLoopDim[j] = loopDim` (from the indexing map).
  // We produce `perm` such that: new[i] = old[perm[i]] with
  // sorted-by-loopOrder.

  static LogicalResult
  buildOperandPermutation(ArrayRef<unsigned> operandDimToLoopDim,
                          ArrayRef<int64_t> loopOrder,
                          SmallVectorImpl<int64_t> &perm) {
    llvm::SmallDenseMap<unsigned, unsigned> loopToPosition;
    for (auto [pos, loop] : llvm::enumerate(loopOrder)) {
      loopToPosition[loop] = pos;
    }

    for (unsigned loop : operandDimToLoopDim) {
      if (!loopToPosition.count(loop)) {
        return failure();
      }
    }

    perm.resize(operandDimToLoopDim.size());
    std::iota(perm.begin(), perm.end(), 0);

    llvm::stable_sort(perm, [&](int64_t i, int64_t j) {
      unsigned posI = loopToPosition[operandDimToLoopDim[i]];
      unsigned posJ = loopToPosition[operandDimToLoopDim[j]];
      return posI < posJ;
    });

    return success();
  }

  // Multiply a slice of dims: [lo, hi).
  static int64_t productOfDims(VectorType vt, unsigned lo, unsigned hi) {
    int64_t p = 1;
    for (unsigned i = lo; i < hi; ++i) {
      p *= vt.getDimSize(i);
    }
    return p;
  }

  // Identity test.
  static bool isIdentityPermutation(ArrayRef<int64_t> p) {
    for (int64_t i = 0, e = p.size(); i < e; ++i) {
      if (p[i] != i) {
        return false;
      }
    }
    return true;
  }

  static Value transposeIfNeeded(PatternRewriter &rewriter, Location loc,
                                 Value input, VectorType inputType,
                                 ArrayRef<int64_t> perm, Type elemTy) {
    if (isIdentityPermutation(perm)) {
      return input;
    }

    auto newShape = permuteShape(inputType.getShape(), perm);
    auto newType = VectorType::get(newShape, elemTy);
    return rewriter.create<vector::TransposeOp>(loc, newType, input, perm);
  }

  // Helper: Process a vector chunk
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

  // Helper: Process a single scalar element
  static Value processScalarFMA(PatternRewriter &rewriter, Location loc,
                                Value lhsRow, Value rhsRow, Value acc,
                                int64_t index) {

    Value a = rewriter.create<vector::ExtractOp>(loc, lhsRow, index);
    Value b = rewriter.create<vector::ExtractOp>(loc, rhsRow, index);
    Value c = rewriter.create<vector::ExtractOp>(loc, acc, index);

    Value fma = rewriter.create<math::FmaOp>(loc, a, b, c);

    return rewriter.create<vector::InsertOp>(loc, fma, acc, index);
  }
  // Core: produce resultFlat := fold_k fma(lhs2D[k,*], rhs2D[k,*], accFlat)
  // using vector<2xf32> chunks, chaining from inner-most FMA (backwards).
  static Value buildFMAChain2D(PatternRewriter &rewriter, Location loc,
                               Value lhs2D, Value rhs2D, Value accFlat,
                               int64_t K, int64_t P, int64_t chunkSize) {
    Value current = accFlat;

    for (int64_t k = K - 1; k >= 0; --k) {
      Value lhsRow =
          rewriter.create<vector::ExtractOp>(loc, lhs2D, ArrayRef<int64_t>{k});
      Value rhsRow =
          rewriter.create<vector::ExtractOp>(loc, rhs2D, ArrayRef<int64_t>{k});

      // Special case: if P < chunkSize, do a single vector FMA
      if (P < chunkSize) {
        current = rewriter.create<math::FmaOp>(loc, lhsRow, rhsRow, current);
        continue;
      }

      // Process full chunks
      int64_t p = 0;
      for (; p + chunkSize <= P; p += chunkSize) {
        current = processFMAChunk(rewriter, loc, lhsRow, rhsRow, current, p,
                                  chunkSize);
      }

      // Process remaining scalars
      for (; p < P; ++p) {
        current = processScalarFMA(rewriter, loc, lhsRow, rhsRow, current, p);
      }
    }
    return current;
  }
};

//struct ContractAddToChainFMA final : OpRewritePattern<vector::ContractionOp> {
//  using OpRewritePattern::OpRewritePattern;
//
//  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
//                                PatternRewriter &rewriter) const override {
//    if (contractOp.getKind() != vector::CombiningKind::ADD) {
//      return failure();
//    }
//
//    Type elemType = getElementTypeOrSelf(contractOp.getLhsType());
//    if (!isa<Float32Type>(elemType)) {
//      return failure();
//    }
//
//    Location loc = contractOp.getLoc();
//    VectorType lhsType = cast<VectorType>(contractOp.getLhsType());
//    VectorType rhsType = cast<VectorType>(contractOp.getRhsType());
//    VectorType accType = cast<VectorType>(contractOp.getAccType());
//
//    if (!lhsType || !rhsType || !accType) {
//      return failure();
//    }
//
//    if (!isStaticVector(lhsType) || !isStaticVector(rhsType) || !isStaticVector(accType)) {
//      return failure();
//    }
//
//    // Get indexing maps and iterator types
//    SmallVector<AffineMap> indexingMaps = contractOp.getIndexingMapsArray();
//    if (indexingMaps.size() != 3) {
//      return failure();
//    }
//    auto iteratorTypes = contractOp.getIteratorTypes();
//
//    // Identify parallel and reduction dimensions
//    SmallVector<int64_t> parallelDims, reductionDims;
//    for (auto [idx, attr] : llvm::enumerate(iteratorTypes)) {
//      if (dyn_cast<vector::IteratorTypeAttr>(attr).getValue() ==
//          vector::IteratorType::parallel) {
//        parallelDims.push_back(idx);
//      } else {
//        reductionDims.push_back(idx);
//      }
//    }
//
//
//    // We want reduction dims in outer positions for processing
//    // Create transpose permutation: [reduction_dims..., parallel_dims...]
//    SmallVector<int64_t> transposePermutation;
//    transposePermutation.append(reductionDims.begin(), reductionDims.end());
//    transposePermutation.append(parallelDims.begin(), parallelDims.end());
//
//    // Check if we need to transpose
//    bool needsTranspose = false;
//    for (size_t i = 0; i < transposePermutation.size(); i++) {
//      if (transposePermutation[i] != i) {
//        needsTranspose = true;
//        break;
//      }
//    }
//
//    Value processedLhs = contractOp.getLhs();
//    Value processedRhs = contractOp.getRhs();
//    Value processedAcc = contractOp.getAcc();
//
//    if (needsTranspose) {
//      // Apply transpose to operands
//      SmallVector<int64_t> newShape =
//          applyPermutation(lhsType.getShape(), transposePermutation);
//      VectorType transposedType = VectorType::get(newShape, lhsType.getElementType());
//
//      processedLhs = rewriter.create<vector::TransposeOp>(
//          loc, transposedType, contractOp.getLhs(), transposePermutation);
//      processedRhs = rewriter.create<vector::TransposeOp>(
//          loc, transposedType, contractOp.getRhs(), transposePermutation);
//    }
//    processedLhs.dump();
//    processedRhs.dump();
//    processedAcc.dump();
//    accType.dump();
//
//    SmallVector<int64_t> accTransposePermutation;
//
//
//
//    // After transpose, we have shape like [reduction_dims..., parallel_dims...]
//    // For example: 4x1x8 -> 8x4x1 if reduction is dim 2
//
//    // Generate FMA chains
//    Value result = generateFMAChains(rewriter, loc, processedLhs, processedRhs,
//                                     contractOp.getAcc(), reductionDims.size(),
//                                     parallelDims.size());
//
//    rewriter.replaceOp(contractOp, result);
//    return success();
//  }
//
//private:
//  Value generateFMAChains(PatternRewriter &rewriter, Location loc, Value lhs,
//                          Value rhs, Value acc, size_t numReductionDims,
//                          size_t numParallelDims) const {
//    auto lhsType = cast<VectorType>(lhs.getType());
//
//    // After transpose, shape is [reduction_dims..., parallel_dims...]
//    // Get the reduction dimension size (product of all reduction dims)
//    int64_t reductionSize = 1;
//    for (size_t i = 0; i < numReductionDims; i++) {
//      reductionSize *= lhsType.getDimSize(i); // e.g., 8
//    }
//
//    // Get the parallel dimension size (product of all parallel dims)
//    int64_t parallelSize = 1;
//    for (size_t i = numReductionDims; i < lhsType.getRank(); i++) {
//      parallelSize *= lhsType.getDimSize(i); // e.g., 4
//    }
//
//    // Reshape to 2D: [reductionSize, parallelSize]
//    // e.g., 8x4
//    SmallVector<int64_t> shape2D = {reductionSize, parallelSize};
//    auto type2D = VectorType::get(shape2D, lhsType.getElementType());
//
//    Value lhs2D = rewriter.create<vector::ShapeCastOp>(loc, type2D, lhs);
//    Value rhs2D = rewriter.create<vector::ShapeCastOp>(loc, type2D, rhs);
//
//    // Transpose accumulator if needed to match [1, parallelSize]
//    // acc comes in as e.g., vector<4x1xf32>, we need vector<1x4xf32>
//    auto accType = cast<VectorType>(acc.getType());
//    Value transposedAcc = acc;
//
//    // Check if acc needs transposing
//    if (accType.getRank() > 1) {
//      // Create transpose permutation to swap dimensions
//      SmallVector<int64_t> accTranspose;
//      for (int64_t i = accType.getRank() - 1; i >= 0; i--) {
//        accTranspose.push_back(i);
//      }
//      auto transposedAccType =
//          VectorType::get({1, parallelSize}, accType.getElementType());
//      transposedAcc = rewriter.create<vector::TransposeOp>(
//          loc, transposedAccType, acc, accTranspose);
//    } else {
//      // If 1D, just add a leading dimension
//      auto reshapedType =
//          VectorType::get({1, parallelSize}, accType.getElementType());
//      transposedAcc =
//          rewriter.create<vector::ShapeCastOp>(loc, reshapedType, acc);
//    }
//
//    // Extract the accumulator row (it's at position 0 after reshape)
//    Value acc1D = rewriter.create<vector::ExtractOp>(loc, transposedAcc,
//                                                     ArrayRef<int64_t>{0});
//
//    // Process in chunks of 2 for v_pk_fma_f32
//    const int64_t chunkSize = 2;
//
//    // Initialize result - will be set in first iteration
//    Value result;
//
//    // Process each reduction step BACKWARDS for proper FMA chaining
//    // This creates: fma(a[0], b[0], fma(a[1], b[1], fma(..., fma(a[n-1],
//    // b[n-1], acc))))
//    for (int64_t redIdx = reductionSize - 1; redIdx >= 0; redIdx--) {
//      // Extract the row for this reduction step
//      Value lhsRow = rewriter.create<vector::ExtractOp>(
//          loc, lhs2D, ArrayRef<int64_t>{redIdx});
//      Value rhsRow = rewriter.create<vector::ExtractOp>(
//          loc, rhs2D, ArrayRef<int64_t>{redIdx});
//
//      // First iteration uses accumulator, subsequent iterations use previous
//      // result
//      bool isFirstIteration = (redIdx == reductionSize - 1);
//      Value currentAcc = isFirstIteration ? acc1D : result;
//
//      // Process in chunks of 2 elements for packed FMA
//      if (parallelSize >= chunkSize && parallelSize % chunkSize == 0) {
//        // Create a new result vector for this iteration
//        Value iterResult = currentAcc; // Start with current accumulator
//
//        for (int64_t chunkIdx = 0; chunkIdx < parallelSize;
//             chunkIdx += chunkSize) {
//          SmallVector<int64_t> offsets = {chunkIdx};
//          SmallVector<int64_t> sizes = {chunkSize};
//          SmallVector<int64_t> strides = {1};
//
//          // auto chunkType =
//              // VectorType::get({chunkSize}, lhsType.getElementType());
//
//          // Extract vector<2xf32> chunks from current row
//          Value lhsChunk = rewriter.create<vector::ExtractStridedSliceOp>(
//              loc, lhsRow, offsets, sizes, strides);
//          Value rhsChunk = rewriter.create<vector::ExtractStridedSliceOp>(
//              loc, rhsRow, offsets, sizes, strides);
//
//          // Extract corresponding chunk from current accumulator
//          Value accChunk = rewriter.create<vector::ExtractStridedSliceOp>(
//              loc, currentAcc, offsets, sizes, strides);
//
//          // Perform vector<2xf32> FMA - enables v_pk_fma_f32!
//          Value fmaResult =
//              rewriter.create<math::FmaOp>(loc, lhsChunk, rhsChunk, accChunk);
//
//          // Insert result back
//          iterResult = rewriter.create<vector::InsertStridedSliceOp>(
//              loc, fmaResult, iterResult, offsets, strides);
//        }
//
//        result = iterResult;
//      } else if (parallelSize >= chunkSize) {
//        // Handle case where parallelSize is not divisible by chunkSize
//        Value iterResult = currentAcc;
//
//        for (int64_t chunkIdx = 0; chunkIdx < parallelSize;
//             chunkIdx += chunkSize) {
//          int64_t actualChunkSize =
//              std::min(chunkSize, parallelSize - chunkIdx);
//
//          if (actualChunkSize == chunkSize) {
//            // Full chunk - use vector ops
//            SmallVector<int64_t> offsets = {chunkIdx};
//            SmallVector<int64_t> sizes = {chunkSize};
//            SmallVector<int64_t> strides = {1};
//
//            // auto chunkType =
//                // VectorType::get({chunkSize}, lhsType.getElementType());
//
//            Value lhsChunk = rewriter.create<vector::ExtractStridedSliceOp>(
//                loc, lhsRow, offsets, sizes, strides);
//            Value rhsChunk = rewriter.create<vector::ExtractStridedSliceOp>(
//                loc, rhsRow, offsets, sizes, strides);
//            Value accChunk = rewriter.create<vector::ExtractStridedSliceOp>(
//                loc, currentAcc, offsets, sizes, strides);
//
//            Value fmaResult =
//                rewriter.create<math::FmaOp>(loc, lhsChunk, rhsChunk, accChunk);
//
//            iterResult = rewriter.create<vector::InsertStridedSliceOp>(
//                loc, fmaResult, iterResult, offsets, strides);
//          } else {
//            // Remainder - handle with scalar FMA
//            for (int64_t i = 0; i < actualChunkSize; i++) {
//              int64_t idx = chunkIdx + i;
//              Value lhsElem = rewriter.create<vector::ExtractOp>(
//                  loc, lhsRow, ArrayRef<int64_t>{idx});
//              Value rhsElem = rewriter.create<vector::ExtractOp>(
//                  loc, rhsRow, ArrayRef<int64_t>{idx});
//              Value accElem = rewriter.create<vector::ExtractOp>(
//                  loc, currentAcc, ArrayRef<int64_t>{idx});
//
//              Value fmaResult =
//                  rewriter.create<math::FmaOp>(loc, lhsElem, rhsElem, accElem);
//
//              iterResult = rewriter.create<vector::InsertOp>(
//                  loc, fmaResult, iterResult, ArrayRef<int64_t>{idx});
//            }
//          }
//        }
//
//        result = iterResult;
//      } else {
//        // Direct FMA on the whole row if it's smaller than chunk size
//        result = rewriter.create<math::FmaOp>(loc, lhsRow, rhsRow, currentAcc);
//      }
//    }
//
//    // Reshape result back to original accumulator shape
//    // e.g., vector<4xf32> -> vector<4x1xf32>
//    result = rewriter.create<vector::ShapeCastOp>(loc, accType, result);
//
//    return result;
//  }
//  //   SmallVector<int64_t> applyPermutation(ArrayRef<int64_t> shape,
//  //                                         ArrayRef<int64_t> permutation)
//  //                                         const {
//  //     SmallVector<int64_t> result;
//  //     for (int64_t idx : permutation) {
//  //       result.push_back(shape[idx]);
//  //     }
//  //     return result;
//  // }
//};

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
