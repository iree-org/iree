// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "llvm/Support/InterleavedRange.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::VectorExt;

using VectorValue = TypedValue<VectorType>;

//===----------------------------------------------------------------------===//
// LayoutConflictResolutionOp
//===----------------------------------------------------------------------===//

// Validate that the layout has the same shape as the input.
LogicalResult ToLayoutOp::verify() {
  return getLayout().isValidLayout(getInput().getType(), getLoc());
}

// to_simd -> to_simt
OpFoldResult ToSIMDOp::fold(FoldAdaptor) {
  if (auto simtOp = getOperand().getDefiningOp<ToSIMTOp>()) {
    return simtOp.getOperand();
  }
  return {};
}

// to_simt -> to_simd
OpFoldResult ToSIMTOp::fold(FoldAdaptor) {
  if (auto simdOp = getOperand().getDefiningOp<ToSIMDOp>()) {
    return simdOp.getOperand();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// TransferGatherOp
//===----------------------------------------------------------------------===//

Speculation::Speculatability TransferGatherOp::getSpeculatability() {
  if (isa<RankedTensorType>(getBase().getType())) {
    return Speculation::Speculatable;
  }
  return Speculation::NotSpeculatable;
}

void TransferGatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (isa<MemRefType>(getBase().getType())) {
    effects.emplace_back(MemoryEffects::Read::get(), &getBaseMutable(),
                         SideEffects::DefaultResource::get());
  }
}

// Verifier.

LogicalResult TransferGatherOp::verify() {
  OperandRange indexVecs = getIndexVecs();
  TypedValue<VectorType> vector = getVector();
  Value mask = getMask();
  SmallVector<AffineMap> indexingMaps = getIndexingMapsArray();

  // Check that we have the correct number of indexing maps.
  int64_t expectedNumIndexingMaps =
      /*sourceIndexingMap=*/1 + /*indexVecIndexingMaps=*/indexVecs.size() +
      /*maskIndexingMap=*/(mask ? 1 : 0);
  if (expectedNumIndexingMaps != static_cast<int64_t>(indexingMaps.size())) {
    return emitOpError("expected ")
           << expectedNumIndexingMaps
           << " indexing maps, got: " << indexingMaps.size();
  }

  int64_t vectorRank = vector.getType().getRank();
  int64_t indexSyms = indexVecs.size();
  for (AffineMap map : indexingMaps) {
    if (map.getNumDims() != vectorRank) {
      return emitOpError("expected all indexing maps to have number of dims "
                         "equal to vector rank. expected: ")
             << vectorRank << ", got: " << map.getNumDims() << " dims";
    }
    if (map.getNumSymbols() != indexSyms) {
      return emitOpError("expected all indexing maps to have number of symbols "
                         "equal to number of index vecs. expected: ")
             << indexSyms << ", got: " << map.getNumSymbols() << " syms";
    }
    for (AffineExpr expr : map.getResults()) {
      if (isa<AffineDimExpr, AffineSymbolExpr>(expr)) {
        continue;
      }
      if (auto constExpr = dyn_cast<AffineConstantExpr>(expr)) {
        if (constExpr.getValue() != 0) {
          return emitOpError("expected constant 0 in indexing map, got: ")
                 << constExpr.getValue();
        }
        continue;
      }
      return emitOpError(
          "expected indexing map results to only be a dim, symbol, or 0");
    }
  }

  // Extra verification for index vecs.
  ArrayRef<int64_t> vectorShape = vector.getType().getShape();
  ArrayRef<AffineMap> vectorIndexingMaps =
      ArrayRef(indexingMaps).slice(1, indexSyms);
  for (auto [i, map] : llvm::enumerate(vectorIndexingMaps)) {
    SmallVector<int64_t> expectedShape;
    for (AffineExpr expr : map.getResults()) {
      if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
        expectedShape.push_back(vectorShape[dim.getPosition()]);
      } else {
        return emitOpError(
            "expected vector indexing maps to not have any symbols");
      }
    }
    // Scalar index: map must have 0 results and type must be plain index.
    if (isa<IndexType>(indexVecs[i].getType())) {
      if (!expectedShape.empty()) {
        return emitOpError("expected empty indexing map for scalar index vec "
                           "at position ")
               << i;
      }
      continue;
    }
    ArrayRef<int64_t> actualShape =
        cast<VectorType>(indexVecs[i].getType()).getShape();
    if (ArrayRef<int64_t>(expectedShape) != actualShape) {
      return emitOpError("Mismatched vector shape for index vec at position ")
             << i << ". Expected: [" << expectedShape << "]" << ", got: ["
             << actualShape << "]";
    }
  }

  // Extra verification for mask.
  if (mask) {
    AffineMap maskMap = indexingMaps.back();
    SmallVector<int64_t> expectedShape;
    for (AffineExpr expr : maskMap.getResults()) {
      if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
        expectedShape.push_back(vectorShape[dim.getPosition()]);
      } else {
        return emitOpError(
            "expected mask indexing map to not have any symbols");
      }
    }
    ArrayRef<int64_t> actualShape = cast<VectorType>(mask.getType()).getShape();
    if (ArrayRef<int64_t>(expectedShape) != actualShape) {
      return emitOpError("Mismatched mask shape")
             << ". Expected: [" << expectedShape << "]" << ", got: ["
             << actualShape << "]";
    }
  }

  return success();
}

// Fold and canonicalization helpers.

static int64_t getVectorRank(Type type) {
  if (auto vecType = dyn_cast<VectorType>(type)) {
    return vecType.getRank();
  }
  return 0;
}

struct IndexingMapFoldResult {
  Value operand;
  AffineMap indexingMap;
  bool changed;
};

using IndexingMapFolder = function_ref<IndexingMapFoldResult(
    int64_t index, Value val, AffineMap valMap, AffineMap &baseMap)>;

static Value foldTransferGatherIndexVecs(TransferGatherOp op,
                                         IndexingMapFolder valueFolder) {
  SmallVector<Value> indexedValues(op.getIndexVecs());
  SmallVector<AffineMap> indexingMaps(
      ArrayRef(op.getIndexingMapsArray()).slice(1, indexedValues.size()));

  AffineMap baseMap = op.getIndexingMapsArray().front();

  bool changed = false;
  SmallVector<Value> newIndexedValues;
  SmallVector<AffineMap> newIndexingMaps;
  llvm::DenseSet<int64_t> deletedSyms;
  for (auto [index, val, map] : llvm::enumerate(indexedValues, indexingMaps)) {
    auto [newVal, newMap, valChanged] = valueFolder(index, val, map, baseMap);
    changed |= valChanged;

    if (newVal) {
      newIndexedValues.push_back(newVal);
      newIndexingMaps.push_back(newMap);
    } else {
      deletedSyms.insert(index);
    }
  }

  // The mask is passed through the same folder as index vecs. Folders must
  // handle the mask case correctly — the mask's index is indexedValues.size()
  // which won't match any symbol in the base map, so index-based folds
  // (FoldSingleElementIndexVec, foldTransferGatherFromStep) will be no-ops
  // on the mask, while shape-based folds (broadcast, transpose) will apply.
  Value mask;
  AffineMap maskMap;
  if (op.getMask()) {
    auto [newMask, newMap, valChanged] =
        valueFolder(indexedValues.size(), op.getMask(),
                    op.getIndexingMapsArray().back(), baseMap);
    changed |= valChanged;
    if (newMask) {
      mask = newMask;
      maskMap = newMap;
    }
  }
  if (!changed) {
    return Value();
  }

  OpBuilder b(op);

  // Collect all the indexing maps.
  SmallVector<AffineMap> updatedIndexingMaps;
  updatedIndexingMaps.push_back(baseMap);
  updatedIndexingMaps.append(newIndexingMaps);
  if (op.getMask()) {
    updatedIndexingMaps.push_back(maskMap);
  }

  // Delete the deleted symbols from these maps.
  if (!deletedSyms.empty()) {
    SmallVector<AffineExpr> symReplacements;
    int currSym = 0;
    for (auto i : llvm::seq<int>(baseMap.getNumSymbols())) {
      if (deletedSyms.contains(i)) {
        symReplacements.push_back(b.getAffineConstantExpr(0));
      } else {
        symReplacements.push_back(b.getAffineSymbolExpr(currSym));
        ++currSym;
      }
    }
    for (AffineMap &map : updatedIndexingMaps) {
      map = map.replaceDimsAndSymbols({}, symReplacements, map.getNumDims(),
                                      currSym);
    }
  }

  SmallVector<Value> operands;
  operands.push_back(op.getBase());
  llvm::append_range(operands, op.getOffsets());
  llvm::append_range(operands, newIndexedValues);
  operands.push_back(op.getPadding());
  if (mask) {
    operands.push_back(mask);
  }

  op.setIndexingMapsAttr(b.getAffineMapArrayAttr(updatedIndexingMaps));
  op->setOperands(operands);
  op.getProperties().setOperandSegmentSizes(
      {1, static_cast<int32_t>(op.getOffsets().size()),
       static_cast<int32_t>(newIndexedValues.size()), 1,
       static_cast<int32_t>(mask ? 1 : 0)});

  return op.getResult();
}

static Value foldTransferGatherFromBroadcast(TransferGatherOp op) {
  return foldTransferGatherIndexVecs(
      op,
      [](int64_t, Value operand, AffineMap map,
         AffineMap &) -> IndexingMapFoldResult {
        auto broadcast = operand.getDefiningOp<vector::BroadcastOp>();
        if (!broadcast) {
          return {operand, map, false};
        }

        int64_t sourceRank = getVectorRank(broadcast.getSourceType());
        int64_t operandRank = getVectorRank(broadcast.getResultVectorType());
        AffineMap newMap =
            map.getSliceMap(operandRank - sourceRank, sourceRank);
        return {broadcast.getSource(), newMap, true};
      });
}

static Value foldTransferGatherFromTranspose(TransferGatherOp op) {
  return foldTransferGatherIndexVecs(
      op,
      [](int64_t, Value operand, AffineMap map,
         AffineMap &) -> IndexingMapFoldResult {
        auto transpose = operand.getDefiningOp<vector::TransposeOp>();
        if (!transpose) {
          return {operand, map, false};
        }

        AffineMap newMap =
            AffineMap::getPermutationMap(
                invertPermutationVector(transpose.getPermutation()),
                transpose.getContext())
                .compose(map);
        return {transpose.getVector(), newMap, true};
      });
}

static Value foldTransferGatherFromStep(TransferGatherOp op) {
  return foldTransferGatherIndexVecs(
      op,
      [](int64_t index, Value operand, AffineMap map,
         AffineMap &baseMap) -> IndexingMapFoldResult {
        auto step = operand.getDefiningOp<vector::StepOp>();
        if (!step) {
          return {operand, map, false};
        }

        assert(map.getNumResults() == 1);
        // Replace the symbol in the base map with the dim expression from the
        // index vec map, making this dimension contiguous.
        SmallVector<AffineExpr> newResults;
        for (AffineExpr expr : baseMap.getResults()) {
          if (auto sym = dyn_cast<AffineSymbolExpr>(expr)) {
            if (sym.getPosition() == index) {
              expr = map.getResult(0);
            }
          }
          newResults.push_back(expr);
        }
        baseMap = AffineMap::get(baseMap.getNumDims(), baseMap.getNumSymbols(),
                                 newResults, baseMap.getContext());
        return {Value(), AffineMap(), true};
      });
}

OpFoldResult TransferGatherOp::fold(FoldAdaptor adaptor) {
  if (auto res = foldTransferGatherFromBroadcast(*this)) {
    return res;
  }
  if (auto res = foldTransferGatherFromTranspose(*this)) {
    return res;
  }
  if (auto res = foldTransferGatherFromStep(*this)) {
    return res;
  }
  return OpFoldResult();
}

/// Apply an affine map transformation to a vector using broadcast and
/// transpose operations.
static Value applyTransformMapToVector(PatternRewriter &rewriter, Location loc,
                                       Value source, AffineMap map,
                                       ArrayRef<int64_t> targetShape) {
  auto sourceType = cast<VectorType>(source.getType());
  int64_t targetRank = map.getNumDims();
  int64_t sourceRank = sourceType.getRank();

  assert(map.getNumResults() == sourceRank &&
         "Map results must match source rank");

  // If already the right shape, no transformation needed.
  if (sourceRank == targetRank) {
    bool isIdentity = true;
    for (unsigned i = 0; i < sourceRank; ++i) {
      auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(i));
      if (!dimExpr || dimExpr.getPosition() != i) {
        isIdentity = false;
        break;
      }
    }
    if (isIdentity) {
      return source;
    }
  }

  // Build direct mapping: for each target dim, which source dim provides it
  SmallVector<int64_t> targetDimToSourceDim(targetRank, -1);
  for (int64_t srcDim = 0; srcDim < sourceRank; ++srcDim) {
    AffineExpr expr = map.getResult(srcDim);
    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
      targetDimToSourceDim[dimExpr.getPosition()] = srcDim;
    }
  }

  int64_t numBroadcastDims = llvm::count(targetDimToSourceDim, -1);

  // Build broadcast shape: [broadcast dim sizes..., source shape...]
  SmallVector<int64_t> broadcastShape;
  for (int64_t i = 0; i < targetRank; ++i) {
    if (targetDimToSourceDim[i] == -1) {
      broadcastShape.push_back(targetShape[i]);
    }
  }
  for (int64_t i = 0; i < sourceRank; ++i) {
    broadcastShape.push_back(sourceType.getDimSize(i));
  }

  // Broadcast to add dimensions
  VectorType broadcastType =
      VectorType::get(broadcastShape, sourceType.getElementType());
  Value result = source;
  if (broadcastType != sourceType) {
    result = vector::BroadcastOp::create(rewriter, loc, broadcastType, source);
  }

  // Compute transpose permutation
  SmallVector<int64_t> transposePerm(targetRank);
  int64_t bcastIdx = 0;
  for (int64_t i = 0; i < targetRank; ++i) {
    transposePerm[i] = targetDimToSourceDim[i] == -1
                           ? bcastIdx++
                           : numBroadcastDims + targetDimToSourceDim[i];
  }

  if (!llvm::equal(transposePerm, llvm::seq<int64_t>(0, targetRank))) {
    result = vector::TransposeOp::create(rewriter, loc, result, transposePerm);
  }

  return result;
}

struct FoldSingleElementIndexVec final : OpRewritePattern<TransferGatherOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(TransferGatherOp op,
                                PatternRewriter &rewriter) const override {

    auto indexVecFolder = [&](int64_t index, Value indexVec, AffineMap map,
                              AffineMap &baseMap) -> IndexingMapFoldResult {
      bool isScalar = isa<IndexType>(indexVec.getType());
      if (!isScalar) {
        auto vectorTy = cast<VectorType>(indexVec.getType());
        if (vectorTy.getNumElements() != 1) {
          return {indexVec, map, false};
        }
      }

      // Find which source dim this symbol corresponds to.
      AffineExpr symbolExpr = getAffineSymbolExpr(index, op.getContext());
      int64_t sourceDim = -1;
      for (auto [i, expr] : llvm::enumerate(baseMap.getResults())) {
        if (expr == symbolExpr) {
          sourceDim = i;
          break;
        }
      }
      if (sourceDim < 0) {
        return {indexVec, map, false};
      }

      // Extract the scalar and add it to the corresponding base offset.
      OpOperand &baseOffset = op.getOffsetsMutable()[sourceDim];
      Value extracted = indexVec;
      if (!isScalar) {
        auto vectorTy = cast<VectorType>(indexVec.getType());
        extracted = vector::ExtractOp::create(
            rewriter, op.getLoc(), indexVec,
            SmallVector<int64_t>(vectorTy.getRank(), 0));
      }

      AffineExpr d0, d1;
      bindDims(op.getContext(), d0, d1);

      Value newIndex = affine::makeComposedAffineApply(
                           rewriter, op.getLoc(), d0 + d1,
                           ArrayRef<OpFoldResult>{baseOffset.get(), extracted})
                           .getResult();
      baseOffset.set(newIndex);

      return {Value(), AffineMap(), true};
    };

    Value newVal = foldTransferGatherIndexVecs(op, indexVecFolder);

    if (!newVal) {
      return failure();
    }

    return success();
  }
};

/// Fold `arith.addi(something, broadcast(scalar))` index vecs by absorbing
/// the scalar into the base offset. This handles the common pattern after
/// unrolling where offsets get added to index vectors as broadcasts.
struct FoldIndexVecAddBroadcast final : OpRewritePattern<TransferGatherOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(TransferGatherOp op,
                                PatternRewriter &rewriter) const override {

    auto indexVecFolder = [&](int64_t index, Value indexVec, AffineMap map,
                              AffineMap &baseMap) -> IndexingMapFoldResult {
      auto addOp = indexVec.getDefiningOp<arith::AddIOp>();
      if (!addOp) {
        return {indexVec, map, false};
      }

      // Try both operand orders (addi is commutative).
      Value scalarSrc;
      Value remaining;
      for (auto [lhs, rhs] : {std::pair(addOp.getLhs(), addOp.getRhs()),
                              std::pair(addOp.getRhs(), addOp.getLhs())}) {
        auto broadcast = lhs.getDefiningOp<vector::BroadcastOp>();
        if (broadcast && isa<IndexType>(broadcast.getSourceType())) {
          scalarSrc = broadcast.getSource();
          remaining = rhs;
          break;
        }
      }
      if (!scalarSrc) {
        return {indexVec, map, false};
      }

      // Find which source dim this symbol corresponds to.
      AffineExpr symbolExpr = getAffineSymbolExpr(index, op.getContext());
      int64_t sourceDim = -1;
      for (auto [i, expr] : llvm::enumerate(baseMap.getResults())) {
        if (expr == symbolExpr) {
          sourceDim = i;
          break;
        }
      }
      if (sourceDim < 0) {
        return {indexVec, map, false};
      }

      // Add the scalar to the corresponding base offset.
      OpOperand &baseOffset = op.getOffsetsMutable()[sourceDim];

      AffineExpr d0, d1;
      bindDims(op.getContext(), d0, d1);

      Value newOffset = affine::makeComposedAffineApply(
                            rewriter, op.getLoc(), d0 + d1,
                            ArrayRef<OpFoldResult>{baseOffset.get(), scalarSrc})
                            .getResult();
      baseOffset.set(newOffset);

      // Replace index vec with the non-broadcast addend.
      return {remaining, map, true};
    };

    Value newVal = foldTransferGatherIndexVecs(op, indexVecFolder);

    if (!newVal) {
      return failure();
    }

    return success();
  }
};

struct FoldContiguousGatherToTransferRead final
    : OpRewritePattern<TransferGatherOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(TransferGatherOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getIndexVecs().empty()) {
      return failure();
    }

    AffineMap permutationMap = op.getPermutationMap();

    Value mask = op.getMask();
    if (mask) {
      // First, apply the mask indexing map if it's not identity.
      AffineMap maskMap = op.getIndexingMapsArray().back();
      ArrayRef<int64_t> targetShape = op.getType().getShape();
      if (!maskMap.isIdentity()) {
        mask = applyTransformMapToVector(rewriter, op.getLoc(), mask, maskMap,
                                         targetShape);
      }
      // Then, compress the mask to match transfer_read's expected mask type
      // (which drops broadcast dims from the permutation map).
      auto expectedMaskType =
          vector::inferTransferOpMaskType(op.getType(), permutationMap);
      if (mask.getType() != expectedMaskType) {
        mask = vector::ShapeCastOp::create(rewriter, op.getLoc(),
                                           expectedMaskType, mask);
      }
    }

    SmallVector<bool> inBoundsVec(op.getType().getRank(), true);
    ArrayAttr inBounds = rewriter.getBoolArrayAttr(inBoundsVec);

    rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
        op, op.getType(), op.getBase(), op.getOffsets(), permutationMap,
        op.getPadding(), mask, inBounds);
    return success();
  };
};

void TransferGatherOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *ctx) {
  results.add<FoldSingleElementIndexVec, FoldIndexVecAddBroadcast,
              FoldContiguousGatherToTransferRead>(ctx);
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyYieldForArgCompare(YieldOp yieldOp,
                                              ArgCompareOp argCompareOp) {
  unsigned numOperands = yieldOp.getNumOperands();
  if (numOperands != 1) {
    return yieldOp.emitOpError("expected 1 yield operand, but got ")
           << numOperands;
  }

  Type yieldType = yieldOp.getOperand(0).getType();
  if (!yieldType.isInteger(1)) {
    return yieldOp.emitOpError(
               "expected yield operand to have type i1, but got ")
           << yieldType;
  }

  return success();
}

LogicalResult YieldOp::verify() {
  // ParentOneOf<["ArgCompareOp"]> ODS trait ensures parent is ArgCompareOp.
  auto argCompareOp = cast<ArgCompareOp>((*this)->getParentOp());
  return verifyYieldForArgCompare(*this, argCompareOp);
}

//===----------------------------------------------------------------------===//
// ArgCompareOp
//===----------------------------------------------------------------------===//

LogicalResult ArgCompareOp::verify() {
  Operation *op = getOperation();

  VectorType inputType = getInputValueType();
  VectorType initValueType = getInitValueType();
  VectorType initIndexType = getInitIndexType();

  int64_t inputRank = inputType.getRank();
  int64_t initValueRank = initValueType.getRank();
  int64_t initIndexRank = initIndexType.getRank();
  int64_t dimension = getDimension();

  if (dimension < 0 || dimension >= inputRank) {
    return op->emitOpError("dimension ")
           << dimension << " is out of range [0, " << inputRank << ")";
  }

  if (initValueRank != inputRank - 1) {
    return op->emitOpError("init value rank (")
           << initValueRank << ") must be input rank - 1 (" << (inputRank - 1)
           << ")";
  }

  if (initIndexRank != inputRank - 1) {
    return op->emitOpError("init index rank (")
           << initIndexRank << ") must be input rank - 1 (" << (inputRank - 1)
           << ")";
  }

  SmallVector<int64_t> expectedShape;
  for (int64_t i = 0; i < inputRank; ++i) {
    if (i != dimension) {
      expectedShape.push_back(inputType.getDimSize(i));
    }
  }

  ArrayRef<int64_t> initValueShape = initValueType.getShape();
  if (expectedShape != initValueShape) {
    return op->emitOpError(
               "init value shape must match input shape with reduction "
               "dimension removed. ")
           << "Expected: " << llvm::interleaved_array(expectedShape)
           << ", but got: " << llvm::interleaved_array(initValueShape);
  }

  ArrayRef<int64_t> initIndexShape = initIndexType.getShape();
  if (expectedShape != initIndexShape) {
    return op->emitOpError(
               "init index shape must match input shape with reduction "
               "dimension removed. ")
           << "Expected: " << llvm::interleaved_array(expectedShape)
           << ", but got: " << llvm::interleaved_array(initIndexShape);
  }

  Type initIndexElementType = initIndexType.getElementType();
  if (!isa<IntegerType, IndexType>(initIndexElementType)) {
    return op->emitOpError(
               "init index must have integer or index element type, but got ")
           << initIndexElementType;
  }

  if (hasExplicitIndexInput()) {
    VectorType inputIndexType = getInputIndexType();
    ArrayRef<int64_t> inputIndexShape = inputIndexType.getShape();
    ArrayRef<int64_t> inputValueShape = inputType.getShape();

    if (inputIndexShape != inputValueShape) {
      return op->emitOpError(
                 "explicit-index mode: value and index inputs must have the "
                 "same shape. ")
             << "Value shape: " << llvm::interleaved_array(inputValueShape)
             << ", index shape: " << llvm::interleaved_array(inputIndexShape);
    }

    Type inputIndexElementType = getInputIndexElementType();
    Type initIndexElementType = initIndexType.getElementType();

    if (!isa<IntegerType, IndexType>(inputIndexElementType)) {
      return op->emitOpError("explicit-index mode: index input must have "
                             "integer or index element type, but got ")
             << inputIndexElementType;
    }

    if (inputIndexElementType != initIndexElementType) {
      return op->emitOpError(
                 "explicit-index mode: input and init index element types "
                 "must match. ")
             << "Input index type: " << inputIndexElementType
             << ", init index type: " << initIndexElementType;
    }

    if (getIndexBase()) {
      return op->emitOpError(
          "index_base must not be used with explicit indices");
    }
  }

  // Region structure is enforced by ODS (SizedRegion<1> and
  // SingleBlockImplicitTerminator), so we can directly access it.
  Block &block = getRegion().front();
  if (block.getNumArguments() != 2) {
    return op->emitOpError("comparator region must have exactly 2 arguments");
  }

  Type inputElementType = inputType.getElementType();
  Type arg0Type = block.getArgument(0).getType();
  Type arg1Type = block.getArgument(1).getType();

  if (arg0Type != inputElementType || arg1Type != inputElementType) {
    return op->emitOpError(
               "comparator arguments must match input value element type. ")
           << "Expected: " << inputElementType << ", but got: " << arg0Type
           << " and " << arg1Type;
  }

  // Since ArgCompareOp is marked Pure, all operations in the comparator must
  // also be pure.
  for (Operation &op : block.getOperations()) {
    if (!isPure(&op)) {
      return op.emitOpError(
          "comparator region must contain only pure operations");
    }
  }

  return success();
}

// clang-format off
#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.cpp.inc" // IWYU pragma: keep
// clang-format on
