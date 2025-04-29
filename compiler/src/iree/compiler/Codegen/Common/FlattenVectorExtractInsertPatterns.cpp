// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===-- FlattenVectorExtractInsertPatterns.cpp --------------------===//
//
//===--------------------------------------------------------------===//

#include <numeric>
#include "iree/compiler/Codegen/Common/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_TESTFLATTENVECTOREXTRACTINSERTPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

/// Return true if the strided slice `small` is strided within `large`.
bool hasStride(VectorType small, VectorType large) {

  int64_t smallRank = small.getRank();
  int64_t largeRank = large.getRank();
  int64_t delta = largeRank - smallRank;
  assert(delta >= 0 && "rank of small assumed no greater than large");

  assert(small.getNumElements() <= large.getNumElements() &&
         "number of elements of small assumed no greater than large");

  int firstNonOneDim = 0;
  while (firstNonOneDim < smallRank && small.getDimSize(firstNonOneDim) == 1)
    ++firstNonOneDim;

  // Check if any subsequent dimensions are not equal.
  ArrayRef<int64_t> largeSubShape = large.getShape().drop_front(delta);
  for (int i = firstNonOneDim + 1; i < smallRank; ++i) {
    if (largeSubShape[i] != small.getDimSize(i)) {
      return true;
    }
  }
  return false;
}

template <typename TOp>
bool stridesAllOne(TOp op) {
  for (Attribute stride : op.getStrides()) {
    if (!isConstantIntValue(stride, 1))
      return false;
  }
  return true;
}

/// Return true if `transpose` does not permute a pair of dimensions that are
/// both not of size 1. By `order preserving` we mean that the flattened
/// versions of the input and output vectors are (numerically) identical.
/// In other words `transpose` is effectively a shape cast.
bool isOrderPreserving(vector::TransposeOp transpose) {
  ArrayRef<int64_t> permutation = transpose.getPermutation();
  ArrayRef<int64_t> inShape = transpose.getSourceVectorType().getShape();
  int64_t current = 0;
  for (auto p : permutation) {
    if (inShape[p] != 1) {
      if (p < current) {
        return false;
      }
      current = p;
    }
  }
  return true;
}

/// If `ndIndex` is the index in the n-dimensional array of shape `shape`, get
/// the corresponding index into the flattened array.
int64_t getIndexInFlattened(ArrayRef<int64_t> ndIndex,
                            ArrayRef<int64_t> shape) {
  assert(ndIndex.size() == shape.size() &&
         "ndIndex and shape assumed to have the same size");
  int64_t index = 0;
  int64_t stride = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    index += ndIndex[i] * stride;
    stride *= shape[i];
  }
  return index;
}

/// Convert OpFoldResults into a integers, failing when an OpFoldResult
/// is not an Attribute (unless the dimension in shape is 1, in which case
/// the offset is 0 irrespective).
FailureOr<SmallVector<int64_t>>
getIntegerOffsetsFromFoldResults(ArrayRef<OpFoldResult> offsetFoldResults,
                                 ArrayRef<int64_t> shape) {
  assert(shape.size() >= offsetFoldResults.size() &&
         "offsets assumed not be be higher rank than shape");
  unsigned deltaRank = shape.size() - offsetFoldResults.size();
  SmallVector<int64_t> offsets;
  offsets.reserve(offsetFoldResults.size());
  for (auto [offsetFoldResult, dimSize] :
       llvm::zip(offsetFoldResults, shape.drop_back(deltaRank))) {
    if (dimSize == 1) {
      offsets.push_back(0);
    } else if (auto offsetAttr = dyn_cast<Attribute>(offsetFoldResult)) {
      offsets.push_back(cast<IntegerAttr>(offsetAttr).getInt());
    } else {
      return failure();
    }
  }
  while (offsets.size() < shape.size()) {
    offsets.push_back(0);
  }
  return offsets;
}

/// Convert an ArrayAttr to a vector of integers, failing when an attribute
/// cannot be converted to an integer (unless the dimension in shape is 1,
/// in which case the offset is 0 irrespective).
FailureOr<SmallVector<int64_t>>
getIntegerOffsetsFromArrayAttr(ArrayAttr offsetAttrs, ArrayRef<int64_t> shape) {
  if (!offsetAttrs)
    return failure();
  assert(offsetAttrs.size() == shape.size() &&
         "offsets and shape assumed to have same rank");
  SmallVector<int64_t> offsets;
  offsets.reserve(offsetAttrs.size());
  for (auto [dimSize, attr] : llvm::zip(shape, offsetAttrs)) {
    if (dimSize == 1) {
      offsets.push_back(0);
    } else if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
      offsets.push_back(intAttr.getInt());
    } else {
      return failure();
    }
  }
  return offsets;
}

FailureOr<int64_t> getNumberOfElements(Type type) {
  if (type.isIntOrIndexOrFloat()) {
    return 1;
  } else if (auto shapedType = dyn_cast<ShapedType>(type)) {
    return shapedType.getNumElements();
  }
  return failure();
}

/// Return an equivalent strided slice to inserting `small` into `large`
/// starting at `offsets`. The result is a tuple of three vectors: 1) The shape
/// of the new small vector. 2) The shape of the new large vector. 3) The
/// offsets of the new large vector.
std::array<SmallVector<int64_t>, 3>
getCollapsedStridedSliceShape(ArrayRef<int64_t> small, ArrayRef<int64_t> large,
                              ArrayRef<int64_t> offsets) {

  // The total number of elements in the small (large, respectively) vector.
  int64_t tSmall = std::accumulate(small.begin(), small.end(), 1,
                                   std::multiplies<int64_t>());
  int64_t tLarge = std::accumulate(large.begin(), large.end(), 1,
                                   std::multiplies<int64_t>());
  assert((tLarge >= tSmall && large.size() >= small.size()) &&
         "confusion of small vs large");
  unsigned delta = large.size() - small.size();

  // The number of cumulative elements from the back currently visited in the
  // small (large, respectively) vector.
  int64_t nSmall = 1;
  int64_t nLarge = 1;

  // The number of cumulative elements from back currently visited within the
  // current collapse group in the small (large, respectively) vector.
  int64_t cSmall = 1;
  int64_t cLarge = 1;

  SmallVector<int64_t> newSmall, newLarge, newOffsets;
  if (large.size() == 0) {
    return {newSmall, newLarge, newOffsets};
  }

  // The offset assigned to the current collapse group.
  int64_t cOff = 0;

  unsigned index = large.size() - 1;
  while (nLarge < tLarge) {
    assert(cSmall <= nSmall && nSmall <= tSmall && //
           cLarge <= nLarge && nLarge <= tLarge &&
           "confusion in element accumulation");
    cOff += offsets[index] * cLarge;
    if (nSmall < tSmall) {
      cSmall *= small[index - delta];
      nSmall *= small[index - delta];
    }
    cLarge *= large[index];
    nLarge *= large[index];
    if ((nSmall < tSmall) && (large[index] != small[index - delta])) {
      newSmall.push_back(cSmall);
      newLarge.push_back(cLarge);
      newOffsets.push_back(cOff);
      cSmall = 1;
      cLarge = 1;
      cOff = 0;
    }
    --index;
  }
  newSmall.push_back(cSmall);
  newLarge.push_back(cLarge);
  newOffsets.push_back(cOff);
  std::reverse(newSmall.begin(), newSmall.end());
  std::reverse(newLarge.begin(), newLarge.end());
  std::reverse(newOffsets.begin(), newOffsets.end());
  return {newSmall, newLarge, newOffsets};
}

SmallVector<int64_t>
getFlattenedStridedSliceIndices(ArrayRef<int64_t> small,
                                ArrayRef<int64_t> large,
                                ArrayRef<int64_t> offsets) {
  // Simplify the problem:
  auto [collapsedSmall, collapsedLarge, collapsedOffsets] =
      getCollapsedStridedSliceShape(small, large, offsets);
  SmallVector<int64_t> indices{0};
  SmallVector<int64_t> nxtIndices;
  int64_t stride = 1;
  for (int i = collapsedSmall.size() - 1; i >= 0; --i) {
    auto currentSize = indices.size();
    auto nxtSize = currentSize * collapsedSmall[i];
    nxtIndices.resize(nxtSize);
    int64_t *base = nxtIndices.begin();
    int64_t offset = collapsedOffsets[i] * stride;
    for (int j = 0; j < collapsedSmall[i]; ++j) {
      for (uint64_t k = 0; k < currentSize; ++k) {
        base[k] = indices[k] + offset;
      }
      offset += stride;
      base += currentSize;
    }
    stride *= collapsedLarge[i];
    std::swap(indices, nxtIndices);
    nxtIndices.clear();
  }
  return indices;
}

/// Convert a vector.insert_strided_slice op into a vector.shuffle op.
LogicalResult
insertStridedSliceToRankOneShuffle(vector::InsertStridedSliceOp insertOp,
                                   PatternRewriter &rewriter) {

  // return failure();

  if (!stridesAllOne(insertOp))
    return failure();

  VectorType outType = insertOp.getType();
  Type elementType = outType.getElementType();
  ArrayRef<int64_t> outShape = outType.getShape();
  int64_t nOutElements = outType.getNumElements();
  VectorType flatOutType = VectorType::get({nOutElements}, elementType);

  TypedValue<VectorType> input = insertOp.getValueToStore();
  VectorType inType = input.getType();
  ArrayRef<int64_t> inShape = inType.getShape();
  int64_t nInElements = inType.getNumElements();
  VectorType flatInType = VectorType::get({nInElements}, elementType);

  auto maybeIntOffsets =
      getIntegerOffsetsFromArrayAttr(insertOp.getOffsets(), outShape);
  if (failed(maybeIntOffsets))
    return failure();
  SmallVector<int64_t> offsets = std::move(maybeIntOffsets.value());

  // Initialize the indices for the shuffle.
  SmallVector<int64_t> sliceIndices =
      getFlattenedStridedSliceIndices(inShape, outShape, offsets);
  SmallVector<int64_t> indices(nOutElements, 0);
  std::iota(indices.begin(), indices.end(), 0);
  for (auto [index, sliceIndex] : llvm::enumerate(sliceIndices)) {
    indices[sliceIndex] = index + nOutElements;
  }

  // Flatten the operands, create a shuffle op, upcast the result.
  Location loc = insertOp.getLoc();
  auto flatToInsert =
      rewriter.createOrFold<vector::ShapeCastOp>(loc, flatInType, input);
  auto flatToInsertInto = rewriter.createOrFold<vector::ShapeCastOp>(
      loc, flatOutType, insertOp.getDest());
  auto replacement = rewriter.create<vector::ShuffleOp>(
      loc, flatOutType, flatToInsertInto, flatToInsert, indices);
  auto upCast =
      rewriter.createOrFold<vector::ShapeCastOp>(loc, outType, replacement);
  rewriter.replaceOp(insertOp, upCast);
  return success();
}

/// Convert a vector.extract_strided_slice op into a vector.shuffle op.
LogicalResult
extractStridedSliceToRankOneShuffle(vector::ExtractStridedSliceOp extractOp,
                                    PatternRewriter &rewriter) {

  return failure();

  if (!stridesAllOne(extractOp))
    return failure();

  VectorType inType = extractOp.getSourceVectorType();
  ArrayRef<int64_t> inputShape = inType.getShape();
  VectorType flatInType =
      VectorType::get({inType.getNumElements()}, inType.getElementType());

  VectorType outputType = extractOp.getType();
  VectorType flatOutType = VectorType::get({outputType.getNumElements()},
                                           outputType.getElementType());

  auto maybeIntOffsets =
      getIntegerOffsetsFromArrayAttr(extractOp.getOffsets(), inputShape);
  if (failed(maybeIntOffsets)) {
    return failure();
  }
  SmallVector<int64_t> offsets = std::move(maybeIntOffsets.value());
  SmallVector<int64_t> indices = getFlattenedStridedSliceIndices(
      outputType.getShape(), inputShape, offsets);

  auto flatIn = rewriter.createOrFold<vector::ShapeCastOp>(
      extractOp.getLoc(), flatInType, extractOp.getOperand());

  auto replacement = rewriter.create<vector::ShuffleOp>(
      extractOp.getLoc(), flatOutType, flatIn, flatIn, indices);

  auto upCast = rewriter.createOrFold<vector::ShapeCastOp>(
      extractOp.getLoc(), outputType, replacement);
  rewriter.replaceOp(extractOp, upCast);
  return success();
}

/// A pattern that matches on elementwise operations that are not rank-1, and
/// flattens the inputs with shape_cast operations. The elementwise operation is
/// then performed in rank-1, and a shape_cast returns to the original shape.
struct ElementwiseToRankOne : public RewritePattern {
  ElementwiseToRankOne(MLIRContext *context, PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {

    // Checks that this pattern should be applied.
    if (!OpTrait::hasElementwiseMappableTraits(op))
      return failure();
    if (op->getNumResults() != 1)
      return failure();
    auto outType = dyn_cast<VectorType>(op->getResultTypes()[0]);
    if (!outType)
      return rewriter.notifyMatchFailure(op, "result is not a vector");
    if (outType.getRank() <= 1)
      return rewriter.notifyMatchFailure(op,
                                         "result is already rank-0 or rank-1");
    ArrayRef<int64_t> outShape = outType.getShape();
    for (OpOperand &operand : op->getOpOperands()) {
      if (auto operandType = dyn_cast<VectorType>(operand.get().getType())) {
        if (operandType.getShape() != outShape) {
          return rewriter.notifyMatchFailure(
              op, "operand and result have different shapes");
        }
      }
    }

    // Create flattened operands. Non-vector operands are used as they are.
    SmallVector<int64_t> flatShape{outType.getNumElements()};
    SmallVector<Value> operands;
    operands.reserve(op->getNumOperands());
    for (OpOperand &operand : op->getOpOperands()) {
      auto operandType = dyn_cast<VectorType>(operand.get().getType());
      if (!operandType) {
        operands.push_back(operand.get());
      } else {
        VectorType flatOperandType =
            VectorType::get(flatShape, operandType.getElementType());
        auto shapeCast = rewriter.create<vector::ShapeCastOp>(
            op->getLoc(), flatOperandType, operand.get());
        operands.push_back(shapeCast);
      }
    }

    // Create clone of the elementwise operation that uses the flattened
    // (rank-1) operands, and then reshape back to the original shape.
    VectorType flatOutType =
        VectorType::get(flatShape, outType.getElementType());
    auto clone = rewriter.create(op->getLoc(), op->getName().getIdentifier(),
                                 operands, flatOutType, op->getAttrs());
    auto reshapedOut = rewriter.create<vector::ShapeCastOp>(
        op->getLoc(), outType, clone->getResult(0));
    rewriter.replaceOp(op, reshapedOut.getResult());

    return success();
  }
};

/// Convert a vector.extract op with input rank > 1, to an operation with input
/// of rank 1 and output of rank <= 1. Two lowering cases:
///
/// 1) If the result of the vector.extract is a scalar, convert it to a
///    vector.extract on a rank-1 input which still outputs a scalar.
///
/// 2) Otherwise, convert to an extract_strided_slice op on a vector of rank-1.
class ExtractOpToRankOne final : public OpRewritePattern<vector::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {

    // The (large) input vector and type.
    TypedValue<VectorType> input = extractOp.getVector();
    VectorType inType = input.getType();
    if (inType.getRank() == 1)
      return failure();

    // The (small) output vector and type.
    Type outType = extractOp.getType();
    FailureOr<int64_t> maybeNumberElementsOut = getNumberOfElements(outType);
    if (failed(maybeNumberElementsOut)) {
      return failure();
    }
    int64_t numberElementsOut = maybeNumberElementsOut.value();

    SmallVector<OpFoldResult> offsets = extractOp.getMixedPosition();
    auto maybeIntOffsets =
        getIntegerOffsetsFromFoldResults(offsets, inType.getShape());
    if (failed(maybeIntOffsets)) {
      return failure();
    }
    auto intOffsets = maybeIntOffsets.value();
    int64_t globalOffset = getIndexInFlattened(intOffsets, inType.getShape());

    VectorType flatInType =
        VectorType::get({inType.getNumElements()}, inType.getElementType());
    Location loc = extractOp.getLoc();

    Value flatIn =
        rewriter.createOrFold<vector::ShapeCastOp>(loc, flatInType, input);

    // Case 1 described above:
    if (outType.isIntOrIndexOrFloat()) {
      Value flattened = rewriter.create<vector::ExtractOp>(
          loc, flatIn, SmallVector<int64_t>{globalOffset});
      rewriter.replaceOp(extractOp, flattened);
      return success();
    }

    // Case 2 described above:
    auto strided = rewriter.create<vector::ExtractStridedSliceOp>(
        loc, flatIn, SmallVector<int64_t>{globalOffset},
        SmallVector<int64_t>{numberElementsOut}, SmallVector<int64_t>{1});
    Value upCast =
        rewriter.createOrFold<vector::ShapeCastOp>(loc, outType, strided);
    rewriter.replaceOp(extractOp, upCast);
    return success();
  }
};

/// Convert vector.insert where the destination is rank > 1. Two cases:
///
/// 1) If the source to insert is a scalar, convert to a vector.insert op
///    where the destination is rank-1.
///
/// 2) Otherwise, convert to a vector.insert_strided_slice op into a vector of
///    rank-1.
class InsertOpToRankOne final : public OpRewritePattern<vector::InsertOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::InsertOp insertOp,
                                PatternRewriter &rewriter) const override {

    // The large operand (gets inserted into)
    TypedValue<VectorType> large = insertOp.getDest();
    VectorType largeType = large.getType();
    if (largeType.getRank() == 1) {
      return failure();
    }

    // The small operand (that gets inserted into the large operand)
    Value small = insertOp.getValueToStore();
    Type smallType = insertOp.getValueToStoreType();
    auto maybeNumberElementsSmall = getNumberOfElements(smallType);
    if (failed(maybeNumberElementsSmall)) {
      return failure();
    }
    int64_t numberElementsSmall = maybeNumberElementsSmall.value();

    SmallVector<OpFoldResult> positions = insertOp.getMixedPosition();
    auto maybeIntOffsets =
        getIntegerOffsetsFromFoldResults(positions, largeType.getShape());
    if (failed(maybeIntOffsets)) {
      return failure();
    }
    auto intOffsets = maybeIntOffsets.value();
    int64_t globalOffset =
        getIndexInFlattened(intOffsets, largeType.getShape());

    Location loc = insertOp.getLoc();
    VectorType flatLargeType = VectorType::get({largeType.getNumElements()},
                                               largeType.getElementType());
    auto flatLarge =
        rewriter.createOrFold<vector::ShapeCastOp>(loc, flatLargeType, large);

    Value updated = [&]() -> Value {
      // Case 1 described above:
      if (smallType.isSignlessIntOrFloat()) {
        return rewriter.create<vector::InsertOp>(
            loc, small, flatLarge, SmallVector<int64_t>{globalOffset});
      }

      // Case 2 described above:
      VectorType flatSmallType =
          VectorType::get({numberElementsSmall}, largeType.getElementType());
      auto flatSmall =
          rewriter.createOrFold<vector::ShapeCastOp>(loc, flatSmallType, small);
      return rewriter.create<vector::InsertStridedSliceOp>(
          insertOp.getLoc(), flatSmall, flatLarge,
          SmallVector<int64_t>{globalOffset}, SmallVector<int64_t>{1});
    }();

    Value replacement = rewriter.createOrFold<vector::ShapeCastOp>(
        insertOp.getLoc(), largeType, updated);

    rewriter.replaceOp(insertOp, replacement);
    return success();
  }
};

/// Pattern to convert a vector.extract_strided_slice into a vector.shuffle
class ExtractStridedSliceToRankOneShuffle final
    : public OpRewritePattern<vector::ExtractStridedSliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::ExtractStridedSliceOp extractOp,
                                PatternRewriter &rewriter) const override {
    return extractStridedSliceToRankOneShuffle(extractOp, rewriter);
  }
};

/// Pattern to convert a vector.insert_strided_slice into a vector.shuffle
class InsertStridedSliceToRankOneShuffle final
    : public OpRewritePattern<vector::InsertStridedSliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::InsertStridedSliceOp insertOp,
                                PatternRewriter &rewriter) const override {
    return insertStridedSliceToRankOneShuffle(insertOp, rewriter);
  }
};

/// This pattern converts a vector.extract_strided_slice into a new
/// vector.extract_strided_slice where the operand and result of the new
/// vector.extract_strided_slice have ranks that are as low as possible.
///
/// If the original vector.extract_strided_slice is a contiguous slice of
/// a vector, then the new vector.extract_strided_slice will have rank-1
/// operand and result. Otherwise additional dimensions will remain in the
/// new operand and result.
class CollapseExtractStridedSliceDims final
    : public OpRewritePattern<vector::ExtractStridedSliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::ExtractStridedSliceOp extractOp,
                                PatternRewriter &rewriter) const override {

    if (!stridesAllOne(extractOp))
      return failure();

    VectorType outType = extractOp.getType();
    VectorType inType = extractOp.getSourceVectorType();

    auto maybeIntOffsets = getIntegerOffsetsFromArrayAttr(
        extractOp.getOffsets(), inType.getShape());
    if (failed(maybeIntOffsets)) {
      return failure();
    }
    auto intOffsets = maybeIntOffsets.value();
    auto [collapsedOutShape, collapsedInShape, collapsedOffsets] =
        getCollapsedStridedSliceShape(outType.getShape(), inType.getShape(),
                                      intOffsets);

    bool rankUnchanged = (collapsedInShape.size() == inType.getRank()) &&
                         (collapsedOutShape.size() == outType.getRank());

    if (rankUnchanged)
      return failure();

    VectorType flatInType =
        VectorType::get(collapsedInShape, inType.getElementType());

    auto flatIn = rewriter.createOrFold<vector::ShapeCastOp>(
        extractOp.getLoc(), flatInType, extractOp.getOperand());

    auto replacement = rewriter.create<vector::ExtractStridedSliceOp>(
        extractOp.getLoc(), flatIn, collapsedOffsets, collapsedOutShape,
        SmallVector<int64_t>(collapsedOffsets.size(), 1));

    auto flatOut = rewriter.createOrFold<vector::ShapeCastOp>(
        extractOp.getLoc(), outType, replacement);

    rewriter.replaceOp(extractOp, flatOut);

    return success();
  }
};

/// A pattern to convert an extract_strided_slice into either a rank-1
/// extract_strided_slice or a shuffle.
///
/// If `convertAllToShuffle` is true, then all extract_strided_slices
/// will be converted to shuffles. Otherwise, only those that are not
/// contiguous will be. In other words, if `convertAllToShuffle` is false,
/// then the preference is for a rank-1 extract_strided_slice to be created
/// when possible.
class ExtractStridedSliceToRankOne final
    : public OpRewritePattern<vector::ExtractStridedSliceOp> {
  using OpRewritePattern::OpRewritePattern;

public:
  ExtractStridedSliceToRankOne(MLIRContext *context, bool allToShuffle,
                               PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), convertAllToShuffle(allToShuffle) {}

  LogicalResult matchAndRewrite(vector::ExtractStridedSliceOp extractOp,
                                PatternRewriter &rewriter) const override {

    if (!stridesAllOne(extractOp))
      return failure();

    VectorType outType = extractOp.getType();
    VectorType inType = extractOp.getSourceVectorType();

    if (!convertAllToShuffle && outType.getRank() == 1 &&
        inType.getRank() == 1) {
      return failure();
    }

    auto maybeIntOffsets = getIntegerOffsetsFromArrayAttr(
        extractOp.getOffsets(), inType.getShape());
    if (failed(maybeIntOffsets)) {
      return failure();
    }
    auto intOffsets = maybeIntOffsets.value();
    auto [collapsedOutShape, collapsedInShape, collapsedOffsets] =
        getCollapsedStridedSliceShape(outType.getShape(), inType.getShape(),
                                      intOffsets);

    bool rankUnchanged = (collapsedInShape.size() == inType.getRank()) &&
                         (collapsedOutShape.size() == outType.getRank());

    if (rankUnchanged)
      return extractStridedSliceToRankOneShuffle(extractOp, rewriter);

    VectorType flatInType =
        VectorType::get(collapsedInShape, inType.getElementType());

    auto flatIn = rewriter.createOrFold<vector::ShapeCastOp>(
        extractOp.getLoc(), flatInType, extractOp.getOperand());

    auto replacement = rewriter.create<vector::ExtractStridedSliceOp>(
        extractOp.getLoc(), flatIn, collapsedOffsets, collapsedOutShape,
        SmallVector<int64_t>(collapsedOffsets.size(), 1));

    auto flatOut = rewriter.createOrFold<vector::ShapeCastOp>(
        extractOp.getLoc(), outType, replacement);

    rewriter.replaceOp(extractOp, flatOut);

    return success();
  }

private:
  bool convertAllToShuffle;
};

class InsertStridedSliceToRankOne final
    : public OpRewritePattern<vector::InsertStridedSliceOp> {
  // using OpRewritePattern::OpRewritePattern;

public:
  InsertStridedSliceToRankOne(MLIRContext *context, bool allToShuffle,
                              PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), convertAllToShuffle(allToShuffle) {}

private:
  bool convertAllToShuffle;

  LogicalResult matchAndRewrite(vector::InsertStridedSliceOp insertOp,
                                PatternRewriter &rewriter) const override {

    if (!stridesAllOne(insertOp))
      return failure();

    VectorType outType = insertOp.getType();

    auto maybeIntOffsets = getIntegerOffsetsFromArrayAttr(insertOp.getOffsets(),
                                                          outType.getShape());
    if (failed(maybeIntOffsets))
      return failure();
    auto intOffsets = maybeIntOffsets.value();

    TypedValue<VectorType> input = insertOp.getValueToStore();
    VectorType inType = input.getType();
    auto [collapsedInShape, collapsedOutShape, collapsedOffsets] =
        getCollapsedStridedSliceShape(inType.getShape(), outType.getShape(),
                                      intOffsets);

    if (!convertAllToShuffle && outType.getRank() == 1 &&
        inType.getRank() == 1) {
      return failure();
    }

    Location loc = insertOp.getLoc();
    bool rankUnchanged = (collapsedInShape.size() == inType.getRank()) &&
                         (collapsedOutShape.size() == outType.getRank());
    if (rankUnchanged)
      return insertStridedSliceToRankOneShuffle(insertOp, rewriter);

    VectorType flatInType =
        VectorType::get(collapsedInShape, inType.getElementType());

    VectorType flatOutType =
        VectorType::get(collapsedOutShape, outType.getElementType());

    Value flatValueToInsert =
        rewriter.createOrFold<vector::ShapeCastOp>(loc, flatInType, input);

    Value flatValueToInsertInto = rewriter.createOrFold<vector::ShapeCastOp>(
        loc, flatOutType, insertOp.getDest());

    SmallVector<int64_t> flatStrides(collapsedOffsets.size(), 1);

    Value replacement = rewriter.create<vector::InsertStridedSliceOp>(
        loc, flatValueToInsert, flatValueToInsertInto, collapsedOffsets,
        flatStrides);

    Value upCast =
        rewriter.createOrFold<vector::ShapeCastOp>(loc, outType, replacement);

    rewriter.replaceOp(insertOp, upCast);

    return success();
  }
};

struct TransposeToShapeCast final
    : public OpRewritePattern<vector::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    if (!isOrderPreserving(transposeOp))
      return failure();

    Value shapeCast = rewriter.create<vector::ShapeCastOp>(
        transposeOp.getLoc(), transposeOp.getType(), transposeOp.getVector());

    rewriter.replaceOp(transposeOp, shapeCast);
    return success();
  }
};

// If the number of elements in the broadcast is unchanged, convert to shape
// cast
struct BroadcastToShapeCast final
    : public OpRewritePattern<vector::BroadcastOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::BroadcastOp broadcastOp,
                                PatternRewriter &rewriter) const override {
    auto sourceType = dyn_cast<VectorType>(broadcastOp.getSourceType());
    if (!sourceType)
      return failure();

    VectorType outType = broadcastOp.getType();
    if (sourceType.getNumElements() != outType.getNumElements())
      return failure();

    Value shapeCast = rewriter.create<vector::ShapeCastOp>(
        broadcastOp.getLoc(), outType, broadcastOp.getSource());

    rewriter.replaceOp(broadcastOp, shapeCast);
    return success();
  }
};

struct ExtractToShapeCast final : public OpRewritePattern<vector::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {
    VectorType sourceType = extractOp.getSourceVectorType();
    VectorType outType = dyn_cast<VectorType>(extractOp.getType());
    if (!outType)
      return failure();

    if (sourceType.getNumElements() != outType.getNumElements())
      return failure();

    Value shapeCast = rewriter.create<vector::ShapeCastOp>(
        extractOp.getLoc(), outType, extractOp.getVector());

    rewriter.replaceOp(extractOp, shapeCast);
    return success();
  }
};

// Because we disable folding, we manually fold shape_casts.
struct ShapeCastOpFold final : public OpRewritePattern<vector::ShapeCastOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::ShapeCastOp shapeCastOp,
                                PatternRewriter &rewriter) const override {

    auto sourceShapeCastOp = dyn_cast_or_null<vector::ShapeCastOp>(
        shapeCastOp.getSource().getDefiningOp());

    if (!sourceShapeCastOp)
      return failure();

    if (sourceShapeCastOp.getSourceVectorType() == shapeCastOp.getType()) {
      rewriter.replaceOp(shapeCastOp, sourceShapeCastOp.getSource());
      return success();
    }

    // Check if the defining op is also a shape_cast:
    shapeCastOp.setOperand(sourceShapeCastOp.getSource());
    return success();
  }
};

struct ExtractStridedSliceToShapeCast final
    : public OpRewritePattern<vector::ExtractStridedSliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::ExtractStridedSliceOp extractOp,
                                PatternRewriter &rewriter) const override {
    VectorType sourceType = extractOp.getSourceVectorType();
    VectorType outType = extractOp.getType();
    if (sourceType.getNumElements() != outType.getNumElements()) {
      return failure();
    }
    Value shapeCast = rewriter.create<vector::ShapeCastOp>(
        extractOp.getLoc(), outType, extractOp.getVector());
    rewriter.replaceOp(extractOp, shapeCast);
    return success();
  }
};

struct TestFlattenVectorExtractInsertPass final
    : public impl::TestFlattenVectorExtractInsertPassBase<
          TestFlattenVectorExtractInsertPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    // Folding must be disable when applying these patterns, because the
    // vector.extract op 'folds' extract(shape_cast(x)) -> extract(x), but
    // the pattern ExtractOpToRankOne converts
    // extract(x) to extract(shape_cast(x)) in some cases.
    GreedyRewriteConfig config;
    config.fold = false;
    populateFlattenVectorExtractInsertPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      return signalPassFailure();
    }
  }
};

} // namespace

void populateConvertToShapeCastPatterns(RewritePatternSet &patterns,
                                        PatternBenefit benefit) {
  patterns.insert<TransposeToShapeCast, BroadcastToShapeCast,
                  ExtractToShapeCast, ExtractStridedSliceToShapeCast>(
      patterns.getContext(), benefit);
}

// These patterns bla
void populateFlattenVectorExtractInsertPatterns(RewritePatternSet &patterns,
                                                PatternBenefit benefit) {

  populateConvertToShapeCastPatterns(patterns, benefit);

  bool allToShuffle = false;
  patterns.add<ExtractStridedSliceToRankOne, InsertStridedSliceToRankOne>(
      patterns.getContext(), allToShuffle, benefit);

  patterns.insert<ExtractOpToRankOne, InsertOpToRankOne, ShapeCastOpFold,
                  ElementwiseToRankOne>(patterns.getContext(), benefit);
}

} // namespace mlir::iree_compiler
