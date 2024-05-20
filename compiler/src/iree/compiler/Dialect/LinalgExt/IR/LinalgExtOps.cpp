// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include <cstdint>

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/IndexingUtils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <optional>

namespace mlir::iree_compiler::IREE::LinalgExt {

//===----------------------------------------------------------------------===//
// Utils.
//===----------------------------------------------------------------------===//

static Type getComplexElementTypeOrSelf(Type ty) {
  if (auto complex = dyn_cast_or_null<ComplexType>(ty)) {
    return complex.getElementType();
  }
  return ty;
}

static void getEffectsImpl(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    ArrayRef<OpOperand *> inputOperands, MutableOperandRange outputOperands) {
  for (OpOperand *operand : inputOperands) {
    if (!llvm::isa<MemRefType>(operand->get().getType())) {
      continue;
    }
    effects.emplace_back(MemoryEffects::Read::get(), operand,
                         SideEffects::DefaultResource::get());
  }
  for (OpOperand &operand : outputOperands) {
    if (!llvm::isa<MemRefType>(operand.get().getType())) {
      continue;
    }
    effects.emplace_back(MemoryEffects::Read::get(), &operand,
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), &operand,
                         SideEffects::DefaultResource::get());
  }
}

/// Return true if `dimsPos` is invalid. It is invalid when: a) it contains
/// duplicate. b) At least one dimension is out of bound (`dimPos` is >= 0 and <
/// rank). c) the number of elements in `dimsPos` is > than `rank`.
static bool isInvalid(ArrayRef<int64_t> dimsPos, int64_t rank) {
  // early exit.
  if (dimsPos.size() > rank) {
    return true;
  }
  DenseSet<int64_t> uniqued;
  for (int64_t dim : dimsPos) {
    uniqued.insert(dim);
  }
  if (dimsPos.size() != uniqued.size()) {
    return true;
  }
  return llvm::any_of(
      dimsPos, [rank](int64_t dimPos) { return dimPos < 0 || dimPos >= rank; });
}

/// Returns true if the dimension of `sourceShape` is smaller than the dimension
/// of the `limitShape`.
static bool isSmallerThan(ArrayRef<int64_t> sourceShape,
                          ArrayRef<int64_t> limitShape) {
  assert(
      sourceShape.size() == limitShape.size() &&
      "expected source shape rank, and limit of the shape to have same rank");
  return llvm::all_of(llvm::zip_equal(sourceShape, limitShape),
                      [](std::tuple<int64_t, int64_t> it) {
                        int64_t sourceExtent = std::get<0>(it);
                        int64_t limit = std::get<1>(it);
                        return ShapedType::isDynamic(sourceExtent) ||
                               ShapedType::isDynamic(limit) ||
                               sourceExtent <= limit;
                      });
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

LogicalResult ScatterOp::verify() {
  Operation *op = getOperation();
  if (getInputs().size() != 2) {
    return op->emitOpError("expected two input operands");
  }
  if (getOutputs().size() != 1) {
    return op->emitOpError("expected one output operand");
  }
  auto checkDimensionsMatch = [&](ShapedType t1, ShapedType t2, unsigned dim) {
    return t1.getShape()[dim] == t2.getShape()[dim];
  };

  auto indicesType = getIndicesType();
  if (indicesType.getRank() != 2 ||
      !indicesType.getElementType().isInteger(32)) {
    return op->emitOpError(
        "expected indices to be of rank 2 of i32 element type");
  }
  auto indexDepth = getIndexDepth();
  if (ShapedType::isDynamic(indexDepth)) {
    return op->emitOpError("expected index depth is static");
  }

  ArrayRef<int64_t> dimMap = getDimensionMap();
  if (dimMap.size() != indexDepth) {
    return op->emitOpError("invalid number of dimension map entries ");
  }

  auto originalType = getOriginalType();
  if (isInvalid(dimMap, originalType.getRank())) {
    return op->emitOpError("dimension map is invalid");
  }

  // The first dimension of the indices should match the first dimension of the
  // output. They indicate to the number of updates.
  auto updateType = getUpdateType();
  if (updateType.getRank() < 1) {
    return op->emitOpError("expected update value to be at least rank 1");
  }
  if (!checkDimensionsMatch(indicesType, updateType, 0)) {
    return op->emitOpError(
        "mismatch in shape of indices and update value at dim#0");
  }
  if (updateType.getRank() - 1 > originalType.getRank()) {
    return op->emitOpError(
        "update value rank exceeds the rank of the original value");
  }

  // indexDepth + update dims should cover the original dims. The first dim of
  // update is the number of updates.
  if (originalType.getRank() > indexDepth + updateType.getRank() - 1) {
    return op->emitOpError(
        "index depth and update value does not cover rank of original value");
  }

  // Validate the non-indexed update dims cover the full slice size of the
  // original tensor.
  int64_t fullSliceDims = originalType.getRank() - indexDepth;
  for (auto it :
       llvm::zip_equal(llvm::seq<unsigned>(indexDepth, originalType.getRank()),
                       llvm::seq<unsigned>(updateType.getRank() - fullSliceDims,
                                           updateType.getRank()))) {
    int64_t originalDim = std::get<0>(it);
    int64_t updateDim = std::get<1>(it);
    if (!originalType.isDynamicDim(originalDim) &&
        updateType.getDimSize(updateDim) >
            originalType.getDimSize(originalDim)) {
      return op->emitOpError("shape of update value dim#")
             << updateDim << " exceeds original value at dim#" << originalDim;
    }
  }

  // Check that the remaining update indices do not exceed the update length.
  int64_t insertDims = originalType.getRank() - updateType.getRank() + 1;
  for (auto it : llvm::zip_equal(
           llvm::seq<unsigned>(insertDims, indexDepth),
           llvm::seq<unsigned>(1, updateType.getRank() - fullSliceDims))) {
    int64_t originalDim = std::get<0>(it);
    int64_t updateDim = std::get<1>(it);
    if (!originalType.isDynamicDim(originalDim) &&
        updateType.getDimSize(updateDim) >
            originalType.getDimSize(originalDim)) {
      return op->emitOpError("indexed shape of update value dim#")
             << updateDim << " exceeds original value at dim#" << originalDim
             << " " << updateType.getDimSize(updateDim) << " "
             << originalType.getDimSize(originalDim);
    }
  }

  Region &region = this->getRegion();
  Block *body = &region.front();
  if (body->getNumArguments() != 2) {
    return op->emitOpError("expected region to have two arguments");
  }
  Type arg0Type = body->getArgument(0).getType();
  Type arg1Type = body->getArgument(1).getType();
  if (!getComplexElementTypeOrSelf(arg0Type).isIntOrFloat() ||
      !getComplexElementTypeOrSelf(arg1Type).isIntOrFloat()) {
    return op->emitOpError(
        "expected region to have scalar argument of integer or float types");
  }
  if (arg0Type != updateType.getElementType()) {
    return op->emitOpError("mismatch in argument 0 of region ")
           << arg0Type << " and element type of update value "
           << updateType.getElementType();
  }
  if (arg1Type != originalType.getElementType()) {
    return op->emitOpError("mismatch in argument 1 of region ")
           << arg1Type << " and element type of original value "
           << originalType.getElementType();
  }
  if (arg0Type != arg1Type) {
    return op->emitOpError("mismatch in region argument types ")
           << arg0Type << " and " << arg1Type;
  }
  auto yieldOp = cast<IREE::LinalgExt::YieldOp>(body->getTerminator());
  if (yieldOp->getNumOperands() != 1) {
    return yieldOp.emitOpError("expected region to yield a single value");
  }
  auto yieldedType = yieldOp->getOperand(0).getType();
  if (yieldedType != arg0Type) {
    return yieldOp.emitOpError("mismatch in type of yielded value ")
           << yieldedType << " and argument of the region " << arg0Type;
  }
  return success();
}

LogicalResult
ScatterOp::reifyResultShapes(OpBuilder &b,
                             ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

SmallVector<AffineMap> ScatterOp::getIndexingMapsForOperands() {
  Builder builder(getContext());
  return {builder.getMultiDimIdentityMap(getUpdateType().getRank()),
          builder.getMultiDimIdentityMap(getIndicesType().getRank()),
          /*output=*/AffineMap(nullptr)};
}

SmallVector<AffineMap> ScatterOp::getIndexingMapsForResults() {
  return {AffineMap(nullptr)};
}

//===----------------------------------------------------------------------===//
// SortOp
//===----------------------------------------------------------------------===//

LogicalResult SortOp::verify() {
  Operation *op = getOperation();
  if (getNumDpsInputs()) {
    return op->emitOpError("does not expect to take any inputs");
  }
  if (getNumDpsInits() == 0) {
    return op->emitOpError("expected at least one `outs` operand");
  }

  Block &block = getRegion().front();
  size_t numOutputs = getNumDpsInits();
  if (block.getNumArguments() != 2 * numOutputs) {
    return op->emitOpError("region block should have ")
           << 2 * numOutputs << " arguments";
  }

  int64_t rank = getOperandRank();
  int sortDim = getDimension();
  if (sortDim < 0 || sortDim >= rank) {
    return op->emitOpError("dimension must be within (0, ") << rank << "]";
  }

  ArrayRef<int64_t> shape = getOperandShape();
  for (auto [index, operand] : llvm::enumerate(getOutputs())) {
    auto operandType = getOperandType(index);
    if (operandType.getRank() != rank) {
      return op->emitOpError("expected operand ")
             << index << " to be rank " << rank << ", same as other operands";
    }
    if (operandType.getShape() != shape) {
      return op->emitOpError("expected operand ")
             << index << " to have same shape as other operands";
    }
    Type elemType = operandType.getElementType();
    for (int i : {2 * index, 2 * index + 1}) {
      Type argType = block.getArgument(i).getType();
      if (argType != elemType) {
        return op->emitOpError("region block argument #")
               << i << " should be of type " << elemType << " but got "
               << argType;
      }
    }
  }

  auto yieldOp = cast<YieldOp>(block.getTerminator());
  if (yieldOp.getNumOperands() != 1) {
    return op->emitOpError("should yield exactly one operand");
  }
  auto ty = dyn_cast<IntegerType>(yieldOp.getOperand(0).getType());
  if (!ty || ty.getWidth() != 1) {
    return op->emitOpError("should yield i1 type");
  }

  return success();
}

LogicalResult
SortOp::reifyResultShapes(OpBuilder &b,
                          ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// FftOp
//===----------------------------------------------------------------------===//

LogicalResult FftOp::verify() {
  Operation *op = getOperation();
  auto length = getFftLength();
  // After tiling, it could be dynamic shape. (Because
  // subview/subtensor does not inference the type correctly
  // on (1 << x)) cases).
  if (ShapedType::isDynamic(length))
    return success();
  if (length & (length - 1)) {
    return op->emitOpError("only powers of 2 are handled currently");
  }
  if (!getNumDpsInputs() || !isScalar(getDpsInputOperand(0))) {
    return op->emitOpError("expected to carry `stage` input");
  }
  if (getNumDpsInputs() != 1) {
    if (getNumDpsInputs() != 3 || isScalar(getDpsInputOperand(1)) ||
        isScalar(getDpsInputOperand(2))) {
      return op->emitOpError("expected to carry real and imag coeff inputs");
    }
  }
  if (getNumDpsInits() != 2) {
    return op->emitOpError(
        "expected outputs to be real and imag tensor/memref");
  }
  return success();
}

LogicalResult
FftOp::reifyResultShapes(OpBuilder &b,
                         ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// ScanOp
//===----------------------------------------------------------------------===//

LogicalResult ScanOp::verify() {
  Operation *op = getOperation();
  if (getNumDpsInputs() != 1) {
    return op->emitOpError("expected one input operands");
  }
  if (getNumDpsInits() != 2) {
    return op->emitOpError("expected two output operands");
  }
  if (!isa<ShapedType>(getInput().getType())) {
    return op->emitOpError("expected first input element type to be shaped");
  }
  auto accumulatorType = cast<ShapedType>(getAccumulator().getType());
  auto inputType = cast<ShapedType>(getInput().getType());
  auto outputType = cast<ShapedType>(getOutput().getType());
  ArrayRef<int64_t> inputShapes = inputType.getShape();
  ArrayRef<int64_t> outputShapes = outputType.getShape();
  if (accumulatorType.getElementType() != inputType.getElementType()) {
    return op->emitOpError(
        "expected input/accumulator element types to be identical");
  }
  ArrayRef<int64_t> accumulatorShape = accumulatorType.getShape();
  int64_t accumulatorRank = accumulatorType.getRank();
  if (accumulatorRank != inputType.getRank() - 1) {
    return op->emitOpError(
        "expected accumulator rank to be equal to input rank - 1");
  }
  SmallVector<int64_t> expectedAccumulatorShape;
  for (int i = 0; i < inputType.getRank(); i++) {
    if (i != getDimension()) {
      expectedAccumulatorShape.push_back(inputShapes[i]);
    }
  }
  if (llvm::any_of(llvm::zip_equal(expectedAccumulatorShape, accumulatorShape),
                   [](std::tuple<int64_t, int64_t> s) {
                     return !ShapedType::isDynamic(std::get<0>(s)) &&
                            !ShapedType::isDynamic(std::get<1>(s)) &&
                            std::get<0>(s) != std::get<1>(s);
                   })) {
    return op->emitOpError("incompatible input/accumulator shapes");
  }
  if (inputType.getElementType() != outputType.getElementType()) {
    return op->emitOpError(
        "expected input/output element types to be identical");
  }
  if (inputShapes.size() != outputShapes.size()) {
    return op->emitOpError("expected input/output to have identical ranks");
  }
  if (llvm::any_of(llvm::zip_equal(inputShapes, outputShapes),
                   [](std::tuple<int64_t, int64_t> s) {
                     return !ShapedType::isDynamic(std::get<0>(s)) &&
                            !ShapedType::isDynamic(std::get<1>(s)) &&
                            std::get<0>(s) != std::get<1>(s);
                   })) {
    return op->emitOpError("incompatible input/output shapes");
  }
  return success();
}

LogicalResult ScanOp::fold(FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

LogicalResult
ScanOp::reifyResultShapes(OpBuilder &b,
                          ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// ReverseOp
//===----------------------------------------------------------------------===//

LogicalResult ReverseOp::verify() {
  Operation *op = getOperation();
  if (getNumDpsInputs() != 1) {
    return op->emitOpError("expected exactly one input");
  }
  if (getNumDpsInits() != 1) {
    return op->emitOpError("expected exactly one output");
  }
  auto inputType = cast<ShapedType>(getInput().getType());
  auto outputType = cast<ShapedType>(getOutput().getType());
  if (inputType.getElementType() != outputType.getElementType()) {
    return op->emitOpError(
        "expected input/output element types to be identical");
  }
  ArrayRef<int64_t> inputShapes = inputType.getShape();
  ArrayRef<int64_t> outputShapes = outputType.getShape();
  if (inputShapes.size() != outputShapes.size()) {
    return op->emitOpError("expexted input/output to have identical ranks");
  }
  if (llvm::any_of(llvm::zip_equal(inputShapes, outputShapes),
                   [](std::tuple<int64_t, int64_t> s) {
                     return !ShapedType::isDynamic(std::get<0>(s)) &&
                            !ShapedType::isDynamic(std::get<1>(s)) &&
                            std::get<0>(s) != std::get<1>(s);
                   })) {
    return op->emitOpError("incompatible input/output shapes");
  }

  int64_t rank = getOperandRank();
  llvm::SmallSetVector<int64_t, 4> s;
  for (auto dim : getDimensionsArray()) {
    if (dim < 0 || dim >= rank) {
      return op->emitOpError("all the dimensions must be within [0, ")
             << rank << ")";
    }
    if (s.contains(dim)) {
      return op->emitOpError("expected dimensions numbers are all unique");
    }
    s.insert(dim);
  }

  return success();
}

LogicalResult
ReverseOp::reifyResultShapes(OpBuilder &b,
                             ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

SmallVector<AffineMap> ReverseOp::getIndexingMapsForOperands() {
  Builder builder(getContext());
  return {builder.getMultiDimIdentityMap(getOperandRank()),
          /*output=*/AffineMap(nullptr)};
}

SmallVector<AffineMap> ReverseOp::getIndexingMapsForResults() {
  return {AffineMap(nullptr)};
}

//===----------------------------------------------------------------------===//
// TopkOp
//===----------------------------------------------------------------------===//

LogicalResult TopkOp::verify() {
  Operation *op = getOperation();
  if (getNumDpsInputs() != 1 && getNumDpsInputs() != 2) {
    return op->emitOpError("expected one or two input operands");
  }
  if (getNumDpsInits() != 2) {
    return op->emitOpError("expected two output operands");
  }
  if (getDimension() >= getInputRank()) {
    return op->emitOpError("dimension exceeds rank");
  }
  // Ensure input/output element types match
  auto inputValuesType = cast<ShapedType>(getValues().getType());
  auto outputValuesType = cast<ShapedType>(outputValues().getType());
  if (inputValuesType.getElementType() != outputValuesType.getElementType()) {
    return op->emitOpError("expected input/output value types to be identical");
  }
  // Indices must be int if provided
  auto outputIndicesType = cast<ShapedType>(outputIndices().getType());
  if (auto inputIndices = getIndices()) {
    auto inputIndicesType = cast<ShapedType>(inputIndices->getType());
    if (!inputIndicesType.getElementType().isInteger(32) ||
        !outputIndicesType.getElementType().isInteger(32)) {
      return op->emitOpError("expected input/output indices types to be int32");
    }
  }

  // Ranks must match
  if (inputValuesType.getRank() != outputValuesType.getRank()) {
    return op->emitOpError("expected input/output to have the same rank");
  }
  if (auto inputIndices = getIndices()) {
    auto inputIndicesType = cast<ShapedType>(inputIndices->getType());
    if (inputIndicesType.getRank() != outputIndicesType.getRank()) {
      return op->emitOpError("expected input/output to have the same rank");
    }
  }
  // Input indicies and values must have the same shape.
  if (auto inputIndices = getIndices()) {
    auto inputIndicesType = cast<ShapedType>(inputIndices->getType());
    if (failed(verifyCompatibleShape(inputValuesType, inputIndicesType))) {
      return op->emitOpError("input indices/values shape must match");
    }
  }
  // Output indicies and values must have the same shape.
  if (failed(verifyCompatibleShape(outputValuesType, outputIndicesType))) {
    return op->emitOpError("output indices/values shape must match");
  }
  // Input shape must match the output shape except for the dimension()
  uint64_t dim = getDimension();
  if (!llvm::all_of(
          llvm::enumerate(llvm::zip_equal(inputValuesType.getShape(),
                                          outputValuesType.getShape())),
          [dim](auto e) {
            if (e.index() == dim) {
              return true;
            }
            std::tuple<int64_t, int64_t> s = e.value();
            return succeeded(
                verifyCompatibleShape(std::get<0>(s), std::get<1>(s)));
          })) {
    return op->emitOpError("incompatible input/output shapes");
  }
  // Check region compatibility
  Block &block = getRegion().front();
  if (block.getNumArguments() != 2) {
    return op->emitOpError("region block should have 2 arguments");
  }
  if (block.getArgument(0).getType() != inputValuesType.getElementType() ||
      block.getArgument(1).getType() != inputValuesType.getElementType()) {
    return op->emitOpError("region block types must match input");
  }
  auto terminatorOp = llvm::dyn_cast<YieldOp>(block.getTerminator());
  if (!terminatorOp || !terminatorOp.getOperand(0).getType().isInteger(1)) {
    return op->emitOpError("region block must end with a linalg_ext.yield i1!");
  }
  return success();
}

LogicalResult
TopkOp::reifyResultShapes(OpBuilder &b,
                          ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// PackOp and UnPackOp utils
//===----------------------------------------------------------------------===//

/// Return true if at least one element in `tiles` is zero.
static bool hasZeros(ArrayRef<OpFoldResult> tiles) {
  return llvm::any_of(
      tiles, [&](OpFoldResult tile) { return isConstantIntValue(tile, 0); });
}

/// Check if we have enough static information to catch undefined behavior when
/// the tile size does not divide perfectly the dimension of the input tensor.
static bool
areNotFullTiles(ArrayRef<int64_t> inputShape,
                DenseMap<int64_t, OpFoldResult> const &dimAndTileMapping) {
  int64_t rank = inputShape.size();
  for (int64_t dim = 0; dim < rank; dim++) {
    if (ShapedType::isDynamic(inputShape[dim]))
      continue;
    auto it = dimAndTileMapping.find(dim);
    if (it != dimAndTileMapping.end()) {
      std::optional<int64_t> constantTile = getConstantIntValue(it->second);
      if (!constantTile)
        continue;
      if (inputShape[dim] % (*constantTile) != 0)
        return true;
    }
  }
  return false;
}

static SmallVector<OpFoldResult> getMixedValues(MLIRContext *context,
                                                ArrayRef<int64_t> staticValues,
                                                OperandRange dynamicValues) {
  OpBuilder b(context);
  return mlir::getMixedValues(staticValues, dynamicValues, b);
}

static SmallVector<int64_t>
getStaticValues(SmallVector<OpFoldResult> mixedValues) {
  SmallVector<Value> dynamicTiles;
  SmallVector<int64_t> staticTiles;
  dispatchIndexOpFoldResults(mixedValues, dynamicTiles, staticTiles);
  return staticTiles;
}

/// Utility function shared between Pack and UnPack to get the tile sizes as
/// OpFoldResults.
// TODO: interface or base class in .td
template <typename OpTy>
static SmallVector<OpFoldResult> getMixedTiles(OpTy op) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  return LinalgExt::getMixedValues(op.getContext(), op.getStaticInnerTiles(),
                                   op.getInnerTiles());
}

/// Return the tile sizes as `int64_t`. If a tile size is dynamic a sentinel
/// `kDynamic` is introduced at that position in the returned vector.
template <typename OpTy>
static SmallVector<int64_t> getStaticTiles(OpTy op) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  return getStaticValues(op.getMixedTiles());
}

/// Utility function shared between Pack and UnPack to get a map between
/// `dim_pos` and `inner_tiles`.
// TODO: interface or base class in .td
template <typename OpTy>
static DenseMap<int64_t, OpFoldResult> getDimAndTileMapping(OpTy op) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  DenseMap<int64_t, OpFoldResult> dimAndTileMapping;
  ArrayRef<int64_t> dimsToBlock = op.getInnerDimsPos();
  SmallVector<OpFoldResult> tiles = op.getMixedTiles();
  assert(tiles.size() == dimsToBlock.size() &&
         "tiles must match indices of dimension to block");
  // bind the dimension with the tile factor.
  for (auto i : llvm::seq<int64_t>(0, dimsToBlock.size())) {
    dimAndTileMapping[dimsToBlock[i]] = tiles[i];
  }
  return dimAndTileMapping;
}

/// Common verifier for `PackOp` and `UnPackOp`.
template <typename OpTy>
static LogicalResult commonVerifierPackAndUnPackOp(OpTy packOrUnPack) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  Operation *op = packOrUnPack.getOperation();
  ShapedType unpackedType = (std::is_same<OpTy, PackOp>::value)
                                ? packOrUnPack.getInputType()
                                : packOrUnPack.getOutputType();
  int64_t unpackedRank = unpackedType.getRank();
  ArrayRef<int64_t> innerDimsPos = packOrUnPack.getInnerDimsPos();
  ArrayRef<int64_t> outerDimPerm = packOrUnPack.getOuterDimsPerm();
  // Verify tiles. Make sure each provided tile is non-zero.
  SmallVector<OpFoldResult> mixedTiles = packOrUnPack.getMixedTiles();
  if (hasZeros(mixedTiles)) {
    return op->emitError("invalid tile factor");
  }
  if (isInvalid(innerDimsPos, unpackedRank)) {
    return op->emitError("invalid inner_dims_pos vector");
  }
  if (isInvalid(outerDimPerm, unpackedRank)) {
    return op->emitError("invalid outer_dims_perm vector");
  }
  if (mixedTiles.size() != innerDimsPos.size()) {
    return op->emitError(
        "blocking factors must equal the number of dimensions to block");
  }

  // Blocking factors must be less or equal than the input rank, and must
  // match the number of `dims_pos`.
  if (mixedTiles.size() > unpackedRank) {
    return op->emitError(
        "blocking factors must be less or equal than the input rank");
  }

  ShapedType packedType = (std::is_same<OpTy, PackOp>::value)
                              ? packOrUnPack.getOutputType()
                              : packOrUnPack.getInputType();
  int64_t packedRank = packedType.getRank();
  // Require output rank to match input rank + number of blocking factors.
  if (unpackedRank + mixedTiles.size() != packedRank) {
    return op->emitError(
        "packed rank must equal unpacked rank + blocking factors");
  }

  // Verify result shape is greater than the minimum expected
  // by the pack operation, and that the output shape
  // represents full tiles.
  ShapedType expectedPackedType = PackOp::getPackedType(
      unpackedType, packOrUnPack.getStaticTiles(), innerDimsPos, outerDimPerm);
  if (!isSmallerThan(expectedPackedType.getShape(), packedType.getShape())) {
    return op->emitError("the shape of output is not large enough to hold the "
                         "packed data. Expected at least ")
           << expectedPackedType << ", got " << packedType;
  }
  if (!llvm::all_of(
          llvm::zip_equal(packedType.getShape().take_back(mixedTiles.size()),
                          mixedTiles),
          [](std::tuple<int64_t, OpFoldResult> it) {
            std::optional<int64_t> constTileSize =
                getConstantIntValue(std::get<1>(it));
            int64_t shape = std::get<0>(it);
            if (!constTileSize) {
              // If specified tile size is dynamic, output shape should
              // be dynamic too.
              return ShapedType::isDynamic(shape);
            } else {
              if (ShapedType::isDynamic(shape)) {
                // For the shape being dynamic when tile size is
                // specified, return true. In canonical form a constant
                // tile size should lead to constant shape of the tiled
                // dimension, but not needed for verification.
                return true;
              }
              return shape == constTileSize.value();
            }
          })) {
    return op->emitError("mismatch in inner tile sizes specified and shaped of "
                         "tiled dimension in the packed type");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// PackOp
//===----------------------------------------------------------------------===//

/// Custom builder methods for pack ops.
void PackOp::build(OpBuilder &builder, OperationState &state, Value source,
                   Value output, ArrayRef<int64_t> innerDimsPos,
                   ArrayRef<OpFoldResult> innerTiles,
                   std::optional<Value> paddingValue,
                   ArrayRef<int64_t> outerDimsPerm) {
  assert(innerDimsPos.size() == innerTiles.size() &&
         "number of tile sizes specified must match the specified number of "
         "original dimensions to be tiled");
  SmallVector<int64_t> staticTileSizes;
  SmallVector<Value> dynamicTileSizes;
  dispatchIndexOpFoldResults(innerTiles, dynamicTileSizes, staticTileSizes);
  SmallVector<Type> resultType;
  auto outputType = output.getType();
  if (isa<RankedTensorType>(outputType)) {
    resultType.push_back(outputType);
  }
  build(builder, state, resultType, source, output,
        outerDimsPerm.empty() ? nullptr
                              : builder.getDenseI64ArrayAttr(outerDimsPerm),
        builder.getDenseI64ArrayAttr(innerDimsPos), dynamicTileSizes,
        builder.getDenseI64ArrayAttr(staticTileSizes),
        (paddingValue ? paddingValue.value() : nullptr));
}

LogicalResult PackOp::verify() {
  if (failed(commonVerifierPackAndUnPackOp(*this))) {
    return failure();
  }

  // Bail out if the tile does not divide the dimension fully. In the case of
  // dynamic tile factors or dimensions, having a partial tile is undefined
  // behavior.
  auto dimAndTileMapping = getDimAndTileMapping();
  if (!getPaddingValue() &&
      areNotFullTiles(getInputShape(), dimAndTileMapping)) {
    return emitOpError("invalid tile factor provided. Only full tiles are "
                       "supported when padding_value is not set");
  }

  if (auto paddingValue = getPaddingValue()) {
    if (paddingValue.getType() != getInputType().getElementType()) {
      return emitOpError("expected padding_value has ")
             << getInputType().getElementType()
             << " but got: " << paddingValue.getType();
    }
  }
  return success();
}

SmallVector<OpFoldResult> PackOp::getMixedTiles() {
  return LinalgExt::getMixedTiles(*this);
}

SmallVector<int64_t> PackOp::getStaticTiles() {
  return LinalgExt::getStaticTiles(*this);
}

// Helper for PackOp::{getResultShape,getPackedType}. Returns the shape of the
// packed type. Having a shared helper helps implement these two methods in a
// way that ensures that they agree on which dimensions are dynamic.
static SmallVector<int64_t> getPackOpResultTypeShape(
    ArrayRef<int64_t> sourceShape, ArrayRef<int64_t> innerTileSizes,
    ArrayRef<int64_t> innerDimsPos, ArrayRef<int64_t> outerDimsPerm) {
  SmallVector<int64_t> resultShape = llvm::to_vector(sourceShape);
  for (auto [idx, tiledDim] : llvm::enumerate(innerDimsPos)) {
    if (ShapedType::isDynamic(resultShape[tiledDim])) {
      continue;
    }
    if (ShapedType::isDynamic(innerTileSizes[idx])) {
      resultShape[tiledDim] = ShapedType::kDynamic;
      continue;
    }
    resultShape[tiledDim] =
        llvm::divideCeil(resultShape[tiledDim], innerTileSizes[idx]);
  }

  // Swap tile loops if outer_dims_perm is available.
  resultShape = interchange<int64_t>(resultShape, outerDimsPerm, /*offset=*/0);

  // Append the inner tile dimensions.
  resultShape.append(innerTileSizes.begin(), innerTileSizes.end());
  return resultShape;
}

SmallVector<OpFoldResult> PackOp::getResultShape(
    OpBuilder &builder, Location loc, ArrayRef<OpFoldResult> sourceDims,
    ArrayRef<OpFoldResult> innerTileSizes, ArrayRef<int64_t> innerDimsPos,
    ArrayRef<int64_t> outerDimsPerm) {
  SmallVector<OpFoldResult> resultDims = llvm::to_vector(sourceDims);

  AffineExpr s0, s1;
  bindSymbols(builder.getContext(), s0, s1);
  AffineExpr ceilDivExpr = s0.ceilDiv(s1);
  for (auto [idx, tiledDim] : llvm::enumerate(innerDimsPos)) {
    resultDims[tiledDim] = affine::makeComposedFoldedAffineApply(
        builder, loc, ceilDivExpr, {resultDims[tiledDim], innerTileSizes[idx]});
  }
  if (!outerDimsPerm.empty()) {
    resultDims =
        interchange<OpFoldResult>(resultDims, outerDimsPerm, /*offset=*/0);
  }
  resultDims.append(innerTileSizes.begin(), innerTileSizes.end());

  SmallVector<int64_t> resultTypeShape =
      getPackOpResultTypeShape(asShapeWithAnyValueAsDynamic(sourceDims),
                               asShapeWithAnyValueAsDynamic(innerTileSizes),
                               innerDimsPos, outerDimsPerm);

  // Fix-up `resultDims` to ensure that they are Value's if and only if the
  // result type shape says it's a dynamic dim. This is needed as callers may
  // use dispatchIndexOpFoldResults on the result, and rely on exact number of
  // dynamic dims returned by that.
  for (unsigned i = 0; i < resultDims.size(); ++i) {
    if (!ShapedType::isDynamic(resultTypeShape[i])) {
      continue;
    }
    resultDims[i] =
        getValueOrCreateConstantIndexOp(builder, loc, resultDims[i]);
  }

  return resultDims;
}

SmallVector<OpFoldResult> PackOp::getResultShape(OpBuilder &builder) {
  return tensor::getMixedSizes(builder, getLoc(), getOutput());
}

ShapedType PackOp::getPackedType(ShapedType sourceType,
                                 ArrayRef<int64_t> innerTileSizes,
                                 ArrayRef<int64_t> innerDimsPos,
                                 ArrayRef<int64_t> outerDimsPerm) {
  SmallVector<int64_t> resultTypeShape = getPackOpResultTypeShape(
      sourceType.getShape(), innerTileSizes, innerDimsPos, outerDimsPerm);

  return TypeSwitch<ShapedType, ShapedType>(sourceType)
      .Case<RankedTensorType>([&](auto shapedType) {
        return RankedTensorType::get(resultTypeShape,
                                     shapedType.getElementType());
      })
      .Case<MemRefType>([&](auto shapedType) {
        return MemRefType::get(resultTypeShape, shapedType.getElementType());
      })
      .Default([&](Type t) {
        assert(false && "unexpected type");
        return nullptr;
      });
}

DenseMap<int64_t, OpFoldResult> PackOp::getDimAndTileMapping() {
  return LinalgExt::getDimAndTileMapping(*this);
}

LogicalResult
PackOp::reifyResultShapes(OpBuilder &builder,
                          ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(builder, reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// UnPackOp
//===----------------------------------------------------------------------===//

/// Custom builder methods for unpack ops.
void UnPackOp::build(OpBuilder &builder, OperationState &state, Value source,
                     Value output, ArrayRef<int64_t> innerDimsPos,
                     ArrayRef<OpFoldResult> innerTiles,
                     ArrayRef<int64_t> outerDimsPerm) {
  SmallVector<int64_t> staticTileSizes;
  SmallVector<Value> dynamicTileSizes;
  dispatchIndexOpFoldResults(innerTiles, dynamicTileSizes, staticTileSizes);
  SmallVector<Type> resultType;
  auto outputType = output.getType();
  if (isa<RankedTensorType>(outputType)) {
    resultType.push_back(outputType);
  }
  build(builder, state, resultType, source, output,
        outerDimsPerm.empty() ? nullptr
                              : builder.getDenseI64ArrayAttr(outerDimsPerm),
        builder.getDenseI64ArrayAttr(innerDimsPos), dynamicTileSizes,
        builder.getDenseI64ArrayAttr(staticTileSizes));
}

SmallVector<OpFoldResult> UnPackOp::getMixedTiles() {
  return LinalgExt::getMixedTiles(*this);
}

SmallVector<int64_t> UnPackOp::getStaticTiles() {
  return LinalgExt::getStaticTiles(*this);
}

DenseMap<int64_t, OpFoldResult> UnPackOp::getDimAndTileMapping() {
  return LinalgExt::getDimAndTileMapping(*this);
}

LogicalResult
UnPackOp::reifyResultShapes(OpBuilder &builder,
                            ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(builder, reifiedReturnShapes);
}

LogicalResult UnPackOp::verify() {
  if (failed(commonVerifierPackAndUnPackOp(*this))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Winograd op utilities
//===----------------------------------------------------------------------===//

template <typename WinogradOp>
static SmallVector<int64_t> getNonInputTileDims(WinogradOp op) {
  static_assert(llvm::is_one_of<WinogradOp, WinogradInputTransformOp,
                                WinogradFilterTransformOp,
                                WinogradOutputTransformOp>::value,
                "applies to only winograd transform operations");
  SetVector<int64_t> inputTileDims(op.getInputTileDimensions().begin(),
                                   op.getInputTileDimensions().end());
  SmallVector<int64_t> dims = llvm::to_vector(
      llvm::seq<int64_t>(op.getTransformedOperandType().getRank()));
  SetVector<int64_t> dimSet(dims.begin(), dims.end());
  dimSet.set_subtract(inputTileDims);
  return dimSet.takeVector();
}

//===----------------------------------------------------------------------===//
// WinogradInputTransformOp
//===----------------------------------------------------------------------===//

SmallVector<int64_t> WinogradInputTransformOp::getNonInputTileDims() {
  return LinalgExt::getNonInputTileDims(*this);
}

LogicalResult WinogradInputTransformOp::verify() {
  Operation *op = getOperation();
  if (getNumDpsInputs() != 1) {
    return op->emitOpError("expected one input operand");
  }
  if (getNumDpsInits() != 1) {
    return op->emitOpError("expected one output operand");
  }
  auto inputType = getInputType();
  auto outputType = getOutputType();
  if (outputType.getElementType() != inputType.getElementType()) {
    return op->emitOpError(
        "expected input/output element types to be identical");
  }
  unsigned inputRank = inputType.getRank();
  unsigned outputRank = outputType.getRank();

  if (inputRank != 2 && inputRank != 4) {
    return op->emitOpError("expected input operand to have rank either 2 or 4");
  }

  if (inputRank == 2) {
    if (outputRank != 2) {
      return op->emitOpError(
          "expected output operand to have rank 2 if input is of rank 2");
    }
    if ((!inputType.isDynamicDim(0) &&
         inputType.getDimSize(0) > getInputTileSize()) ||
        (inputType.isDynamicDim(1) &&
         inputType.getDimSize(1) > getInputTileSize())) {
      return op->emitOpError("expected input dims not greater than input tile "
                             "size if input is of rank 2");
    }
    SmallVector<int64_t> expectedOutputShape(2, getInputTileSize());
    if (failed(verifyCompatibleShape(expectedOutputShape,
                                     outputType.getShape()))) {
      return op->emitOpError(
          "expected output dims equal to inputTileSize if input is of rank 2");
    }
    return success();
  }

  if (getOutputRank() != getInputRank() + 2) {
    return op->emitOpError(
        "expected output rank to be equal to input rank + 2");
  }
  ArrayRef<int64_t> imageDims = getImageDimensions();
  llvm::SmallSetVector<int64_t, 2> imageDimsSet(imageDims.begin(),
                                                imageDims.end());
  if (imageDims.size() != 2) {
    return op->emitOpError("expected only 2 image dimensions");
  }
  if (!isNchw() && !isNhwc()) {
    return op->emitOpError(
        "expect image dimensions to be either [1, 2] or [2, 3]");
  }
  SmallVector<int64_t> expectedOutputShape(getOutputRank(), getInputTileSize());
  int outputIndex;
  ArrayRef<int64_t> inputShape = inputType.getShape();
  for (int i = 0; i < inputShape.size(); i++) {
    outputIndex = i + imageDims.size();
    if (ShapedType::isDynamic(inputShape[i])) {
      expectedOutputShape[outputIndex] = inputShape[i];
      continue;
    }
    if (!imageDimsSet.contains(i)) {
      expectedOutputShape[outputIndex] = inputShape[i];
    } else {
      expectedOutputShape[outputIndex] =
          std::ceil(static_cast<float>(inputShape[i] - getKernelSize() + 1) /
                    getOutputTileSize());
    }
  }
  if (isNchw()) {
    permute<Permutation::TTNCHW_TO_TTNHWC>(expectedOutputShape);
  }
  SmallVector<int64_t> outputShape(outputType.getShape());
  SmallVector<int64_t> perm(getInputTileDimensions());
  perm.append(getNonInputTileDims());
  applyPermutationToVector(outputShape, perm);
  if (failed(verifyCompatibleShape(expectedOutputShape, outputShape))) {
    return op->emitOpError("incompatible output shape");
  }
  return success();
}

LogicalResult WinogradInputTransformOp::fold(FoldAdaptor,
                                             SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

LogicalResult WinogradInputTransformOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// WinogradFilterTransformOp
//===----------------------------------------------------------------------===//

SmallVector<int64_t> WinogradFilterTransformOp::getNonInputTileDims() {
  return LinalgExt::getNonInputTileDims(*this);
}

LogicalResult WinogradFilterTransformOp::verify() {
  Operation *op = getOperation();
  if (getNumDpsInputs() != 1) {
    return op->emitOpError("expected one input operand");
  }
  if (getNumDpsInits() != 1) {
    return op->emitOpError("expected one output operand");
  }
  auto inputType = getInputType();
  auto outputType = getOutputType();
  if (outputType.getElementType() != inputType.getElementType()) {
    return op->emitOpError(
        "expected input/output element types to be identical");
  }
  unsigned inputRank = inputType.getRank();
  unsigned outputRank = outputType.getRank();

  if (inputRank != 2 && inputRank != 4) {
    return op->emitOpError("expected input operand to have rank either 2 or 4");
  }

  if (inputRank == 2) {
    if (outputRank != 2) {
      return op->emitOpError(
          "expected output operand to have rank 2 if input is of rank 2");
    }
    SmallVector<int64_t> expectedInputShape(2, getKernelSize());
    if (failed(
            verifyCompatibleShape(expectedInputShape, inputType.getShape()))) {
      return op->emitOpError("expected input dims to be equal to kernel size "
                             "if input is of rank 2");
    }
    SmallVector<int64_t> expectedOutputShape(2, getInputTileSize());
    if (failed(verifyCompatibleShape(expectedOutputShape,
                                     outputType.getShape()))) {
      return op->emitOpError("expected output dims equal to input tile size if "
                             "input is of rank 2");
    }
    return success();
  }

  if (getOutputRank() != getInputRank()) {
    return op->emitOpError("expected output rank to be equal to input rank");
  }
  const ArrayRef<int64_t> kernelDims = getKernelDimensions();
  if (kernelDims.size() != 2) {
    return op->emitOpError("expected only 2 kernel dimensions");
  }
  if (!isHwcf() && !isFchw()) {
    return op->emitOpError(
        "expect kernel dimensions to be either [0, 1] or [2, 3]");
  }
  const int64_t kernelSize = getKernelSize();
  for (auto kernelDim : kernelDims) {
    if (inputType.getDimSize(kernelDim) != kernelSize) {
      return op->emitOpError(
          "expect all kernel dimensions to have the kernel size");
    }
  }
  const int64_t inputTileSize = getInputTileSize();
  SmallVector<int64_t> expectedOutputShape(kernelDims.size(), inputTileSize);
  llvm::SmallSetVector<int64_t, 2> kernelDimsSet(kernelDims.begin(),
                                                 kernelDims.end());
  for (int i = 0; i < inputType.getRank(); i++) {
    if (!kernelDimsSet.contains(i)) {
      expectedOutputShape.push_back(inputType.getDimSize(i));
    }
  }
  if (isFchw()) {
    permute<Permutation::TTFC_TO_TTCF>(expectedOutputShape);
  }
  SmallVector<int64_t> outputShape(outputType.getShape());
  SmallVector<int64_t> perm(getInputTileDimensions());
  perm.append(getNonInputTileDims());
  applyPermutationToVector(outputShape, perm);
  if (failed(verifyCompatibleShape(expectedOutputShape, outputShape))) {
    return op->emitOpError("incompatible output shape");
  }
  return success();
}

LogicalResult WinogradFilterTransformOp::fold(FoldAdaptor,
                                              SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

LogicalResult WinogradFilterTransformOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// WinogradOutputTransformOp
//===----------------------------------------------------------------------===//

SmallVector<int64_t> WinogradOutputTransformOp::getNonInputTileDims() {
  return LinalgExt::getNonInputTileDims(*this);
}

LogicalResult WinogradOutputTransformOp::verify() {
  Operation *op = getOperation();
  if (getNumDpsInputs() != 1) {
    return op->emitOpError("expected one input operand");
  }
  if (getNumDpsInits() != 1) {
    return op->emitOpError("expected one output operand");
  }
  auto inputType = getInputType();
  auto outputType = getOutputType();
  unsigned inputRank = inputType.getRank();
  unsigned outputRank = outputType.getRank();

  if (inputRank != 2 && inputRank != 6) {
    return op->emitOpError("expected input operand to have rank either 2 or 6");
  }

  if (inputRank == 2) {
    if (outputRank != 2) {
      return op->emitOpError(
          "expected output operand to have rank 2 if input is of rank 2");
    }
    SmallVector<int64_t> expectedInputShape(2, getInputTileSize());
    if (failed(
            verifyCompatibleShape(expectedInputShape, inputType.getShape()))) {
      return op->emitOpError("expected input dims to be equal to input tile "
                             "size if input is of rank 2");
    }
    SmallVector<int64_t> expectedOutputShape(2, getOutputTileSize());
    if (failed(verifyCompatibleShape(expectedOutputShape,
                                     outputType.getShape()))) {
      return op->emitOpError("expected output dims equal to output tile size "
                             "if input is of rank 2");
    }
    return success();
  }
  if (outputType.getElementType() != inputType.getElementType()) {
    return op->emitOpError(
        "expected input/output element types to be identical");
  }
  if (outputRank != inputRank - 2) {
    return op->emitOpError(
        "expected output rank to be equal to input rank - 2");
  }
  ArrayRef<int64_t> imageDims = getImageDimensions();
  llvm::SmallSetVector<int64_t, 2> imageDimsSet(imageDims.begin(),
                                                imageDims.end());
  if (imageDims.size() != 2) {
    return op->emitOpError("expected only 2 image dimensions");
  }
  if (!isNchw() && !isNhwc()) {
    return op->emitOpError(
        "expect image dimensions to be either [1, 2] or [2, 3]");
  }
  SmallVector<int64_t> inputShape(inputType.getShape());
  SmallVector<int64_t> perm(getInputTileDimensions());
  perm.append(getNonInputTileDims());
  applyPermutationToVector(inputShape, perm);
  if (isNchw()) {
    permute<Permutation::TTNHWC_TO_TTNCHW>(inputShape);
  }
  SmallVector<int64_t> expectedOutputShape(getOutputRank(), 1);
  int outputIndex;
  for (int i = imageDims.size(); i < inputShape.size(); i++) {
    outputIndex = i - imageDims.size();
    if (ShapedType::isDynamic(inputShape[i])) {
      expectedOutputShape[outputIndex] = inputShape[i];
      continue;
    }
    if (!imageDimsSet.contains(outputIndex)) {
      expectedOutputShape[outputIndex] = inputShape[i];
    } else {
      expectedOutputShape[outputIndex] = getOutputTileSize() * inputShape[i];
    }
  }
  if (failed(
          verifyCompatibleShape(expectedOutputShape, outputType.getShape()))) {
    return op->emitOpError("incompatible output shape");
  }
  return success();
}

LogicalResult WinogradOutputTransformOp::fold(FoldAdaptor,
                                              SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

LogicalResult WinogradOutputTransformOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// AttentionOp
//===----------------------------------------------------------------------===//

LogicalResult AttentionOp::verify() {
  Operation *op = getOperation();

  int numInputs = getNumDpsInputs();
  int numOutputs = getNumDpsInits();

  if (numInputs != 4) {
    return op->emitOpError(
        "expected 4 input operands: Query, Key, Value and Scale");
  }

  if (numOutputs != 1 && numOutputs != 3) {
    return op->emitOpError(
        "expected 1 or 3 output operands: Output, [Max and Sum]");
  }

  bool isTiled = numOutputs == 3;

  if (!llvm::all_of(llvm::drop_end(getDpsInputs()), [](Value input) {
        return isa<ShapedType>(input.getType());
      })) {
    return op->emitOpError(
        "expected Query, Key, Value inputs to be of shaped type");
  }

  ShapedType queryType = getQueryType();
  ShapedType keyType = getKeyType();
  ShapedType valueType = getValueType();
  ShapedType outputType = getOutputType();
  Type queryElementType = queryType.getElementType();
  Type keyElementType = keyType.getElementType();
  Type valueElementType = valueType.getElementType();
  Type outputElementType = outputType.getElementType();

  FloatType scaleElementType = dyn_cast<FloatType>(getScale().getType());
  if (!scaleElementType) {
    return op->emitOpError("expected scale to be of floating point type");
  }

  // Check shape compatibility based on indexing maps.
  SmallVector<int64_t> shape(getIterationDomainRank());
  SmallVector<bool> foundDims(getIterationDomainRank(), false);
  auto checkShape = [&shape, &foundDims,
                     &op](StringRef operandName, ArrayRef<int64_t> valShape,
                          AffineMap indexingMap) -> LogicalResult {
    if (indexingMap.getNumResults() != valShape.size()) {
      return op->emitError("Rank Mismatch for ")
             << operandName << ". Expected: " << indexingMap.getNumResults()
             << " Got: " << valShape.size();
    }
    for (auto [i, dimExpr] : llvm::enumerate(indexingMap.getResults())) {
      AffineDimExpr dim = cast<AffineDimExpr>(dimExpr);
      int64_t pos = dim.getPosition();
      if (ShapedType::isDynamic(valShape[i])) {
        continue;
      }
      if (!foundDims[pos]) {
        foundDims[pos] = true;
        shape[pos] = valShape[i];
      }
      if (shape[pos] != valShape[i]) {
        return op->emitError("Shape Mismatch for ")
               << operandName << ". Expected: " << shape[pos]
               << " Got: " << valShape[i];
      }
    }
    return success();
  };

  if (failed(checkShape("Query", getQueryType().getShape(), getQueryMap())) ||
      failed(checkShape("Key", getKeyType().getShape(), getKeyMap())) ||
      failed(checkShape("Value", getValueType().getShape(), getValueMap()))) {
    return failure();
  }

  if (queryElementType != keyElementType ||
      queryElementType != valueElementType ||
      queryElementType != scaleElementType) {
    return op->emitOpError(
        "element types of (Q)uery, (K)ey and (V)alue and scale should be "
        "same");
  }
  if (!isTiled) {
    // Vanilla attention.
    if (queryElementType != outputElementType) {
      return op->emitOpError("expected element type for Output ")
             << queryElementType << "but found " << outputElementType
             << " instead";
    }
  }
  if (isTiled) {
    // Tiled/Flash attention.
    Type maxElementType = getMaxType()->getElementType();
    Type sumElementType = getSumType()->getElementType();
    if (outputElementType != maxElementType ||
        maxElementType != sumElementType) {
      return op->emitOpError(
          "element types of tiled output, max and sum should be same");
    }

    if (failed(checkShape("Max", getMaxType()->getShape(), *getMaxMap())) ||
        failed(checkShape("Sum", getSumType()->getShape(), *getSumMap()))) {
      return failure();
    }
  }

  return success();
}

LogicalResult AttentionOp::fold(FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

LogicalResult AttentionOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

SmallVector<AffineMap> AttentionOp::getIndexingMapsArray() {
  MLIRContext *ctx = getContext();

  AffineExpr batch, m, k1, k2, n;
  bindDims(ctx, batch, m, k1, k2, n);

  AffineMap qMap =
      AffineMap::get(/*dimCount=*/5, /*symbolCount=*/0, {batch, m, k1}, ctx);
  AffineMap kMap =
      AffineMap::get(/*dimCount=*/5, /*symbolCount=*/0, {batch, k2, k1}, ctx);

  AffineMap vMap;
  if (getTransposeV()) {
    vMap =
        AffineMap::get(/*dimCount=*/5, /*symbolCount=*/0, {batch, n, k2}, ctx);
  } else {
    vMap =
        AffineMap::get(/*dimCount=*/5, /*symbolCount=*/0, {batch, k2, n}, ctx);
  }

  AffineMap resMap =
      AffineMap::get(/*dimCount=*/5, /*symbolCount=*/0, {batch, m, n}, ctx);

  SmallVector<AffineMap> results = {qMap, kMap, vMap, resMap};

  if (getMax()) {
    AffineMap maxMap =
        AffineMap::get(/*dimCount=*/5, /*symbolCount=*/0, {batch, m}, ctx);
    results.push_back(maxMap);
  }

  if (getSum()) {
    AffineMap sumMap =
        AffineMap::get(/*dimCount=*/5, /*symbolCount=*/0, {batch, m}, ctx);
    results.push_back(sumMap);
  }

  // Remove batch dim for tiled operands.
  // TODO: This is a weird expectation from TileAndDecomposeAttention.
  bool isTiled = getNumResults() == 3;
  if (isTiled) {
    for (AffineMap &map : results) {
      map = map.dropResult(0);
    }
  }

  return results;
}

//===----------------------------------------------------------------------===//
// OnlineAttentionOp
//===----------------------------------------------------------------------===//

LogicalResult OnlineAttentionOp::verify() {
  OnlineAttentionOp attnOp = *this;

  SmallVector<AffineMap> indexingMaps = attnOp.getIndexingMapsArray();

  // Check if indexing maps can represent attention.
  FailureOr<AttentionOpDetail> maybeOpInfo =
      AttentionOpDetail::get(indexingMaps);

  // Check shape compatibility based on indexing maps.
  SmallVector<int64_t> shape(getIterationDomainRank());
  SmallVector<bool> foundDims(getIterationDomainRank(), false);
  auto checkShape = [&shape, &foundDims,
                     &attnOp](StringRef operandName, ArrayRef<int64_t> valShape,
                              AffineMap indexingMap) -> LogicalResult {
    if (indexingMap.getNumResults() != valShape.size()) {
      return attnOp->emitError("Rank Mismatch for ")
             << operandName << ". Expected: " << indexingMap.getNumResults()
             << " Got: " << valShape.size();
    }
    for (auto [i, dimExpr] : llvm::enumerate(indexingMap.getResults())) {
      AffineDimExpr dim = cast<AffineDimExpr>(dimExpr);
      int64_t pos = dim.getPosition();
      if (ShapedType::isDynamic(valShape[i])) {
        continue;
      }
      if (!foundDims[pos]) {
        foundDims[pos] = true;
        shape[pos] = valShape[i];
      }
      if (shape[pos] != valShape[i]) {
        return attnOp->emitError("Shape Mismatch for ")
               << operandName << ". Expected: " << shape[pos]
               << " Got: " << valShape[i];
      }
    }
    return success();
  };

  if (failed(checkShape("Query", getQuery().getType().getShape(),
                        getQueryMap())) ||
      failed(checkShape("Key", getKey().getType().getShape(), getKeyMap())) ||
      failed(checkShape("Value", getValue().getType().getShape(),
                        getValueMap())) ||
      failed(checkShape("Output", getOutput().getType().getShape(),
                        getOutputMap())) ||
      failed(checkShape("Max", getMax().getType().getShape(), getMaxMap())) ||
      failed(checkShape("Sum", getSum().getType().getShape(), getSumMap()))) {
    return failure();
  }

  return success();
}

MutableOperandRange OnlineAttentionOp::getDpsInitsMutable() {
  return MutableOperandRange(*this, /*numInputs=*/4, /*numInits=*/3);
}

LogicalResult OnlineAttentionOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

SmallVector<AffineMap> OnlineAttentionOp::getIndexingMapsArray() {
  return SmallVector<AffineMap>(
      getIndexingMaps().getAsValueRange<AffineMapAttr>());
}

//===----------------------------------------------------------------------===//
// Im2colOp
//===----------------------------------------------------------------------===//

/// Return all static and dynamic kernel_size as OpFoldResults.
SmallVector<OpFoldResult> Im2colOp::getMixedKernelSize() {
  return LinalgExt::getMixedValues(getContext(), getStaticKernelSize(),
                                   getKernelSize());
}

/// Return all static and dynamic k_offset as OpFoldResults.
SmallVector<OpFoldResult> Im2colOp::getMixedKOffset() {
  return LinalgExt::getMixedValues(getContext(), getStaticKOffset(),
                                   getKOffset());
}

/// Return all static and dynamic k_offset as OpFoldResults.
SmallVector<OpFoldResult> Im2colOp::getMixedMOffset() {
  return LinalgExt::getMixedValues(getContext(), getStaticMOffset(),
                                   getMOffset());
}

void Im2colOp::setMixedKOffset(SmallVector<OpFoldResult> kOffset) {
  SmallVector<int64_t> staticKOffset;
  SmallVector<Value> dynamicKOffset;
  dispatchIndexOpFoldResults(kOffset, dynamicKOffset, staticKOffset);
  setStaticKOffset(staticKOffset);
  getKOffsetMutable().assign(dynamicKOffset);
}

void Im2colOp::setMixedMOffset(SmallVector<OpFoldResult> mOffset) {
  SmallVector<int64_t> staticMOffset;
  SmallVector<Value> dynamicMOffset;
  dispatchIndexOpFoldResults(mOffset, dynamicMOffset, staticMOffset);
  setStaticMOffset(staticMOffset);
  getMOffsetMutable().assign(dynamicMOffset);
}

/// Custom builder methods for im2col op.
void Im2colOp::build(OpBuilder &builder, OperationState &state, Value input,
                     Value output, ArrayRef<int64_t> strides,
                     ArrayRef<int64_t> dilations,
                     ArrayRef<OpFoldResult> kernelSize,
                     ArrayRef<OpFoldResult> kOffset,
                     ArrayRef<OpFoldResult> mOffset, ArrayRef<int64_t> batchPos,
                     ArrayRef<int64_t> mPos, ArrayRef<int64_t> kPos) {
  assert(strides.size() == kernelSize.size() &&
         dilations.size() == kernelSize.size() &&
         mPos.size() == kernelSize.size() &&
         "strides, dilations, m_pos, and kernel expected to be the same rank");
  SmallVector<int64_t> staticKernelSize, staticMOffset, staticKOffset;
  SmallVector<Value> dynamicKernelSize, dynamicMOffset, dynamicKOffset;
  dispatchIndexOpFoldResults(kernelSize, dynamicKernelSize, staticKernelSize);
  dispatchIndexOpFoldResults(mOffset, dynamicMOffset, staticMOffset);
  dispatchIndexOpFoldResults(kOffset, dynamicKOffset, staticKOffset);
  SmallVector<Type> resultType;
  auto outputType = output.getType();
  if (isa<RankedTensorType>(outputType)) {
    resultType.push_back(outputType);
  }
  build(builder, state, resultType, input, output,
        builder.getDenseI64ArrayAttr(strides),
        builder.getDenseI64ArrayAttr(dilations), dynamicKernelSize,
        builder.getDenseI64ArrayAttr(staticKernelSize), dynamicKOffset,
        builder.getDenseI64ArrayAttr(staticKOffset), dynamicMOffset,
        builder.getDenseI64ArrayAttr(staticMOffset),
        builder.getDenseI64ArrayAttr(batchPos),
        builder.getDenseI64ArrayAttr(mPos), builder.getDenseI64ArrayAttr(kPos));
}

LogicalResult Im2colOp::verify() {
  Operation *op = getOperation();
  if (llvm::count_if(getDpsInputs(), [](Value v) {
        return isa<ShapedType>(v.getType());
      }) != 1) {
    return op->emitOpError("expected only one ShapedType operand");
  }
  if (getNumDpsInits() != 1) {
    return op->emitOpError("expected one output operand");
  }

  // TODO(Max191): Support cases with more than 1 m or k dimension, and remove
  // the check for a single m_offset and k_offset.
  if (getMixedMOffset().size() != 1) {
    return op->emitOpError("expected one m_offset");
  }
  if (getMixedKOffset().size() != 1) {
    return op->emitOpError("expected one k_offset");
  }
  auto inputType = getInputType();
  unsigned inputRank = inputType.getRank();
  ArrayRef<int64_t> batchPos = getBatchPos();
  ArrayRef<int64_t> mPos = getMPos();
  ArrayRef<int64_t> kPos = getKPos();
  if (inputRank != batchPos.size() + mPos.size() + kPos.size()) {
    return op->emitOpError(
        "expected input rank to be the sum of batch, m, and k ranks");
  }
  ArrayRef<int64_t> strides = getStrides();
  ArrayRef<int64_t> dilations = getDilations();
  SmallVector<OpFoldResult> kernelSize = getMixedKernelSize();
  if (kernelSize.size() != mPos.size()) {
    return op->emitOpError(
        "expected kernel rank to be equal to the m_pos rank");
  }
  if (strides.size() != kernelSize.size()) {
    return op->emitOpError(
        "expected strides rank to be equal to the kernel rank");
  }
  if (dilations.size() != kernelSize.size()) {
    return op->emitOpError(
        "expected dilations rank to be equal to the kernel rank");
  }

  ArrayRef<int64_t> inputShape = inputType.getShape();
  SmallVector<int64_t> expectedOutputShape;
  for (auto pos : batchPos) {
    expectedOutputShape.push_back(inputShape[pos]);
  }
  ArrayRef<int64_t> outputShape = getOutputType().getShape();
  // When the op is tiled, the m and k dimensions of the output are tiled, but
  // they are not tiled in the input, so we cannot verify the output size of
  // these dimensions.
  expectedOutputShape.push_back(outputShape[outputShape.size() - 2]);
  expectedOutputShape.push_back(outputShape.back());
  if (failed(verifyCompatibleShape(expectedOutputShape, outputShape))) {
    return op->emitOpError("incompatible output shape");
  }
  return success();
}

LogicalResult Im2colOp::fold(FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

LogicalResult
Im2colOp::reifyResultShapes(OpBuilder &b,
                            ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

#define DEFINE_OP_GET_EFFECTS(OP_NAME)                                         \
  void OP_NAME::getEffects(                                                    \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>      \
          &effects) {                                                          \
    getEffectsImpl(effects, getDpsInputOperands(), getDpsInitsMutable());      \
  }

DEFINE_OP_GET_EFFECTS(ScatterOp)
DEFINE_OP_GET_EFFECTS(SortOp)
DEFINE_OP_GET_EFFECTS(FftOp)
DEFINE_OP_GET_EFFECTS(ReverseOp)
DEFINE_OP_GET_EFFECTS(ScanOp)
DEFINE_OP_GET_EFFECTS(TopkOp)
DEFINE_OP_GET_EFFECTS(PackOp)
DEFINE_OP_GET_EFFECTS(UnPackOp)
DEFINE_OP_GET_EFFECTS(WinogradInputTransformOp)
DEFINE_OP_GET_EFFECTS(WinogradFilterTransformOp)
DEFINE_OP_GET_EFFECTS(WinogradOutputTransformOp)
DEFINE_OP_GET_EFFECTS(AttentionOp)
DEFINE_OP_GET_EFFECTS(OnlineAttentionOp)
DEFINE_OP_GET_EFFECTS(Im2colOp)

} // namespace mlir::iree_compiler::IREE::LinalgExt

// clang-format off
#define GET_OP_CLASSES
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp.inc" // IWYU pragma: keep
// clang-format: on
