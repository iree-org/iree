// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"

// clang-format off
#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.cpp.inc" // IWYU pragma: keep
// clang-format on

using namespace mlir;
using namespace mlir::iree_compiler::IREE::Codegen;
namespace IREE = mlir::iree_compiler::IREE;

//===----------------------------------------------------------------------===//
// ExtractStridedMetadataOp
//===----------------------------------------------------------------------===//

/// The number and type of the results are inferred from the
/// shape of the source.
LogicalResult ExtractStridedMetadataOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    ExtractStridedMetadataOp::Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto sourceType = llvm::dyn_cast<MemRefType>(adaptor.getSource().getType());
  if (!sourceType)
    return failure();

  unsigned sourceRank = sourceType.getRank();
  IndexType indexType = IndexType::get(context);
  auto memrefType =
      MemRefType::get({}, sourceType.getElementType(),
                      MemRefLayoutAttrInterface{}, sourceType.getMemorySpace());
  // Base.
  inferredReturnTypes.push_back(memrefType);
  // Offset.
  inferredReturnTypes.push_back(indexType);
  // Sizes and strides.
  for (unsigned i = 0; i < sourceRank * 2; ++i)
    inferredReturnTypes.push_back(indexType);
  return success();
}

void ExtractStridedMetadataOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getBaseBuffer(), "base_buffer");
  setNameFn(getOffset(), "offset");
  // For multi-result to work properly with pretty names and packed syntax `x:3`
  // we can only give a pretty name to the first value in the pack.
  if (!getSizes().empty()) {
    setNameFn(getSizes().front(), "sizes");
    setNameFn(getStrides().front(), "strides");
  }
}

//===----------------------------------------------------------------------===//
// LoadFromBufferOp
//===----------------------------------------------------------------------===//

LogicalResult LoadFromBufferOp::verify() {
  RankedTensorType tensorType = getTensor().getType();
  MemRefType memrefType = getBuffer().getType();
  if (failed(verifyCompatibleShape(tensorType.getShape(),
                                   memrefType.getShape())) ||
      tensorType.getElementType() != memrefType.getElementType()) {
    return emitOpError("buffer and tensor shapes must be compatible and "
                       "element types must match");
  }
  return success();
}

void LoadFromBufferOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getBufferMutable(),
                       SideEffects::DefaultResource::get());
}

LogicalResult LoadFromBufferOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointAfterValue(getBuffer());
  reifiedReturnShapes.resize(1);
  reifiedReturnShapes[0] = memref::getMixedSizes(b, getLoc(), getBuffer());
  return success();
}

//===----------------------------------------------------------------------===//
// StoreToBufferOp
//===----------------------------------------------------------------------===//

LogicalResult StoreToBufferOp::verify() {
  RankedTensorType tensorType = getTensor().getType();
  MemRefType memrefType = getBuffer().getType();
  if (failed(verifyCompatibleShape(tensorType.getShape(),
                                   memrefType.getShape())) ||
      tensorType.getElementType() != memrefType.getElementType()) {
    return emitOpError("tensor and buffer shapes must be compatible and "
                       "element types must match");
  }
  return success();
}

void StoreToBufferOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getBufferMutable(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// InnerTiledOp
//===----------------------------------------------------------------------===//

void InnerTiledOp::build(
    OpBuilder &builder, OperationState &result, ValueRange inputs,
    ValueRange outputs, ArrayRef<AffineMap> indexingMaps,
    ArrayRef<utils::IteratorType> iteratorTypes,
    InnerTileDescAttrInterface kind,
    std::optional<SmallVector<SmallVector<int64_t>>> permutations) {
  ArrayAttr indexingMapsAttr = builder.getAffineMapArrayAttr(indexingMaps);
  ArrayAttr iteratorTypesAttr = builder.getArrayAttr(llvm::map_to_vector(
      iteratorTypes, [&](utils::IteratorType t) -> mlir::Attribute {
        return linalg::IteratorTypeAttr::get(builder.getContext(), t);
      }));
  std::optional<ArrayAttr> permutationsAttr;
  if (permutations) {
    permutationsAttr = builder.getArrayAttr(llvm::map_to_vector(
        *permutations, [&](const SmallVector<int64_t> &perm) -> Attribute {
          return builder.getDenseI64ArrayAttr(perm);
        }));
  }
  build(builder, result, inputs, outputs, indexingMapsAttr, iteratorTypesAttr,
        kind, permutationsAttr);
}

void InnerTiledOp::build(
    OpBuilder &builder, OperationState &result, ValueRange inputs,
    ValueRange outputs, ArrayRef<ArrayRef<AffineExpr>> indexingExprs,
    ArrayRef<utils::IteratorType> iteratorTypes,
    InnerTileDescAttrInterface kind,
    std::optional<SmallVector<SmallVector<int64_t>>> permutations) {
  SmallVector<AffineMap> indexingMaps =
      AffineMap::inferFromExprList(indexingExprs, builder.getContext());
  build(builder, result, inputs, outputs, indexingMaps, iteratorTypes, kind,
        permutations);
}

void InnerTiledOp::build(OpBuilder &builder, OperationState &result,
                         ValueRange inputs, ValueRange outputs,
                         ArrayAttr indexingMaps, ArrayAttr iteratorTypes,
                         InnerTileDescAttrInterface kind,
                         std::optional<ArrayAttr> permutations) {
  result.addOperands(inputs);
  result.addOperands(outputs);
  result.addTypes(outputs.getTypes());
  Properties &inherentAttrs = result.getOrAddProperties<Properties>();
  inherentAttrs.setOperandSegmentSizes(
      {static_cast<int>(inputs.size()), static_cast<int>(outputs.size())});
  inherentAttrs.setIndexingMaps(indexingMaps);
  inherentAttrs.setIteratorTypes(iteratorTypes);
  inherentAttrs.setKind(kind);
  if (permutations) {
    inherentAttrs.setPermutations(*permutations);
  }
}

// Note: we can't use an "AllTypesMatch" constraint because it will cause an
// inferReturnTypes() method that doesn't understand variadic inputs to
// be generated.
LogicalResult
InnerTiledOp::inferReturnTypes(MLIRContext *, std::optional<Location>,
                               Adaptor adaptor,
                               SmallVectorImpl<Type> &inferredReturnTypes) {
  llvm::append_range(inferredReturnTypes, adaptor.getOutputs().getTypes());
  return success();
}

// This helper struct captures the traits of a tile shape that are relevant to
// matching an inner_tiled op as distributed or undistributed.
struct TileMatchingTraits {
  // Currently the only trait of a tile that is used in matching inner_tiled op
  // semantics, is the element count. The reason why we don't simply match the
  // full tile shape, or even just the rank, is that lots of tests rely on
  // tolerating unit dimensions.
  int64_t elemCount;
};

// This helper struct captures a diagnostic about a failure to match a
// inner_tiled op's tiles as either distributed or undistributed.
struct TileMismatchDiagnostic {
  // Index of operand that the diagnostic refers to.
  int index;
  // An opaque score assigned to the diagnosed mismatch. A higher value means
  // a larger mismatch. This is used to guess whether distributed or
  // undistributed semantics were more likely intended, for the purpose of
  // generating a more helpful diagnostic.
  int magnitude;
};

static std::optional<TileMismatchDiagnostic>
getFirstMismatch(ArrayRef<TileMatchingTraits> tileMatchingTraits,
                 ArrayRef<VectorType> tileTypes) {
  for (auto [i, mt, ty] : llvm::enumerate(tileMatchingTraits, tileTypes)) {
    if (!(mt.elemCount == ty.getNumElements())) {
      // Extremely crude additive-difference estimate. Obviously should be a
      // multiplicative estimate, but maybe there's no need to overthink this.
      // What likely matters is that this magnitude is 0 if and only if the
      // elemCount is correct. This should suffice to determine whether to
      // emit a diagnostic based or distributed or undistributed semantics.
      int magnitude = std::abs(mt.elemCount - ty.getNumElements());
      return TileMismatchDiagnostic{static_cast<int>(i), magnitude};
    }
  }
  return {};
}

static SmallVector<TileMatchingTraits>
getTileMatchingTraits(InnerTiledOp tiledOp) {
  SmallVector<TileMatchingTraits> result;
  result.reserve(tiledOp.getNumOperands());
  for (auto [opType, map] : llvm::zip_equal(
           tiledOp.getOperandTypes(),
           tiledOp.getIndexingMapsAttr().getAsValueRange<AffineMapAttr>())) {
    ArrayRef<int64_t> shape = cast<ShapedType>(opType).getShape();
    ArrayRef<int64_t> innerShape = shape.drop_front(map.getNumResults());
    int64_t innerElemsCount = llvm::product_of(innerShape);
    result.push_back({innerElemsCount});
  }
  return result;
}

LogicalResult InnerTiledOp::verify() {
  int64_t expectedNumIns = getKind().getExpectedNumInputs();
  if (expectedNumIns != getNumInputs()) {
    return emitOpError("number of inputs (" + Twine(getNumInputs()) +
                       ") doesn't match expected number from kind (" +
                       Twine(expectedNumIns) + ")");
  }
  int64_t expectedNumOuts = getKind().getExpectedNumOutputs();
  if (expectedNumOuts != getNumOutputs()) {
    return emitOpError("number of outputs (" + Twine(getNumOutputs()) +
                       ") doesn't match expected number from kind (" +
                       Twine(expectedNumOuts) + ")");
  }

  SmallVector<ShapedType> opTypes = llvm::map_to_vector(
      getOperandTypes(), [](auto t) { return llvm::cast<ShapedType>(t); });
  SmallVector<AffineMap, 4> indexingMaps = getIndexingMapsArray();

  // Verify that an indexing map was specified for each operand.
  if (indexingMaps.size() != expectedNumIns + expectedNumOuts)
    return emitOpError("expected an indexing map for each operand");

  // Verify that each index map has 'numIterators' inputs, no symbols, and
  // that the number of map outputs equals the rank of its associated
  // vector operand.
  unsigned numIterators = getIteratorTypes().getValue().size();
  for (const auto &it : llvm::enumerate(indexingMaps)) {
    auto index = it.index();
    auto map = it.value();
    if (map.getNumSymbols() != 0)
      return emitOpError("expected indexing map ")
             << index << " to have no symbols";
    auto shapedType = opTypes[index];
    unsigned rank = shapedType.getRank();
    // Verify that the map has the right number of inputs, outputs, and indices.
    // This also correctly accounts for (..) -> () for rank-0 results.
    if (map.getNumDims() != numIterators) {
      return emitOpError("expected indexing map ")
             << index << " to have " << numIterators << " input dims";
    }
    if (map.getNumResults() >= rank) {
      return emitOpError("expected indexing map ")
             << index << " to have fewer than " << rank << " results";
    }
    if (!map.isProjectedPermutation()) {
      return emitOpError("expected indexing map ")
             << index << " to be a projected permutation";
    }

    for (int64_t size :
         shapedType.getShape().take_back(rank - map.getNumResults())) {
      if (ShapedType::isDynamic(size)) {
        return emitOpError("Unexpected dynamic inner dim for operand ")
               << index << " of type " << shapedType;
      }
    }
  }

  if (failed(getKind().verifyIndexingMaps(indexingMaps))) {
    return emitOpError("failed to verify indexing maps");
  }

  SmallVector<int64_t> bounds;
  getIterationBounds(bounds);
  for (auto [type, map] : llvm::zip_equal(opTypes, indexingMaps)) {
    // The truncation functionality of llvm::zip is intentional here to ignore
    // the inner dimensions.
    for (auto [dim, size] : llvm::zip(map.getResults(), type.getShape())) {
      int64_t dimIdx = cast<AffineDimExpr>(dim).getPosition();
      if (size != bounds[dimIdx]) {
        return emitOpError("shape does not match iteration bounds");
      }
    }
  }

  SmallVector<VectorType> preThreadTypes;
  getKind().getUndistributedTileTypes(preThreadTypes);
  SmallVector<VectorType> threadTypes;
  getKind().getDistributedTileTypes(threadTypes);

  SmallVector<TileMatchingTraits> tileMatchingTraits =
      getTileMatchingTraits(*this);
  for (auto [opNum, opType, tileType] :
       llvm::enumerate(opTypes, preThreadTypes)) {
    if (opType.getElementType() != tileType.getElementType()) {
      return emitOpError("operand " + Twine(opNum) + " element type ")
             << opType.getElementType() << " does not match expected tile type "
             << tileType << " for operator";
    }
  }

  std::optional<TileMismatchDiagnostic> undistributedMismatch =
      getFirstMismatch(tileMatchingTraits, preThreadTypes);
  std::optional<TileMismatchDiagnostic> distributedMismatch =
      getFirstMismatch(tileMatchingTraits, threadTypes);

  bool hasUndistributedSemantics = !undistributedMismatch;
  bool hasDistributedSemantics = !distributedMismatch;
  if (!hasUndistributedSemantics && !hasDistributedSemantics) {
    // When both semantics mismatch, we know we have an invalid op. We need
    // to decide whether to emit an op based on an assumption of intended
    // distributed or undistributed semantics. The small `magnitude` mismatch
    // points to the more likely intended semantics. In case of equality,
    // default to assuming undistributed semantics as that is how ops come in
    // most of the time.
    bool diagnoseAsDistributed =
        undistributedMismatch->magnitude > distributedMismatch->magnitude;
    StringRef semantics =
        diagnoseAsDistributed ? "Distributed" : "Undistributed";
    const TileMismatchDiagnostic &diag =
        diagnoseAsDistributed ? *distributedMismatch : *undistributedMismatch;
    Type operandType = getOperandTypes()[diag.index];
    int elemCount = tileMatchingTraits[diag.index].elemCount;
    VectorType tileType = diagnoseAsDistributed ? threadTypes[diag.index]
                                                : preThreadTypes[diag.index];
    return emitOpError(llvm::formatv(
        "tile shapes do not match, regardless of semantics. The semantics "
        "that this op comes closest to matching is: {}. "
        "However, even under that assumption, there is still a mismatch in "
        "operand {}, which has type {}, whose inner tile dimensions have an "
        "element count of {}, which doesn't match the target "
        "vector type {}.",
        semantics, diag.index, operandType, elemCount, tileType));
  }
  if (hasDistributedSemantics) {
    if (getPermutations()) {
      return emitOpError("permutations require undistributed semantics");
    }
  }

  if (getPermutations()) {
    for (auto permAttr : getPermutations()->getAsRange<DenseI64ArrayAttr>()) {
      if (!isPermutationVector(permAttr.asArrayRef()))
        return emitOpError("invalid permutation");
    }
  }

  return success();
}

bool InnerTiledOp::hasThreadSemantics() {
  SmallVector<TileMatchingTraits> innerElemCountsAndRanks =
      getTileMatchingTraits(*this);
  SmallVector<VectorType> preThreadTiles;
  getKind().getUndistributedTileTypes(preThreadTiles);
  return !!getFirstMismatch(innerElemCountsAndRanks, preThreadTiles);
}

static int64_t getResultIndex(AffineMap map, AffineExpr targetExpr) {
  for (int64_t i = 0, e = map.getNumResults(); i < e; ++i)
    if (targetExpr == map.getResult(i))
      return i;
  return -1;
}

void InnerTiledOp::getIterationBounds(
    SmallVectorImpl<int64_t> &iterationBounds) {
  SmallVector<ShapedType> operandTypes = getOperandShapedTypes();
  SmallVector<AffineMap, 4> indexingMaps(getIndexingMapsArray());
  AffineMap combinedMap = concatAffineMaps(indexingMaps, getContext());
  SmallVector<int64_t> combinedOuterShapes;
  for (auto [opType, map] : llvm::zip_equal(operandTypes, indexingMaps)) {
    llvm::append_range(combinedOuterShapes,
                       opType.getShape().take_front(map.getNumResults()));
  }
  AffineMap inverseMap = inversePermutation(combinedMap);
  iterationBounds.append(inverseMap.compose(combinedOuterShapes));
}

std::optional<SmallVector<int64_t, 4>> InnerTiledOp::getShapeForUnroll() {
  SmallVector<int64_t, 4> shape;
  getIterationBounds(shape);
  return shape;
}
