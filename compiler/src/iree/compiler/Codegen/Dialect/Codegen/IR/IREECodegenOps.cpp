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
#include "mlir/IR/DialectImplementation.h"
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
  auto sourceType = dyn_cast<MemRefType>(adaptor.getSource().getType());
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
    InnerTileDescAttrInterface kind, InnerTiledSemanticsAttrInterface semantics,
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
        kind, semantics, permutationsAttr);
}

void InnerTiledOp::build(
    OpBuilder &builder, OperationState &result, ValueRange inputs,
    ValueRange outputs, ArrayRef<ArrayRef<AffineExpr>> indexingExprs,
    ArrayRef<utils::IteratorType> iteratorTypes,
    InnerTileDescAttrInterface kind, InnerTiledSemanticsAttrInterface semantics,
    std::optional<SmallVector<SmallVector<int64_t>>> permutations) {
  SmallVector<AffineMap> indexingMaps =
      AffineMap::inferFromExprList(indexingExprs, builder.getContext());
  build(builder, result, inputs, outputs, indexingMaps, iteratorTypes, kind,
        semantics, permutations);
}

void InnerTiledOp::build(OpBuilder &builder, OperationState &result,
                         ValueRange inputs, ValueRange outputs,
                         ArrayAttr indexingMaps, ArrayAttr iteratorTypes,
                         InnerTileDescAttrInterface kind,
                         InnerTiledSemanticsAttrInterface semantics,
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
  inherentAttrs.setSemantics(semantics);
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

static bool tileShapesMatch(ArrayRef<int64_t> operandTile,
                            ArrayRef<int64_t> mmaTile, bool opaque) {
  if (opaque) {
    // Opaque tiles: only compare element counts.
    return llvm::product_of(operandTile) == llvm::product_of(mmaTile);
  }
  // Compare tile shapes for equality modulo unit dims.
  auto nonUnitDim = [](int64_t n) { return n != 1; };
  return llvm::filter_to_vector(operandTile, nonUnitDim) ==
         llvm::filter_to_vector(mmaTile, nonUnitDim);
}

static LogicalResult verifyOperandTypes(InnerTiledOp tiledOp) {
  SmallVector<VectorType> mmaVectorTypes;
  tiledOp.getSemantics().getTileTypes(tiledOp.getKind(), mmaVectorTypes);

  for (auto [index, tuple] : llvm::enumerate(llvm::zip_equal(
           tiledOp.getOperandTypes(),
           tiledOp.getIndexingMapsAttr().getAsValueRange<AffineMapAttr>(),
           mmaVectorTypes))) {
    auto [opType, map, mmaVectorType] = tuple;
    ShapedType operandShapedType = cast<ShapedType>(opType);
    Type operandElemType = operandShapedType.getElementType();
    Type mmaElemType = mmaVectorType.getElementType();
    if (operandElemType != mmaElemType) {
      return tiledOp.emitOpError(
          llvm::formatv("operand element type {} does not match expected MMA "
                        "tile element type {}",
                        operandElemType, mmaElemType));
    }
    ArrayRef<int64_t> operandShape = cast<ShapedType>(opType).getShape();
    ArrayRef<int64_t> operandTileShape(
        operandShape.drop_front(map.getNumResults()));
    ArrayRef<int64_t> mmaTileShape = mmaVectorType.getShape();
    SmallVector<int64_t> permutedMmaTileShape(mmaTileShape);
    std::optional<ArrayAttr> permutations = tiledOp.getPermutations();
    bool opaque = tiledOp.getSemantics().getOpaque();
    if (permutations && !opaque) {
      ArrayRef<int64_t> perm =
          cast<DenseI64ArrayAttr>(permutations.value()[index]).asArrayRef();
      applyPermutationToVector(permutedMmaTileShape, perm);
    }
    if (!tileShapesMatch(operandTileShape, permutedMmaTileShape, opaque)) {
      VectorType operandTileType =
          VectorType::get(operandTileShape, operandElemType);
      VectorType permutedMmaTileType =
          VectorType::get(permutedMmaTileShape, operandElemType);

      std::string permutationNote;
      if (permutations) {
        permutationNote = llvm::formatv(
            " Note: the permuted InnerTiledDescAttr tile type {} comes from "
            "applying the permutation {} to the original InnerTiledDescAttr "
            "tile type {}.",
            permutedMmaTileType, permutations.value()[index], mmaVectorType);
      }
      return tiledOp->emitOpError(llvm::formatv(
          "operand type {}, implying tile type {}, is incompatible with "
          "permuted InnerTiledDescAttr tile type {} under semantics {}.{}",
          opType, operandTileType, permutedMmaTileType, tiledOp.getSemantics(),
          permutationNote));
    }
  }
  return success();
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

  SmallVector<ShapedType> opTypes =
      llvm::map_to_vector(getOperandTypes(), llvm::CastTo<ShapedType>);
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

  if (getPermutations()) {
    auto permRange = getPermutations()->getAsRange<DenseI64ArrayAttr>();
    int permCount = permRange.end() - permRange.begin();
    if (permCount != opTypes.size()) {
      return emitOpError(
          llvm::formatv("mismatch between the number of permutations ({}) and "
                        "the number of operands ({})",
                        permCount, opTypes.size()));
    }
    for (auto [index, permAttr] : llvm::enumerate(permRange)) {
      ArrayRef<int64_t> perm = permAttr.asArrayRef();
      ShapedType operandType = getOperandShapedTypes()[index];
      int64_t rank = operandType.getRank();
      int64_t outerRank = getOperandOuterRank(index);
      int64_t innerRank = rank - outerRank;
      if (perm.size() != innerRank) {
        return emitOpError(
            llvm::formatv("permutation #{} length {} does not match the inner "
                          "rank {} of the corresponding operand of type {}",
                          index, perm.size(), innerRank, operandType));
      }
      if (!isPermutationVector(perm)) {
        return emitOpError(llvm::formatv(
            "permutation #{} is not a permutation vector", index));
      }
    }
  }

  return verifyOperandTypes(*this);
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

//===----------------------------------------------------------------------===//
// WorkgroupCountHintOp
//===----------------------------------------------------------------------===//

ParseResult WorkgroupCountHintOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 3> dynamicSizes;
  DenseI64ArrayAttr staticSizesAttr;

  if (parseDynamicIndexList(parser, dynamicSizes, staticSizesAttr,
                            /*valueTypes=*/{},
                            /*delimiter=*/AsmParser::Delimiter::Paren)) {
    return failure();
  }

  // All sizes are of index type. `parseDynamicIndexList` does not set the sizes
  // correctly when used as a custom directive so manually infer it from the
  // number of parsed sizes.
  IndexType indexType = parser.getBuilder().getIndexType();
  SmallVector<Type> dynamicSizeTypes(dynamicSizes.size(), indexType);

  if (parser.resolveOperands(dynamicSizes, dynamicSizeTypes,
                             parser.getCurrentLocation(), result.operands)) {
    return failure();
  }

  result.addAttribute("static_sizes", staticSizesAttr);
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  return success();
}

void WorkgroupCountHintOp::print(OpAsmPrinter &printer) {
  printDynamicIndexList(printer, getOperation(), getSizes(), getStaticSizes(),
                        /*valueTypes=*/{},
                        /*delimiter=*/AsmParser::Delimiter::Paren);
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                /*elidedAttrs=*/{"static_sizes"});
}

void WorkgroupCountHintOp::build(OpBuilder &builder, OperationState &state,
                                 ArrayRef<OpFoldResult> sizes) {
  SmallVector<int64_t> staticSizes;
  SmallVector<Value> dynamicSizes;
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  build(builder, state, dynamicSizes,
        builder.getDenseI64ArrayAttr(staticSizes));
}
