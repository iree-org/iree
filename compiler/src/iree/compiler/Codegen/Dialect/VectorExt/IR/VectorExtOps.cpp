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

VectorType TransferGatherOp::getVectorType() { return getType(); }

Speculation::Speculatability TransferGatherOp::getSpeculatability() {
  if (isa<RankedTensorType>(getBase().getType())) {
    return Speculation::Speculatable;
  }
  return Speculation::NotSpeculatable;
}

void TransferGatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (llvm::isa<MemRefType>(getBase().getType())) {
    effects.emplace_back(MemoryEffects::Read::get(), &getBaseMutable(),
                         SideEffects::DefaultResource::get());
  }
}

const char *kNonIndexedSymbol = "None";

static ParseResult
parseIndexVecs(OpAsmParser &parser,
               SmallVectorImpl<OpAsmParser::UnresolvedOperand> &indexVecs,
               SmallVectorImpl<Type> &indexVecTypes, ArrayAttr &indexed) {
  if (parser.parseLSquare())
    return failure();

  SMLoc loc;
  SmallVector<bool> indexedArr;
  while (parser.parseOptionalRSquare()) {
    // Check if this dimension is not indexed
    if (!parser.parseOptionalKeyword(kNonIndexedSymbol)) {
      indexedArr.push_back(false);
      (void)parser.parseOptionalComma();
      continue;
    }

    OpAsmParser::UnresolvedOperand indexVec;
    Type indexVecType;
    if (parser.getCurrentLocation(&loc) || parser.parseOperand(indexVec) ||
        parser.parseColon() || parser.parseType(indexVecType)) {
      return parser.emitError(loc, "Expected `none` or `operand : type`");
    }

    indexVecs.push_back(indexVec);
    indexVecTypes.push_back(indexVecType);
    indexedArr.push_back(true);

    (void)parser.parseOptionalComma();
  }

  // OpBuilder is only used as a helper to build an BoolArrayAttr.
  Builder b(parser.getContext());
  indexed = b.getBoolArrayAttr(indexedArr);
  return success();
}

static void printIndexVecs(OpAsmPrinter &p, Operation *op,
                           OperandRange indexVecs, TypeRange indexVecTypes,
                           ArrayAttr indexed) {
  int64_t rank = indexed.size();
  SmallVector<bool> indexedArr(indexed.getAsValueRange<BoolAttr>());

  int64_t currIndexDim = 0;
  p << "[";
  for (int64_t i : llvm::seq<int64_t>(rank)) {
    if (indexedArr[i]) {
      p << indexVecs[currIndexDim] << ": " << indexVecTypes[currIndexDim];
      ++currIndexDim;
    } else {
      p << kNonIndexedSymbol;
    }

    if (i != rank - 1) {
      p << ", ";
    }
  }
  p << "]";
}

static void printTransferAttrs(OpAsmPrinter &p, VectorTransferOpInterface op,
                               SmallVector<StringRef, 3> elidedAttrs = {}) {
  elidedAttrs.push_back(TransferGatherOp::getOperandSegmentSizeAttr());
  if (op.getPermutationMap().isMinorIdentity())
    elidedAttrs.push_back(op.getPermutationMapAttrName());
  // Elide in_bounds attribute if all dims are out-of-bounds.
  if (llvm::none_of(op.getInBoundsValues(), [](bool b) { return b; }))
    elidedAttrs.push_back(op.getInBoundsAttrName());
  p.printOptionalAttrDict(op->getAttrs(), elidedAttrs);
}

void TransferGatherOp::print(OpAsmPrinter &p) {
  p << " " << getBase() << "[" << getIndices() << "]";
  printIndexVecs(p, *this, getIndexVecs(), getIndexVecs().getTypes(),
                 getIndexedAttr());
  p << ", " << getPadding();
  if (getMask())
    p << ", " << getMask();
  printTransferAttrs(p, *this, {"indexed"});
  p << " : " << getShapedType() << ", " << getType();
}

static LogicalResult
verifyTransferOp(VectorTransferOpInterface op, ShapedType shapedType,
                 VectorType vectorType, VectorType maskType,
                 VectorType inferredMaskType, AffineMap permutationMap,
                 ArrayAttr inBounds) {
  if (!llvm::isa<MemRefType, RankedTensorType>(shapedType))
    return op->emitOpError(
        "requires source to be a memref or ranked tensor type");

  Type elementType = shapedType.getElementType();
  DataLayout dataLayout = DataLayout::closest(op);
  if (auto vectorElementType = llvm::dyn_cast<VectorType>(elementType)) {
    // Memref or tensor has vector element type.
    unsigned sourceVecSize =
        dataLayout.getTypeSizeInBits(vectorElementType.getElementType()) *
        vectorElementType.getShape().back();
    unsigned resultVecSize =
        dataLayout.getTypeSizeInBits(vectorType.getElementType()) *
        vectorType.getShape().back();
    if (resultVecSize % sourceVecSize != 0) {
      return op->emitOpError(
          "requires the bitwidth of the minor 1-D vector to be an integral "
          "multiple of the bitwidth of the minor 1-D vector of the source");
    }

    unsigned sourceVecEltRank = vectorElementType.getRank();
    unsigned resultVecRank = vectorType.getRank();
    if (sourceVecEltRank > resultVecRank) {
      return op->emitOpError(
          "requires source vector element and vector result ranks to match.");
    }
    unsigned rankOffset = resultVecRank - sourceVecEltRank;
    // Check that permutation map results match 'rankOffset' of vector type.
    if (permutationMap.getNumResults() != rankOffset) {
      return op->emitOpError("requires a permutation_map with result dims of "
                             "the same rank as the vector type");
    }

    if (maskType) {
      return op->emitOpError("does not support masks with vector element type");
    }
  } else {
    // Memref or tensor has scalar element type.
    unsigned minorSize =
        vectorType.getRank() == 0 ? 1 : vectorType.getShape().back();
    unsigned resultVecSize =
        dataLayout.getTypeSizeInBits(vectorType.getElementType()) * minorSize;
    if (resultVecSize % dataLayout.getTypeSizeInBits(elementType) != 0) {
      return op->emitOpError(
          "requires the bitwidth of the minor 1-D vector to be an integral "
          "multiple of the bitwidth of the source element type");
    }

    // Check that permutation map results match rank of vector type.
    if (permutationMap.getNumResults() != vectorType.getRank()) {
      return op->emitOpError("requires a permutation_map with result dims of "
                             "the same rank as the vector type");
    }
  }

  if (permutationMap.getNumSymbols() != 0) {
    return op->emitOpError("requires permutation_map without symbols");
  }

  if (permutationMap.getNumInputs() != shapedType.getRank()) {
    return op->emitOpError("requires a permutation_map with input dims of the "
                           "same rank as the source type");
  }

  if (maskType && maskType != inferredMaskType) {
    return op->emitOpError("inferred mask type (")
           << inferredMaskType << ") and mask operand type (" << maskType
           << ") don't match";
  }

  if (permutationMap.getNumResults() != static_cast<int64_t>(inBounds.size())) {
    return op->emitOpError("expects the in_bounds attr of same rank "
                           "as permutation_map results: ")
           << AffineMapAttr::get(permutationMap)
           << " vs inBounds of size: " << inBounds.size();
  }

  return success();
}

static LogicalResult verifyPermutationMap(
    AffineMap permutationMap,
    llvm::function_ref<InFlightDiagnostic(Twine)> emitOpError) {
  SmallVector<bool, 8> seen(permutationMap.getNumInputs(), false);
  for (auto expr : permutationMap.getResults()) {
    auto dim = dyn_cast<AffineDimExpr>(expr);
    auto zero = dyn_cast<AffineConstantExpr>(expr);
    if (zero) {
      if (zero.getValue() != 0) {
        return emitOpError("requires a projected permutation_map (at most "
                           "one dim or the zero "
                           "constant can appear in each result)");
      }
      continue;
    }
    if (!dim) {
      return emitOpError("requires a projected permutation_map (at most one "
                         "dim or the zero constant can appear in each result)");
    }
    if (seen[dim.getPosition()]) {
      return emitOpError(
          "requires a permutation_map that is a permutation (found one dim "
          "used more than once)");
    }
    seen[dim.getPosition()] = true;
  }
  return success();
}

LogicalResult TransferGatherOp::verify() {
  // Consistency of elemental types in source and vector.
  ShapedType shapedType = getShapedType();
  VectorType vectorType = getType();
  VectorType maskType = getMaskType();
  Type paddingType = getPadding().getType();
  AffineMap permutationMap = getPermutationMap();
  VectorType inferredMaskType =
      maskType ? vector::inferTransferOpMaskType(vectorType, permutationMap)
               : VectorType();
  auto sourceElementType = shapedType.getElementType();

  if (static_cast<int64_t>(getIndices().size()) != shapedType.getRank())
    return emitOpError("requires ") << shapedType.getRank() << " indices";

  if (failed(verifyTransferOp(cast<VectorTransferOpInterface>(getOperation()),
                              shapedType, vectorType, maskType,
                              inferredMaskType, permutationMap, getInBounds())))
    return failure();

  if (auto sourceVectorElementType =
          llvm::dyn_cast<VectorType>(sourceElementType)) {
    // Source has vector element type.
    // Check that 'sourceVectorElementType' and 'paddingType' types match.
    if (sourceVectorElementType != paddingType)
      return emitOpError(
          "requires source element type and padding type to match.");

  } else {
    // Check that 'paddingType' is valid to store in a vector type.
    if (!VectorType::isValidElementType(paddingType))
      return emitOpError("requires valid padding vector elemental type");

    // Check that padding type and vector element types match.
    if (paddingType != sourceElementType)
      return emitOpError(
          "requires formal padding and source of the same elemental type");
  }

  if (failed(verifyPermutationMap(permutationMap,
                                  [&](Twine t) { return emitOpError(t); }))) {
    return failure();
  }

  OperandRange indexVecs = getIndexVecs();
  SmallVector<AffineMap> indexedMaps = getIndexedMapsArray();
  SmallVector<bool> indexed(getIndexed().getAsValueRange<BoolAttr>());
  if (indexed.size() != shapedType.getRank()) {
    return emitOpError("requires indexed to have a value for each dimension of "
                       "the shaped type");
  }

  int64_t numIndexed = llvm::count(indexed, true);
  if (numIndexed != indexVecs.size()) {
    return emitOpError("requires ") << numIndexed << " index vectors";
  }
  if (indexVecs.size() != indexedMaps.size()) {
    return emitOpError("requires ") << numIndexed << " indexing map";
  }

  SmallVector<int64_t> sliceSize = getTransferChunkAccessed();
  for (auto [i, indexVec, indexMap] : llvm::enumerate(indexVecs, indexedMaps)) {
    auto indexVecTy = cast<VectorType>(indexVec.getType());
    if (indexVecTy.getRank() != indexMap.getNumResults()) {
      return emitOpError(
                 "requires number of results for corressponding "
                 "indexing map to match the rank of index vector at dim: ")
             << i;
    }

    SmallVector<int64_t> expectedShape =
        applyPermutationMap(indexMap, ArrayRef(sliceSize));
    if (expectedShape != indexVecTy.getShape()) {
      return emitOpError("Invalid index vector shape at dim: ")
             << i << ", expected: " << expectedShape
             << ", got: " << indexVecTy.getShape();
    }
  }

  return success();
}

ParseResult TransferGatherOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  auto &builder = parser.getBuilder();
  SMLoc typesLoc;
  OpAsmParser::UnresolvedOperand sourceInfo;
  SmallVector<OpAsmParser::UnresolvedOperand, 8> indexInfo;
  SmallVector<OpAsmParser::UnresolvedOperand, 8> indexVecInfo;
  OpAsmParser::UnresolvedOperand paddingInfo;
  SmallVector<Type, 2> types;
  OpAsmParser::UnresolvedOperand maskInfo;
  // Parsing with support for paddingValue.
  if (parser.parseOperand(sourceInfo) ||
      parser.parseOperandList(indexInfo, OpAsmParser::Delimiter::Square))
    return failure();

  SmallVector<Type, 2> indexVecTypes;
  ArrayAttr indexed;
  if (parseIndexVecs(parser, indexVecInfo, indexVecTypes, indexed))
    return failure();
  result.addAttribute("indexed", indexed);

  if (parser.parseComma() || parser.parseOperand(paddingInfo))
    return failure();

  ParseResult hasMask = parser.parseOptionalComma();
  if (hasMask.succeeded()) {
    if (parser.parseOperand(maskInfo))
      return failure();
  }

  // Parse attributes and types.
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || parser.parseColonTypeList(types))
    return failure();

  // Check if number of types given are correct.
  int64_t nRequiredTypes = 2;
  if (types.size() != nRequiredTypes) {
    return parser.emitError(typesLoc, "expected ")
           << nRequiredTypes << " types";
  }

  // The types are arranged as:
  // sourceTy, resultTy
  auto shapedType = llvm::dyn_cast<ShapedType>(types[0]);
  VectorType vectorType = llvm::dyn_cast<VectorType>(types[1]);
  if (!shapedType || !llvm::isa<MemRefType, RankedTensorType>(shapedType))
    return parser.emitError(typesLoc, "requires memref or ranked tensor type");
  if (!vectorType)
    return parser.emitError(typesLoc, "requires vector type");
  auto permMapAttrName =
      TransferGatherOp::getPermutationMapAttrName(result.name);
  Attribute permMapAttr = result.attributes.get(permMapAttrName);
  AffineMap permMap;
  if (!permMapAttr) {
    permMap = vector::getTransferMinorIdentityMap(shapedType, vectorType);
    result.attributes.set(permMapAttrName, AffineMapAttr::get(permMap));
  } else {
    permMap = llvm::cast<AffineMapAttr>(permMapAttr).getValue();
  }
  auto inBoundsAttrName = TransferGatherOp::getInBoundsAttrName(result.name);
  Attribute inBoundsAttr = result.attributes.get(inBoundsAttrName);
  if (!inBoundsAttr) {
    result.addAttribute(inBoundsAttrName,
                        builder.getBoolArrayAttr(
                            SmallVector<bool>(permMap.getNumResults(), false)));
  }
  if (parser.resolveOperand(sourceInfo, shapedType, result.operands) ||
      parser.resolveOperands(indexInfo, builder.getIndexType(),
                             result.operands) ||
      parser.resolveOperands(indexVecInfo, indexVecTypes, typesLoc,
                             result.operands) ||
      parser.resolveOperand(paddingInfo, shapedType.getElementType(),
                            result.operands))
    return failure();
  if (hasMask.succeeded()) {
    if (llvm::dyn_cast<VectorType>(shapedType.getElementType()))
      return parser.emitError(
          maskInfo.location, "does not support masks with vector element type");
    if (vectorType.getRank() != permMap.getNumResults()) {
      return parser.emitError(typesLoc,
                              "expected the same rank for the vector and the "
                              "results of the permutation map");
    }
    // Instead of adding the mask type as an op type, compute it based on the
    // vector type and the permutation map (to keep the type signature small).
    auto maskType = vector::inferTransferOpMaskType(vectorType, permMap);
    if (parser.resolveOperand(maskInfo, maskType, result.operands))
      return failure();
  }
  result.addAttribute(TransferGatherOp::getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr(
                          {1, static_cast<int32_t>(indexInfo.size()),
                           static_cast<int32_t>(indexVecInfo.size()), 1,
                           static_cast<int32_t>(hasMask.succeeded())}));
  return parser.addTypeToList(vectorType, result.types);
}

static int64_t getVectorRank(Type type) {
  if (auto vecType = dyn_cast<VectorType>(type)) {
    return vecType.getRank();
  }
  return 0;
}

struct IndexVecFoldResult {
  Value indexVec;
  AffineMap indexMap;
  bool changed;
};

static Value foldTransferGatherIndexVecs(
    TransferGatherOp gatherOp,
    function_ref<IndexVecFoldResult(Value, AffineMap, int64_t)>
        indexVecFolder) {
  bool changed = false;
  SmallVector<Value> newIndexVecs;
  SmallVector<AffineMap> newIndexedMaps;
  SmallVector<bool> indexed(gatherOp.getIndexed().getAsValueRange<BoolAttr>());
  int64_t currIndexVec = 0;
  for (auto i : llvm::seq<int64_t>(gatherOp.getIndices().size())) {
    if (!indexed[i]) {
      continue;
    }
    Value operand = gatherOp.getIndexVecs()[currIndexVec];
    AffineMap map = gatherOp.getIndexedMapsArray()[currIndexVec];
    ++currIndexVec;

    auto [indexVec, indexMap, vecChanged] = indexVecFolder(operand, map, i);
    changed |= vecChanged;

    if (indexVec) {
      newIndexVecs.push_back(indexVec);
      newIndexedMaps.push_back(indexMap);
      indexed[i] = true;
    } else {
      indexed[i] = false;
    }
  }

  if (!changed) {
    return Value();
  }

  OpBuilder b(gatherOp);

  SmallVector<Value> operands;
  SmallVector<int32_t> operandSegmentSizes;

  // Source.
  operands.push_back(gatherOp.getBase());
  operandSegmentSizes.push_back(1);
  // Indices.
  SmallVector<Value> indices = gatherOp.getIndices();
  operands.append(indices);
  operandSegmentSizes.push_back(indices.size());
  // IndexVecs.
  operands.append(newIndexVecs);
  operandSegmentSizes.push_back(newIndexVecs.size());
  // Padding.
  operands.push_back(gatherOp.getPadding());
  operandSegmentSizes.push_back(1);
  // Mask.
  if (gatherOp.getMask()) {
    operands.push_back(gatherOp.getMask());
    operandSegmentSizes.push_back(1);
  } else {
    operandSegmentSizes.push_back(0);
  }

  gatherOp.setIndexedMapsAttr(b.getAffineMapArrayAttr(newIndexedMaps));
  gatherOp->setOperands(operands);
  gatherOp.setIndexedAttr(b.getBoolArrayAttr(indexed));
  gatherOp.getProperties().setOperandSegmentSizes(operandSegmentSizes);

  return gatherOp.getResult();
}

static Value foldTransferGatherFromBroadcast(TransferGatherOp gatherOp) {
  return foldTransferGatherIndexVecs(
      gatherOp,
      [](Value operand, AffineMap map, int64_t) -> IndexVecFoldResult {
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

static Value foldTransferGatherFromTranspose(TransferGatherOp gatherOp) {
  return foldTransferGatherIndexVecs(
      gatherOp,
      [](Value operand, AffineMap map, int64_t) -> IndexVecFoldResult {
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

static Value foldTransferGatherFromStep(TransferGatherOp gatherOp) {
  return foldTransferGatherIndexVecs(
      gatherOp,
      [](Value operand, AffineMap map, int64_t index) -> IndexVecFoldResult {
        auto step = operand.getDefiningOp<vector::StepOp>();
        if (!step) {
          return {operand, map, false};
        }

        assert(map.getNumResults() == 1);
        int64_t resultDim = cast<AffineDimExpr>(map.getResult(0)).getPosition();

        // If the map is indexing along the memory dimension, and the vector is
        // contiguous, this is a contiguous load on this dimension.
        if (resultDim == index) {
          return {Value(), AffineMap(), true};
        }

        return {operand, map, false};
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

struct FoldSingleElementIndexVec final : OpRewritePattern<TransferGatherOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TransferGatherOp xferOp,
                                PatternRewriter &rewriter) const override {

    auto indexVecFolder = [&](Value indexVec, AffineMap map,
                              int64_t index) -> IndexVecFoldResult {
      auto vectorTy = cast<VectorType>(indexVec.getType());
      if (vectorTy.getNumElements() != 1) {
        return {indexVec, map, false};
      }

      // Extract the scalar and add it to the
      // corressponding base.
      OpOperand &base = xferOp.getIndicesMutable()[index];
      Value extracted = rewriter.create<vector::ExtractOp>(
          xferOp.getLoc(), indexVec,
          SmallVector<int64_t>(vectorTy.getRank(), 0));
      AffineExpr d0, d1;
      bindDims(xferOp.getContext(), d0, d1);
      Value newIndex = affine::makeComposedAffineApply(
                           rewriter, xferOp.getLoc(), d0 + d1,
                           ArrayRef<OpFoldResult>{base.get(), extracted})
                           .getResult();
      base.set(newIndex);

      return {Value(), AffineMap(), true};
    };

    Value newVal = foldTransferGatherIndexVecs(xferOp, indexVecFolder);

    if (!newVal) {
      return failure();
    }

    return success();
  }
};

struct FoldContigousGatherToTransferRead final
    : OpRewritePattern<TransferGatherOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TransferGatherOp xferOp,
                                PatternRewriter &rewriter) const override {
    if (!xferOp.getIndexVecs().empty()) {
      return failure();
    }

    // Canonicalize to vector.transfer_read.
    rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
        xferOp, xferOp.getType(), xferOp.getBase(), xferOp.getIndices(),
        xferOp.getPermutationMap(), xferOp.getPadding(), xferOp.getMask(),
        xferOp.getInBounds());
    return success();
  };
};

void TransferGatherOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *ctx) {
  results.add<FoldSingleElementIndexVec, FoldContigousGatherToTransferRead>(
      ctx);
}

// clang-format off
#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.cpp.inc" // IWYU pragma: keep
// clang-format on
