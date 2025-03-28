// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Utils/ParsingUtils.h"
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

static int64_t getVectorRank(Type type) {
  return llvm::isa<VectorType>(type) ? llvm::cast<VectorType>(type).getRank()
                                     : 0;
}

struct IndexVecFoldResult {
  Value indexVec;
  AffineMap indexMap;
  int64_t indexedVal;
  bool changed;
};

static Value foldTransferGatherIndexVecs(
    TransferGatherOp gatherOp,
    function_ref<IndexVecFoldResult(Value, AffineMap, int64_t)>
        indexVecFolder) {
  bool changed = false;
  SmallVector<Value> newIndexVecs;
  SmallVector<AffineMap> newIndexedMaps;
  SmallVector<int64_t> indexed(gatherOp.getIndexed());
  int64_t currIndexVec = 0;
  for (auto i : llvm::seq<int64_t>(gatherOp.getIndices().size())) {
    if (indexed[i] != -2) {
      continue;
    }
    Value operand = gatherOp.getIndexVecs()[currIndexVec];
    AffineMap map = gatherOp.getIndexedMapsArray()[currIndexVec];
    ++currIndexVec;

    auto [indexVec, indexMap, indexedVal, vecChanged] =
        indexVecFolder(operand, map, i);
    changed |= vecChanged;
    indexed[i] = indexedVal;
    if (indexedVal == -2) {
      newIndexVecs.push_back(indexVec);
      newIndexedMaps.push_back(indexMap);
    }
  }

  if (!changed) {
    return Value();
  }

  OpBuilder b(gatherOp);

  SmallVector<Value> operands;
  SmallVector<int32_t> operandSegmentSizes;

  // Source.
  operands.push_back(gatherOp.getSource());
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
  gatherOp.setIndexed(indexed);
  gatherOp.getProperties().setOperandSegmentSizes(operandSegmentSizes);

  return gatherOp.getResult();
}

static Value foldTransferGatherFromBroadcast(TransferGatherOp gatherOp) {
  return foldTransferGatherIndexVecs(
      gatherOp,
      [](Value operand, AffineMap map, int64_t) -> IndexVecFoldResult {
        auto broadcast = operand.getDefiningOp<vector::BroadcastOp>();
        if (!broadcast) {
          return {operand, map, -2, false};
        }

        int64_t sourceRank = getVectorRank(broadcast.getSourceType());
        int64_t operandRank = getVectorRank(broadcast.getResultVectorType());
        AffineMap newMap =
            map.getSliceMap(operandRank - sourceRank, sourceRank);
        return {broadcast.getSource(), newMap, -2, true};
      });
}

static Value foldTransferGatherFromTranspose(TransferGatherOp gatherOp) {
  return foldTransferGatherIndexVecs(
      gatherOp,
      [](Value operand, AffineMap map, int64_t) -> IndexVecFoldResult {
        auto transpose = operand.getDefiningOp<vector::TransposeOp>();
        if (!transpose) {
          return {operand, map, -2, false};
        }

        AffineMap newMap =
            AffineMap::getPermutationMap(
                invertPermutationVector(transpose.getPermutation()),
                transpose.getContext())
                .compose(map);
        return {transpose.getVector(), newMap, -2, true};
      });
}

static Value foldTransferGatherFromStep(TransferGatherOp gatherOp) {
  return foldTransferGatherIndexVecs(
      gatherOp,
      [](Value operand, AffineMap map, int64_t) -> IndexVecFoldResult {
        auto step = operand.getDefiningOp<vector::StepOp>();
        if (!step) {
          return {operand, map, -2, false};
        }

        assert(map.getNumResults() == 1);
        int64_t resultDim = cast<AffineDimExpr>(map.getResult(0)).getPosition();
        return {Value(), AffineMap(), resultDim, true};
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

struct FoldSingleElementIndexVec : public OpRewritePattern<TransferGatherOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TransferGatherOp xferOp,
                                PatternRewriter &rewriter) const override {

    auto indexVecFolder = [&](Value indexVec, AffineMap map,
                              int64_t index) -> IndexVecFoldResult {
      auto vectorTy = cast<VectorType>(indexVec.getType());
      if (vectorTy.getNumElements() != 1) {
        return {indexVec, map, -2, false};
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

      return {Value(), AffineMap(), -1, true};
    };

    Value newVal = foldTransferGatherIndexVecs(xferOp, indexVecFolder);

    if (!newVal) {
      return failure();
    }

    return success();
  }
};

struct FoldContigousGatherToTransferRead
    : public OpRewritePattern<TransferGatherOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TransferGatherOp xferOp,
                                PatternRewriter &rewriter) const override {
    if (!xferOp.getIndexVecs().empty()) {
      return failure();
    }

    // Canonicalize to vector.transfer_read.
    rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
        xferOp, xferOp.getVectorType(), xferOp.getSource(), xferOp.getIndices(),
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

Speculation::Speculatability TransferGatherOp::getSpeculatability() {
  if (isa<RankedTensorType>(getSource().getType())) {
    return Speculation::Speculatable;
  }
  return Speculation::NotSpeculatable;
}

void TransferGatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (llvm::isa<MemRefType>(getSource().getType())) {
    effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable(),
                         SideEffects::DefaultResource::get());
  }
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
  p << " " << getSource() << "[" << getIndices() << "]";
  printIndexVecs(p, *this, getIndexVecs(), getIndexVecs().getTypes(),
                 getIndexedAttr());
  if (getMask())
    p << ", " << getMask();
  printTransferAttrs(p, *this, {"indexed"});
  p << " : ";
  p << getShapedType() << ", ";
  llvm::interleaveComma(getIndexVecs().getType(), p);
  p << ", " << getVectorType();
}

LogicalResult TransferGatherOp::verify() { return success(); }

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
  DenseI64ArrayAttr indexed;
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

// clang-format off
#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.cpp.inc" // IWYU pragma: keep
// clang-format on
