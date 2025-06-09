// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include <functional>
#include <numeric>

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"

// clang-format off
#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.cpp.inc" // IWYU pragma: keep
// clang-format on

namespace mlir::iree_compiler::IREE::GPU {
//===----------------------------------------------------------------------===//
// BarrierRegionOp
//===----------------------------------------------------------------------===//

// Build a BarrierRegionOp with an empty.
void BarrierRegionOp::build(OpBuilder &b, OperationState &result,
                            TypeRange resultTypes, ValueRange inputs) {
  result.addOperands(inputs);
  (void)result.addRegion();
  result.addTypes(resultTypes);
  SmallVector<Location> blockArgLocs(inputs.size(), result.location);

  Region *region = result.regions[0].get();

  // `builder.createBlock` changes the insertion point within the block. Create
  // a guard to reset the insertion point of the builder after it is destroyed.
  OpBuilder::InsertionGuard guard(b);
  b.createBlock(region, region->end(), inputs.getTypes(), blockArgLocs);
}

LogicalResult BarrierRegionOp::verify() { return success(); }

LogicalResult BarrierRegionOp::verifyRegions() {
  auto &region = getRegion();
  Block &block = region.front();
  if (block.getNumArguments() != getNumOperands()) {
    return emitError(
        "expected the block argument count to match operand count");
  }

  if (!llvm::all_of_zip(block.getArgumentTypes(), getOperandTypes(),
                        [](Type a, Type b) { return a == b; })) {
    return emitError("expected block argument types to match operand types");
  }

  // Ensure that the region yields an element of the right type.
  auto yieldOp = llvm::cast<GPU::YieldOp>(block.getTerminator());
  if (yieldOp->getNumOperands() != getNumResults()) {
    return emitOpError(
        "expected body to yield same number of values as results");
  }

  if (!llvm::all_of_zip(yieldOp->getOperandTypes(), getResultTypes(),
                        [](Type a, Type b) { return a == b; })) {
    return emitError("expected yielded value types to match result types");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MultiMmaOp
//===----------------------------------------------------------------------===//

void MultiMmaOp::build(OpBuilder &builder, OperationState &result, Value lhs,
                       Value rhs, Value acc, ArrayRef<AffineMap> indexingMaps,
                       ArrayRef<utils::IteratorType> iteratorTypes,
                       MmaInterfaceAttr kind,
                       std::optional<SmallVector<int64_t>> lhsPermutation,
                       std::optional<SmallVector<int64_t>> rhsPermutation,
                       std::optional<SmallVector<int64_t>> accPermutation) {
  result.addOperands({lhs, rhs, acc});
  result.addTypes(acc.getType());
  result.addAttribute(getIndexingMapsAttrName(result.name),
                      builder.getAffineMapArrayAttr(indexingMaps));
  result.addAttribute(
      getIteratorTypesAttrName(result.name),
      builder.getArrayAttr(llvm::to_vector(llvm::map_range(
          iteratorTypes, [&](utils::IteratorType t) -> mlir::Attribute {
            return IteratorTypeAttr::get(builder.getContext(), t);
          }))));
  result.addAttribute(getKindAttrName(result.name), kind);
  if (lhsPermutation) {
    result.addAttribute(getLhsPermutationAttrName(result.name),
                        builder.getDenseI64ArrayAttr(*lhsPermutation));
  }
  if (rhsPermutation) {
    result.addAttribute(getRhsPermutationAttrName(result.name),
                        builder.getDenseI64ArrayAttr(*rhsPermutation));
  }
  if (accPermutation) {
    result.addAttribute(getAccPermutationAttrName(result.name),
                        builder.getDenseI64ArrayAttr(*accPermutation));
  }
}

void MultiMmaOp::build(OpBuilder &builder, OperationState &result, Value lhs,
                       Value rhs, Value acc,
                       ArrayRef<ArrayRef<AffineExpr>> indexingExprs,
                       ArrayRef<utils::IteratorType> iteratorTypes,
                       MmaInterfaceAttr kind,
                       std::optional<SmallVector<int64_t>> lhsPermutation,
                       std::optional<SmallVector<int64_t>> rhsPermutation,
                       std::optional<SmallVector<int64_t>> accPermutation) {
  build(builder, result, lhs, rhs, acc,
        AffineMap::inferFromExprList(indexingExprs, builder.getContext()),
        iteratorTypes, kind, lhsPermutation, rhsPermutation, accPermutation);
}

void MultiMmaOp::build(OpBuilder &builder, OperationState &result, Value lhs,
                       Value rhs, Value acc, ArrayAttr indexingMaps,
                       ArrayAttr iteratorTypes, MmaInterfaceAttr kind,
                       std::optional<DenseI64ArrayAttr> lhsPermutation,
                       std::optional<DenseI64ArrayAttr> rhsPermutation,
                       std::optional<DenseI64ArrayAttr> accPermutation) {
  result.addOperands({lhs, rhs, acc});
  result.addTypes(acc.getType());
  result.addAttribute(getIndexingMapsAttrName(result.name), indexingMaps);
  result.addAttribute(getIteratorTypesAttrName(result.name), iteratorTypes);
  result.addAttribute(getKindAttrName(result.name), kind);
  if (lhsPermutation) {
    result.addAttribute(getLhsPermutationAttrName(result.name),
                        *lhsPermutation);
  }
  if (rhsPermutation) {
    result.addAttribute(getRhsPermutationAttrName(result.name),
                        *rhsPermutation);
  }
  if (accPermutation) {
    result.addAttribute(getAccPermutationAttrName(result.name),
                        *accPermutation);
  }
}

static int64_t multiplyAcc(ArrayRef<int64_t> shape) {
  return std::accumulate(shape.begin(), shape.end(), 1,
                         std::multiplies<int64_t>());
}

static bool countsMatchTileTypes(ArrayRef<int64_t> innerElemCounts,
                                 ArrayRef<VectorType> tileTypes) {
  return llvm::all_of_zip(
      innerElemCounts, tileTypes,
      [](int64_t ec, VectorType tt) { return ec == tt.getNumElements(); });
}

static SmallVector<int64_t> getInnerElemCounts(MultiMmaOp mmaOp) {
  SmallVector<int64_t> result;
  result.reserve(mmaOp.getNumOperands());
  for (auto [opType, map] : llvm::zip_equal(
           mmaOp.getOperandTypes(),
           mmaOp.getIndexingMapsAttr().getAsValueRange<AffineMapAttr>())) {
    ArrayRef<int64_t> shape = cast<ShapedType>(opType).getShape();
    result.push_back(multiplyAcc(shape.drop_front(map.getNumResults())));
  }
  return result;
}

LogicalResult MultiMmaOp::verify() {
  int64_t expectedNumIns = getKind().getExpectedNumInputs();
  if (expectedNumIns != 2) {
    return emitOpError("we're mid-refactoring, input can only be LHS + RHS");
  }

  SmallVector<ShapedType> opTypes = llvm::map_to_vector(
      getOperandTypes(), [](auto t) { return llvm::cast<ShapedType>(t); });
  SmallVector<AffineMap, 4> indexingMaps = getIndexingMapsArray();

  // Verify that an indexing map was specified for each operand.
  if (indexingMaps.size() != expectedNumIns + 1)
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
    if (map.getNumDims() != numIterators)
      return emitOpError("expected indexing map ")
             << index << " to have " << numIterators << " number of inputs";
    if (map.getNumResults() >= rank)
      return emitOpError("expected indexing map ")
             << index << " to have fewer than " << rank << " number of outputs";
    if (!map.isProjectedPermutation())
      return emitOpError("expected indexing map ")
             << index << " to be a projected permutation of its inputs";

    for (auto size :
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
  // The truncation functionality of llvm::zip is intentional here to ignore
  // the inner dimensions.
  for (auto [type, map] : llvm::zip_equal(opTypes, indexingMaps)) {
    for (auto [dim, size] : llvm::zip(map.getResults(), type.getShape())) {
      int64_t dimIdx = cast<AffineDimExpr>(dim).getPosition();
      if (size != bounds[dimIdx]) {
        return emitOpError("shape does not match iteration bounds");
      }
    }
    return success();
  };

  SmallVector<VectorType> preThreadTypes;
  getKind().getUndistributedTileTypes(preThreadTypes);
  SmallVector<VectorType> threadTypes;
  getKind().getDistributedTileTypes(threadTypes);

  SmallVector<int64_t> innerElemCounts = getInnerElemCounts(*this);
  for (auto [opNum, opType, tileType] :
       llvm::enumerate(opTypes, preThreadTypes)) {
    if (opType.getElementType() != tileType.getElementType()) {
      return emitOpError("operand " + Twine(opNum) + " element type ")
             << opType.getElementType() << " does not match expected tile type "
             << tileType << " for operator";
    }
  }

  bool hasSubgroupSemantics =
      countsMatchTileTypes(innerElemCounts, preThreadTypes);
  bool hasThreadSemantics = countsMatchTileTypes(innerElemCounts, threadTypes);
  if (!hasSubgroupSemantics && !hasThreadSemantics) {
    return emitOpError("operation parallel semantics can't be inferred as "
                       "either thread or subgroup");
  }
  if (hasThreadSemantics) {
    if (getLhsPermutation() || getRhsPermutation() || getAccPermutation()) {
      return emitOpError("permutations require subgroup semantics");
    }
  }

  if (getLhsPermutation() && !isPermutationVector(*getLhsPermutation())) {
    return emitOpError("invalid lhs permutation");
  }
  if (getRhsPermutation() && !isPermutationVector(*getRhsPermutation())) {
    return emitOpError("invalid rhs permutation");
  }
  if (getAccPermutation() && !isPermutationVector(*getAccPermutation())) {
    return emitOpError("invalid accumulator permutation");
  }

  return success();
}

bool MultiMmaOp::hasThreadSemantics() {
  SmallVector<int64_t> innerElemCounts = getInnerElemCounts(*this);
  SmallVector<VectorType> preThreadTiles;
  getKind().getUndistributedTileTypes(preThreadTiles);
  return !countsMatchTileTypes(innerElemCounts, preThreadTiles);
}

static int64_t getResultIndex(AffineMap map, AffineExpr targetExpr) {
  for (int64_t i = 0, e = map.getNumResults(); i < e; ++i)
    if (targetExpr == map.getResult(i))
      return i;
  return -1;
}

void MultiMmaOp::getIterationBounds(SmallVectorImpl<int64_t> &iterationBounds) {
  auto lhsShape = getLhsType().getShape();
  auto resType = getResultType();
  SmallVector<AffineMap, 4> indexingMaps(getIndexingMapsArray());
  SmallVector<int64_t, 2> iterationShape;
  for (const auto &it : llvm::enumerate(getIteratorTypes())) {
    // Search lhs/rhs map results for 'targetExpr'.
    auto targetExpr = getAffineDimExpr(it.index(), getContext());
    auto iteratorType = llvm::cast<IteratorTypeAttr>(it.value()).getValue();
    // TODO: search all input indexing maps
    if (iteratorType == utils::IteratorType::reduction) {
      // Get reduction dim size from lhs shape (same size in rhsShape).
      int64_t lhsDimIndex = getResultIndex(indexingMaps[0], targetExpr);
      assert(lhsDimIndex >= 0);
      iterationBounds.push_back(lhsShape[lhsDimIndex]);
      continue;
    }
    // Get parallel dimension size from result shape.
    int64_t resDimIndex = getResultIndex(indexingMaps[2], targetExpr);
    assert(resDimIndex >= 0);
    iterationBounds.push_back(resType.getShape()[resDimIndex]);
  }
}

std::optional<SmallVector<int64_t, 4>> MultiMmaOp::getShapeForUnroll() {
  SmallVector<int64_t, 4> shape;
  getIterationBounds(shape);
  return shape;
}

//===----------------------------------------------------------------------===//
// ValueBarrierOp
//===----------------------------------------------------------------------===//

void ValueBarrierOp::build(OpBuilder &builder, OperationState &result,
                           ValueRange input) {
  result.addOperands(input);
  result.addTypes(llvm::map_range(input, [](Value v) { return v.getType(); }));
}

LogicalResult ValueBarrierOp::verify() {
  if (getNumOperands() == 0) {
    return emitOpError("Atleast one input required");
  }

  // Make sure we either have all tensors or all vectors.
  if (hasTensorSemantics()) {
    bool allTensor =
        llvm::all_of(getInputTypes(), llvm::IsaPred<RankedTensorType>);
    if (!allTensor) {
      return emitOpError(
          "All inputs should be either of tensor or vector type");
    }
    return success();
  }

  bool allVector = llvm::all_of(getInputTypes(), llvm::IsaPred<VectorType>);
  if (!allVector) {
    return emitOpError("All inputs should be either of tensor or vector type");
  }

  return success();
}

// AMD Specific Operations

//===----------------------------------------------------------------------===//
// BufferResourceCastOp
//===----------------------------------------------------------------------===//

static RankedTensorType getMaximumStaticType(tensor::CastOp castOp) {
  auto inputType = dyn_cast<RankedTensorType>(castOp.getSource().getType());
  auto resultType = dyn_cast<RankedTensorType>(castOp.getType());
  if (!inputType || !resultType) {
    return RankedTensorType();
  }

  assert(inputType.getRank() == resultType.getRank() &&
         "Rank must match for ranked -> ranked cast");

  SmallVector<int64_t> join;
  join.reserve(inputType.getRank());
  for (int64_t i = 0; i < inputType.getRank(); ++i) {
    if (inputType.isDynamicDim(i)) {
      join.push_back(resultType.getDimSize(i));
      continue;
    }
    if (resultType.isDynamicDim(i)) {
      join.push_back(inputType.getDimSize(i));
      continue;
    }

    // Cast verifier requires that static sizes match.
    join.push_back(inputType.getDimSize(i));
  }
  return RankedTensorType::get(join, inputType.getElementType());
}

struct FoldBufferCastOfTensorCast final
    : OpRewritePattern<BufferResourceCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BufferResourceCastOp castOp,
                                PatternRewriter &rewriter) const override {
    // Check whether the cast increases the amount of available static info.
    auto tensorCast = castOp.getInput().getDefiningOp<tensor::CastOp>();
    if (!tensorCast) {
      return failure();
    }

    RankedTensorType maxStaticType = getMaximumStaticType(tensorCast);
    if (!maxStaticType || maxStaticType == castOp.getInput().getType()) {
      return failure();
    }

    Value newSource = tensorCast.getSource();
    if (newSource.getType() != maxStaticType) {
      // Cast to the type with maximum static information if the input and
      // result types contain different static info.
      newSource = rewriter.create<tensor::CastOp>(castOp.getLoc(),
                                                  maxStaticType, newSource);
    }
    auto newBufferCast = rewriter.create<IREE::GPU::BufferResourceCastOp>(
        castOp.getLoc(), maxStaticType, newSource,
        castOp.getCacheSwizzleStride());
    newBufferCast->setDiscardableAttrs(castOp->getDiscardableAttrDictionary());

    // Cast back to the original result type.
    rewriter.replaceOpWithNewOp<tensor::CastOp>(
        castOp, castOp.getResult().getType(), newBufferCast);
    return success();
  };
};

void BufferResourceCastOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *ctx) {
  results.add<FoldBufferCastOfTensorCast>(ctx);
}

} // namespace mlir::iree_compiler::IREE::GPU
