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
#include "llvm/ADT/iterator_range.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"

// clang-format off
#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.cpp.inc" // IWYU pragma: keep
// clang-format on

namespace mlir::iree_compiler::IREE::GPU {

//===----------------------------------------------------------------------===//
// MultiMmaOp
//===----------------------------------------------------------------------===//

void MultiMmaOp::build(OpBuilder &builder, OperationState &result, Value lhs,
                       Value rhs, Value acc, ArrayRef<AffineMap> indexingMaps,
                       ArrayRef<GPU::IteratorType> iteratorTypes,
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
          iteratorTypes, [&](IteratorType t) -> mlir::Attribute {
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
                       ArrayRef<GPU::IteratorType> iteratorTypes,
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

LogicalResult MultiMmaOp::verify() {
  ShapedType lhsType = getLhsType();
  ShapedType rhsType = getRhsType();
  ShapedType accType = getAccType();

  SmallVector<AffineMap, 4> indexingMaps = getIndexingMapsArray();

  // Verify that an indexing map was specified for each operand.
  if (indexingMaps.size() != 3)
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
    auto shapedType = llvm::dyn_cast<ShapedType>(getOperand(index).getType());
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

  if (failed(linalg::inferContractionDims(indexingMaps))) {
    return emitOpError("failed to infer contraction dims");
  }

  SmallVector<int64_t> bounds;
  getIterationBounds(bounds);
  // The truncation functionality of llvm::zip is intentional here to ignore
  // the inner dimensions.
  auto verifyOperandShape = [&](ShapedType type, AffineMap map) {
    for (auto [dim, size] : llvm::zip(map.getResults(), type.getShape())) {
      int64_t dimIdx = cast<AffineDimExpr>(dim).getPosition();
      if (size != bounds[dimIdx]) {
        return failure();
      }
    }
    return success();
  };
  if (failed(verifyOperandShape(lhsType, indexingMaps[0]))) {
    return emitOpError("lhs shape does not match iteration bounds");
  }
  if (failed(verifyOperandShape(rhsType, indexingMaps[1]))) {
    return emitOpError("rhs shape does not match iteration bounds");
  }
  if (failed(verifyOperandShape(accType, indexingMaps[2]))) {
    return emitOpError("accumulator shape does not match iteration bounds");
  }

  // Verify supported combining kind.
  auto [lType, rType, aType] = getKind().getABCElementTypes();
  if (lType != lhsType.getElementType()) {
    return emitOpError("lhs element type ")
           << lhsType.getElementType()
           << " does not match expected element type " << lType
           << " for intrinsic";
  }
  if (rType != rhsType.getElementType()) {
    return emitOpError("rhs element type ")
           << rhsType.getElementType()
           << " does not match expected element type " << rType
           << " for intrinsic";
  }
  if (aType != accType.getElementType()) {
    return emitOpError("accumulator element type ")
           << accType.getElementType()
           << " does not match expected element type " << aType
           << " for intrinsic";
  }

  int64_t lhsInnerElementCount = multiplyAcc(getLhsInnerShape());
  int64_t rhsInnerElementCount = multiplyAcc(getRhsInnerShape());
  int64_t accInnerElementCount = multiplyAcc(getAccInnerShape());

  auto [m, n, k] = getKind().getMNKShape();
  if (m * k != lhsInnerElementCount || n * k != rhsInnerElementCount ||
      m * n != accInnerElementCount) {
    auto [lhsThreadType, rhsThreadType, accThreadType] =
        getKind().getABCVectorTypes();
    int64_t lhsThreadElementCount = multiplyAcc(lhsThreadType.getShape());
    int64_t rhsThreadElementCount = multiplyAcc(rhsThreadType.getShape());
    int64_t accThreadElementCount = multiplyAcc(accThreadType.getShape());
    if (lhsInnerElementCount != lhsThreadElementCount ||
        rhsInnerElementCount != rhsThreadElementCount ||
        accInnerElementCount != accThreadElementCount) {
      return emitOpError("operation parallel semantics can't be inferred as "
                         "either thread or subgroup");
    }

    if (getLhsPermutation() || getRhsPermutation() || getAccPermutation()) {
      return emitOpError("permutations require subgroup semantics");
    }
  }

  if (getLhsPermutation()) {
    if (!isPermutationVector(*getLhsPermutation())) {
      return emitOpError("invalid lhs permutation");
    }
  }
  if (getRhsPermutation()) {
    if (!isPermutationVector(*getRhsPermutation())) {
      return emitOpError("invalid rhs permutation");
    }
  }
  if (getAccPermutation()) {
    if (!isPermutationVector(*getAccPermutation())) {
      return emitOpError("invalid accumulator permutation");
    }
  }

  return success();
}

bool MultiMmaOp::hasThreadSemantics() {
  int64_t lhsInnerElementCount = multiplyAcc(getLhsInnerShape());
  int64_t rhsInnerElementCount = multiplyAcc(getRhsInnerShape());
  int64_t accInnerElementCount = multiplyAcc(getAccInnerShape());

  // If it does not have subgroup semantics, then it must have thread semantics.
  auto [m, n, k] = getKind().getMNKShape();
  return m * k != lhsInnerElementCount || n * k != rhsInnerElementCount ||
         m * n != accInnerElementCount;
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
    if (iteratorType == IteratorType::reduction) {
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
// ShuffleTensorOp
//===----------------------------------------------------------------------===//

LogicalResult ShuffleTensorOp::verify() {
  // Get the equivalent tensor type for the alloc to verify against.
  MemRefType allocType = getSharedAllocType();
  Type allocElementType = allocType.getElementType();
  RankedTensorType allocTensorType =
      RankedTensorType::get(allocType.getShape(), allocElementType);

  // Verify source type against inferred type. Slice insertion and extraction
  // use the same verification logic.
  RankedTensorType expectedType = tensor::ExtractSliceOp::inferResultType(
      allocTensorType, getMixedSourceOffsets(), getMixedSourceSizes(),
      getMixedSourceStrides());
  SliceVerificationResult result =
      isRankReducedType(expectedType, getSourceType());
  if (result != SliceVerificationResult::Success) {
    return emitError("Invalid source slice type");
  }

  // Do the same for the resulting tensor type
  expectedType = tensor::ExtractSliceOp::inferResultType(
      allocTensorType, getMixedResultOffsets(), getMixedResultSizes(),
      getMixedResultStrides());
  result = isRankReducedType(expectedType, getType());
  if (result != SliceVerificationResult::Success) {
    return emitError("Invalid result slice type");
  }

  if (allocElementType != getSourceType().getElementType() ||
      allocElementType != getType().getElementType()) {
    return emitError(
        "Element type mismatch between source, allocation, and result");
  }

  // TODO: Verification of the allocation size in the static case.
  return success();
}

} // namespace mlir::iree_compiler::IREE::GPU
