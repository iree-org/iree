// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>

#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Utils/Indexing.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/InterleavedRange.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace mlir::iree_compiler::IREE::VectorExt {

using VectorValue = TypedValue<VectorType>;

// Project the nested layout. This take a mask on the dimensions of the vector
// associated with this layout and projects out those dimensions. This reduces
// the rank of the layout in the process.
VectorLayoutInterface
NestedLayoutAttr::project(ArrayRef<bool> droppedDims) const {
  assert(droppedDims.size() == getRank() &&
         "droppedDims size must match layout rank");

  // Projection for this layout simply means the sizes along the projected
  // are dropped.
  SmallVector<int64_t> subgroupCount;
  SmallVector<int64_t> batchCount;
  SmallVector<int64_t> outerCount;
  SmallVector<int64_t> threadCount;
  SmallVector<int64_t> elementCount;
  SmallVector<int64_t> subgroupStrides;
  SmallVector<int64_t> threadStrides;
  int64_t count = 0;
  // Map to track pre-projection -> post-projection indices. Used to update
  // the dimension orders.
  llvm::DenseMap<int64_t, int64_t> indexToRankReducedIndexMap;
  for (auto [idx, isProjected] : llvm::enumerate(droppedDims)) {
    if (!isProjected) {
      subgroupCount.push_back(getSubgroupTile()[idx]);
      batchCount.push_back(getBatchTile()[idx]);
      outerCount.push_back(getOuterTile()[idx]);
      threadCount.push_back(getThreadTile()[idx]);
      elementCount.push_back(getElementTile()[idx]);
      subgroupStrides.push_back(getSubgroupStrides()[idx]);
      threadStrides.push_back(getThreadStrides()[idx]);
      indexToRankReducedIndexMap[idx] = count++;
    }
  }
  // This layout is invalid for rank-0 vectors.
  assert(count >= 0 && "unimplemented rank-0 vector");

  return NestedLayoutAttr::get(getContext(), subgroupCount, batchCount,
                               outerCount, threadCount, elementCount,
                               subgroupStrides, threadStrides);
}

VectorLayoutInterface NestedLayoutAttr::apply(AffineMap map) const {
  assert(map.getNumDims() == getRank() &&
         "map domain size must match layout rank");

  SmallVector<int64_t> subgroupCount(map.getNumResults(), 1);
  SmallVector<int64_t> batchCount(map.getNumResults(), 1);
  SmallVector<int64_t> outerCount(map.getNumResults(), 1);
  SmallVector<int64_t> threadCount(map.getNumResults(), 1);
  SmallVector<int64_t> elementCount(map.getNumResults(), 1);
  SmallVector<int64_t> subgroupStrides(map.getNumResults(), 0);
  SmallVector<int64_t> threadStrides(map.getNumResults(), 0);

  for (auto [idx, expr] : llvm::enumerate(map.getResults())) {
    if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
      int64_t pos = dim.getPosition();
      subgroupCount[idx] = getSubgroupTile()[pos];
      batchCount[idx] = getBatchTile()[pos];
      outerCount[idx] = getOuterTile()[pos];
      threadCount[idx] = getThreadTile()[pos];
      elementCount[idx] = getElementTile()[pos];
      subgroupStrides[idx] = getSubgroupStrides()[pos];
      threadStrides[idx] = getThreadStrides()[pos];
    }
  }

  return NestedLayoutAttr::get(getContext(), subgroupCount, batchCount,
                               outerCount, threadCount, elementCount,
                               subgroupStrides, threadStrides);
}

VectorLayoutInterface
NestedLayoutAttr::permute(ArrayRef<int64_t> permutation) const {
  SmallVector<int64_t> invPerm = invertPermutationVector(permutation);
  SmallVector<int64_t> subgroupCount =
      applyPermutation(getSubgroupTile(), permutation);
  SmallVector<int64_t> batchCount =
      applyPermutation(getBatchTile(), permutation);
  SmallVector<int64_t> outerCount =
      applyPermutation(getOuterTile(), permutation);
  SmallVector<int64_t> threadCount =
      applyPermutation(getThreadTile(), permutation);
  SmallVector<int64_t> elementCount =
      applyPermutation(getElementTile(), permutation);
  SmallVector<int64_t> subgroupStrides =
      applyPermutation(getSubgroupStrides(), permutation);
  SmallVector<int64_t> threadStrides =
      applyPermutation(getThreadStrides(), permutation);
  return NestedLayoutAttr::get(getContext(), subgroupCount, batchCount,
                               outerCount, threadCount, elementCount,
                               subgroupStrides, threadStrides);
}

/// We distribute to:
/// <BATCH x OUTER x ELEMENT>
SmallVector<int64_t> NestedLayoutAttr::getDistributedShape() const {
  SmallVector<int64_t> shape;
  shape.append(getBatchTile().begin(), getBatchTile().end());
  shape.append(getOuterTile().begin(), getOuterTile().end());
  shape.append(getElementTile().begin(), getElementTile().end());
  return shape;
}

/// Before we distribute, we would like to see this as:
/// <SUBGROUP x BATCH x OUTER x THREAD x ELEMENT>
SmallVector<int64_t> NestedLayoutAttr::getUndistributedPackedShape() const {
  SmallVector<int64_t> shape;
  int64_t rank = getRank();
  shape.reserve(rank * 5);
  shape.append(getSubgroupTile().begin(), getSubgroupTile().end());
  shape.append(getBatchTile().begin(), getBatchTile().end());
  shape.append(getOuterTile().begin(), getOuterTile().end());
  shape.append(getThreadTile().begin(), getThreadTile().end());
  shape.append(getElementTile().begin(), getElementTile().end());
  return shape;
}

SmallVector<int64_t> NestedLayoutAttr::getUndistributedShape() const {
  int64_t rank = getRank();
  SmallVector<int64_t> shape;
  shape.reserve(rank);
  for (int64_t i : llvm::seq<int64_t>(rank)) {
    int64_t expectedDimLen = getSubgroupTile()[i] * getBatchTile()[i] *
                             getOuterTile()[i] * getThreadTile()[i] *
                             getElementTile()[i];
    shape.push_back(expectedDimLen);
  }
  return shape;
}

SmallVector<int64_t>
NestedLayoutAttr::getPackedShapeForUndistributedDim(int64_t dim) const {
  SmallVector<int64_t> shape;
  shape.reserve(5);
  shape.push_back(getSubgroupTile()[dim]);
  shape.push_back(getBatchTile()[dim]);
  shape.push_back(getOuterTile()[dim]);
  shape.push_back(getThreadTile()[dim]);
  shape.push_back(getElementTile()[dim]);
  return shape;
}

SmallVector<int64_t> NestedLayoutAttr::getDistributedUnpackedShape() const {
  SmallVector<int64_t> shape;
  shape.reserve(getRank());
  for (auto [batch, outer, element] :
       llvm::zip(getBatchTile(), getOuterTile(), getElementTile())) {
    shape.push_back(batch * outer * element);
  }
  return shape;
}

// Gets the rank of the undistributed vector for this layout.
int64_t NestedLayoutAttr::getRank() const {
  // The layout requires that all size lists are the same length and match
  // the rank of the undistributed vector, so just return the length of one
  // of the fields.
  return getBatchTile().size();
}

LogicalResult NestedLayoutAttr::isValidLayout(ShapedType shapeTy,
                                              Location loc) const {
  int64_t rank = getRank();
  ArrayRef<int64_t> shape = shapeTy.getShape();
  if (shape.size() != rank) {
    return emitError(loc, "Rank of vector (")
           << shape.size() << ") does not match rank of layout (" << rank
           << ").";
  }
  // Multiply all shapes in the layout.
  for (int i = 0, e = rank; i < e; ++i) {
    int64_t expectedShape = getSubgroupTile()[i] * getBatchTile()[i] *
                            getOuterTile()[i] * getThreadTile()[i] *
                            getElementTile()[i];
    if (ShapedType::isStatic(shape[i]) && expectedShape != shape[i]) {
      std::string layoutStr;
      llvm::raw_string_ostream layoutOs(layoutStr);
      printStripped(layoutOs);
      return emitError(loc, "Vector shape: ")
             << llvm::interleaved_array(shape) << " does not match the layout ("
             << layoutStr + ") at dim " << i
             << ". Dimension expected by layout: " << expectedShape
             << " actual: " << shape[i];
    }
  }
  return success();
}
NestedLayoutAttr NestedLayoutAttr::getChecked(
    llvm::function_ref<InFlightDiagnostic()> emitError, MLIRContext *context,
    ArrayRef<int64_t> subgroupTile, ArrayRef<int64_t> batchTile,
    ArrayRef<int64_t> outerTile, ArrayRef<int64_t> threadTile,
    ArrayRef<int64_t> elementTile, ArrayRef<int64_t> subgroupStrides,
    ArrayRef<int64_t> threadStrides) {
  if (failed(NestedLayoutAttr::verify(emitError, subgroupTile, batchTile,
                                      outerTile, threadTile, elementTile,
                                      subgroupStrides, threadStrides))) {
    return NestedLayoutAttr();
  }

  return NestedLayoutAttr::get(context, subgroupTile, batchTile, outerTile,
                               threadTile, elementTile, subgroupStrides,
                               threadStrides);
}

NestedLayoutAttr NestedLayoutAttr::get(
    MLIRContext *context, ArrayRef<int64_t> subgroupTile,
    ArrayRef<int64_t> batchTile, ArrayRef<int64_t> outerTile,
    ArrayRef<int64_t> threadTile, ArrayRef<int64_t> elementTile,
    ArrayRef<int64_t> subgroupStrides, ArrayRef<int64_t> threadStrides) {

  SmallVector<int64_t> normalizedSubgroupStrides(subgroupStrides);
  SmallVector<int64_t> normalizedThreadStrides(threadStrides);

  // Dimension of size 1 only have one element to distribute, so stride can be
  // anything. We normalize the stride to be 0, to have consistency.

  for (auto [stride, size] :
       llvm::zip_equal(normalizedSubgroupStrides, subgroupTile)) {
    if (size == 1) {
      stride = 0;
    }
  }

  for (auto [stride, size] :
       llvm::zip_equal(normalizedThreadStrides, threadTile)) {
    if (size == 1) {
      stride = 0;
    }
  }

  return Base::get(context, subgroupTile, batchTile, outerTile, threadTile,
                   elementTile, normalizedSubgroupStrides,
                   normalizedThreadStrides);
}

static SmallVector<int64_t> appendDims(ArrayRef<int64_t> tileLens,
                                       ArrayRef<int64_t> appendLens) {
  SmallVector<int64_t> tileLensResult = llvm::to_vector(tileLens);
  tileLensResult.insert(tileLensResult.end(), appendLens.begin(),
                        appendLens.end());
  return tileLensResult;
}

NestedLayoutAttr NestedLayoutAttr::get(MLIRContext *context,
                                       NestedLayoutAttr source,
                                       ArrayRef<int64_t> appendSubGroupLens,
                                       ArrayRef<int64_t> appendBatchLens,
                                       ArrayRef<int64_t> appendOuterLens,
                                       ArrayRef<int64_t> appendThreadLens,
                                       ArrayRef<int64_t> appendElementLens,
                                       ArrayRef<int64_t> appendSubgroupStrides,
                                       ArrayRef<int64_t> appendThreadStrides) {
  SmallVector<int64_t> subgroupTile =
      appendDims(source.getSubgroupTile(), appendSubGroupLens);
  SmallVector<int64_t> batchTile =
      appendDims(source.getBatchTile(), appendBatchLens);
  SmallVector<int64_t> outerTile =
      appendDims(source.getOuterTile(), appendOuterLens);
  SmallVector<int64_t> threadTile =
      appendDims(source.getThreadTile(), appendThreadLens);
  SmallVector<int64_t> elementTile =
      appendDims(source.getElementTile(), appendElementLens);
  SmallVector<int64_t> subgroupStrides =
      appendDims(source.getSubgroupStrides(), appendSubgroupStrides);
  SmallVector<int64_t> threadStrides =
      appendDims(source.getThreadStrides(), appendThreadStrides);
  return NestedLayoutAttr::get(context, subgroupTile, batchTile, outerTile,
                               threadTile, elementTile, subgroupStrides,
                               threadStrides);
}

VectorLayoutInterface
NestedLayoutAttr::getRecombinedLayout(ArrayRef<VectorLayoutInterface> layouts,
                                      ArrayRef<AffineMap> maps,
                                      AffineMap resultMap) {
  constexpr int64_t kInvalid = -1;
  if (llvm::any_of(layouts, [](VectorLayoutInterface layout) {
        return !mlir::isa<NestedLayoutAttr>(layout);
      })) {
    return NestedLayoutAttr();
  }
  MLIRContext *context = resultMap.getContext();

  SmallVector<NestedLayoutAttr> nestedLayouts;
  llvm::transform(layouts, std::back_inserter(nestedLayouts),
                  [&](VectorLayoutInterface layout) {
                    return mlir::cast<NestedLayoutAttr>(layout);
                  });

  int64_t resRank = resultMap.getNumResults();
  SmallVector<int64_t> subgroupTile(resRank, kInvalid);
  SmallVector<int64_t> batchTile(resRank, kInvalid);
  SmallVector<int64_t> outerTile(resRank, kInvalid);
  SmallVector<int64_t> threadTile(resRank, kInvalid);
  SmallVector<int64_t> elementTile(resRank, kInvalid);
  SmallVector<int64_t> subgroupStrides(resRank, kInvalid);
  SmallVector<int64_t> threadStrides(resRank, kInvalid);

  // a helper to perform a valid update when recombining
  // layouts. If there is a conflict, this will return
  // false.
  auto checkedUpdate = [&](int64_t &data, int64_t v) -> bool {
    if (data != kInvalid && data != v) {
      return false;
    }
    data = v;
    return true;
  };

  for (auto [layout, indexingMap] : llvm::zip(nestedLayouts, maps)) {
    for (int64_t resultIdx : llvm::seq<int64_t>(indexingMap.getNumResults())) {
      int64_t iterSpacePos = indexingMap.getDimPosition(resultIdx);
      std::optional<unsigned int> mayBeResultPos =
          resultMap.getResultPosition(getAffineDimExpr(iterSpacePos, context));
      if (!mayBeResultPos.has_value()) {
        continue;
      }
      int64_t resultPos = mayBeResultPos.value();
      if (!checkedUpdate(subgroupTile[resultPos],
                         layout.getSubgroupTile()[resultIdx])) {
        return NestedLayoutAttr();
      }
      if (!checkedUpdate(batchTile[resultPos],
                         layout.getBatchTile()[resultIdx])) {
        return NestedLayoutAttr();
      }
      if (!checkedUpdate(outerTile[resultPos],
                         layout.getOuterTile()[resultIdx])) {
        return NestedLayoutAttr();
      }
      if (!checkedUpdate(threadTile[resultPos],
                         layout.getThreadTile()[resultIdx])) {
        return NestedLayoutAttr();
      }
      if (!checkedUpdate(elementTile[resultPos],
                         layout.getElementTile()[resultIdx])) {
        return NestedLayoutAttr();
      }

      if (!checkedUpdate(subgroupStrides[resultPos],
                         layout.getSubgroupStrides()[resultIdx])) {
        return NestedLayoutAttr();
      }
      if (!checkedUpdate(threadStrides[resultPos],
                         layout.getThreadStrides()[resultIdx])) {
        return NestedLayoutAttr();
      }
    }
  }

  // All the tiles should have valid data
  // after a successful recombination.
  for (const llvm::SmallVector<int64_t> &tile :
       {subgroupTile, batchTile, outerTile, threadTile, subgroupStrides,
        threadStrides}) {
    if (llvm::any_of(tile, [&](int64_t v) { return v == kInvalid; })) {
      return NestedLayoutAttr();
    }
  }

  return NestedLayoutAttr::get(context, subgroupTile, batchTile, outerTile,
                               threadTile, elementTile, subgroupStrides,
                               threadStrides);
}

LogicalResult NestedLayoutAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    ArrayRef<int64_t> subgroupTile, ArrayRef<int64_t> batchTile,
    ArrayRef<int64_t> outerTile, ArrayRef<int64_t> threadTile,
    ArrayRef<int64_t> elementTile, ArrayRef<int64_t> subgroupStrides,
    ArrayRef<int64_t> threadStrides) {

  size_t rank = subgroupTile.size();
  auto checkTile = [&](ArrayRef<int64_t> tileShape) {
    if (tileShape.size() != rank) {
      emitError() << "all fields must have the same rank as the layout";
      return failure();
    }
    return success();
  };

  if (failed(checkTile(subgroupTile)) || failed(checkTile(batchTile)) ||
      failed(checkTile(outerTile)) || failed(checkTile(threadTile)) ||
      failed(checkTile(elementTile)) || failed(checkTile(subgroupStrides)) ||
      failed(checkTile(threadStrides))) {
    return failure();
  }

  return success();
}

/// Given a single flat thread ID, compute the indices of the distributed
/// dimensions (subgroup and thread ids). The only difference between subgroup
/// and thread dimensions is the order in which they are "divided out" of the
/// underlying vector (i.e. vector_shape /= subgroups -> batches -> outers ->
/// threads -> elements). There is no requirement that a subgroup id only
/// spans subgroups.
SmallVector<Value>
NestedLayoutAttr::computeThreadIds(Value threadId, int64_t subgroupSize,
                                   RewriterBase &rewriter) const {
  SmallVector<Value> virtualTids;

  Location loc = threadId.getLoc();

  SmallVector<int64_t> subgroupBasis, threadBasis;
  SmallVector<size_t> subgroupDimToResult, threadDimToResult;

  if (failed(basisFromSizesStrides(getSubgroupTile(), getSubgroupStrides(),
                                   subgroupBasis, subgroupDimToResult)))
    return {};
  if (failed(basisFromSizesStrides(getThreadTile(), getThreadStrides(),
                                   threadBasis, threadDimToResult)))
    return {};

  // Add the subgroup_size to the end of the subgroup delinearization basis.
  subgroupBasis.push_back(subgroupSize);

  auto subgroupSplit = rewriter.create<affine::AffineDelinearizeIndexOp>(
      loc, threadId, subgroupBasis, /*hasOuterBound=*/false);
  auto threadSplit = rewriter.create<affine::AffineDelinearizeIndexOp>(
      loc, threadId, threadBasis, /*hasOuterBound=*/false);

  llvm::transform(subgroupDimToResult, std::back_inserter(virtualTids),
                  [&](size_t idx) { return subgroupSplit.getResult(idx); });
  llvm::transform(threadDimToResult, std::back_inserter(virtualTids),
                  [&](size_t idx) { return threadSplit.getResult(idx); });

  return virtualTids;
}

} // namespace mlir::iree_compiler::IREE::VectorExt

using namespace mlir::iree_compiler::IREE::VectorExt;

#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtEnums.cpp.inc" // IWYU pragma: keep

#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtAttrs.cpp.inc" // IWYU pragma: keep

void IREEVectorExtDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtAttrs.cpp.inc"
      >();
}
