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
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "iree-vector-ext-layout-attr"

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
NestedLayoutAttr::reshape(ArrayRef<int64_t> newShape) const {
  SmallVector<int64_t> subgroupCount;
  SmallVector<int64_t> batchCount;
  SmallVector<int64_t> outerCount;
  SmallVector<int64_t> threadCount;
  SmallVector<int64_t> elementCount;
  SmallVector<int64_t> subgroupStrides;
  SmallVector<int64_t> threadStrides;
  // For each dimension of the layout, we have five levels that we need to
  // process. We will start at the lowest level, i.e., the element level and
  // then work our way upwards through thread, outer and batch level to
  // eventually reach the subgroup level. While reshaping, we need to ensure
  // that we always go upwards in the levels as long as we haven't fully
  // exhausted that dimension of the layout. We can only reset to the element
  // level after we have fully exhausted that dimension and move on to the next
  // dimension of the layout.
  //
  // This implementation supports both cases of reshape, i.e.,
  // expansion (e.g., 64x64 -> 16x4x16x4) as well as
  // contraction (e.g., 16x4x16x4 -> 64x64).
  //
  // We process all shapes and layouts from the inside out, i.e., we start with
  // the innermost dimension and work towards the outermost dimension. To that
  // end, the input shapes & layouts are effectively reversed and we reverse
  // back for the end result.
  //
  // When merging a thread/subgroup level, we need to also ensure that the
  // strides can be merged into a single stride.
  SmallVector<int64_t> remainingLevels = {1, 1, 1, 1, 1};
  SmallVector<int64_t> remainingStrides = {0, 0};
  SmallVector<bool> unitDims = {true, true, true, true, true};
  int64_t currDim = getRank();
  // We initialize to 5 here to immediately trigger the initialization of
  // data-structures inside the do-while-loop.
  int64_t currLevel = 5;
  int64_t minLevel = 0;
  // We need to process all dimensions of the target shape.
  for (int64_t dim : llvm::reverse(newShape)) {
    // levels: element, thread, outer, batch, subgroup
    // strides: thread, subgroup
    SmallVector<int64_t> levels(5, 1);
    SmallVector<int64_t> strides(2, 0);

    // The idea of this loop is to distribute the current dimension of the
    // target shape onto one or multiple dimensions of the layout.
    // In case of an expansion, distributing one dimension might not fully
    // deplete one dimension of the layout and we will keep distributing that
    // dimension of the layout when processing the next dimension of the target
    // shape on the next iteration of the surrounding loop.
    //
    // To illustrate this case, assume an expansion from 64x64 to 16x4x6x4 and
    // the following layout:
    // subgroup_tile = [1, 1]
    // batch_tile = [4, 4]
    // outer_tile = [1, 1]
    // thread_tile = [16, 4]
    // element_tile = [1, 4]
    // We start of with the 4 in the target shape (remember we process in
    // reverse order). In the first iteration, we reset to the element dimension
    // and initialize all data-structures, decrementing currDim from 2 to 1.
    // Then, the 4 is entirely consumed by the element tile at position 1, which
    // also consumes the element level. We then set the current and minimum
    // level to the thread level and start processing the 16 of the target
    // shape. That is partially consumed by the thread level, depleting the
    // thread level, so we move on to the batch level with 16 / 4 = 4 remaining
    // for the current dimension of the target shape. That remaining 4 is
    // consumed by the batch level, depleting the batch level and the dimension
    // of the layout. As a consequence, we move on to the next dimension of the
    // target shape and the next dimension of the layout, where this process
    // repeats.
    //
    // In case of a contraction on the other hand, we might consume more than
    // one dimension of the layout for this dimension of the target shape. This
    // case is handled by the do-while-loop and we will only move to the next
    // dimension of the target shape when we have fully distributed it.
    //
    // To illustrate this case, assume an contraction from 16x4x6x4 to 64x64 and
    // the following layout:
    // subgroup_tile = [1, 1, 1, 1]
    // batch_tile = [4, 1, 4, 1]
    // outer_tile = [1, 1, 1, 1]
    // thread_tile = [4, 4, 4, 1]
    // element_tile = [1, 1, 1, 4]
    // We start with 64 for the current dimension of the target
    // shape. The element level of the innermost dimension of the layout
    // consumes the 4, so we are left with 64/4 = 16. All remaining levels for
    // the current dimension of the layout are unit, so we need to move on to
    // the next dimension of the layout, resetting to the element level. The
    // element level is unit, so we move on to the thread level, which consumes
    // 4, leaving us with 16/4 = 4. We move on to the batch dimension, which
    // consumes the remaining 4, completing this dimension of the target shape.
    // For the next dimension of the target shape, a similar process repeats.
    int64_t dimRemaining = dim;
    do {
      if (dimRemaining == 1) {
        // We have fully consumed this dimension of the target shape and can
        // move on to the next dimension of the target shape.
        break;
      }

      if (currLevel == 5) {
        // We have reached the uppermost level (subgroup) of the layout and need
        // to move on to the next level of the layout.
        if (currDim == 0) {
          // There are no more dimensions of the layout to consume. In all
          // well-formed cases, this should only be the case if we have also
          // fully consumed the target shape. This is checked by the assert
          // outside the loop.
          break;
        }
        --currDim;
        currLevel = 0;
        // This is one of two cases where we reset the minLevel. If we start a
        // new dimension, we start again with the element level, so it's fine to
        // reset it here.
        minLevel = 0;
        remainingLevels = llvm::to_vector(
            llvm::reverse(getPackedShapeForUndistributedDim(currDim)));
        remainingStrides = {getThreadStrides()[currDim],
                            getSubgroupStrides()[currDim]};
        unitDims = llvm::map_to_vector(remainingLevels,
                                       [](int64_t v) { return v == 1; });
      }

      // Handle cases where we have nothing left to consume at this level, i.e.
      // remaining[currLevel] = 1.
      if (unitDims[currLevel]) {
        // Only increment the current level, since this is a unit dim, we don't
        // need to update the minLevel.
        ++currLevel;
        continue;
      }
      if (remainingLevels[currLevel] == 1) {
        // We have fully consumed this level, go up. We cannot go below the
        // minLevel.
        ++currLevel;
        ++minLevel;
        continue;
      }

      if (currLevel < minLevel) {
        // Check that invariant that we're always moving upwards in the levels
        // of the layout (see above for definition). Bail out if the invariant
        // doesn't hold.
        LDBG() << "invariant violated, trying to move below "
                  "minimum level, aborting layout reshaping";
        return VectorLayoutInterface();
      }

      // Distribute remaining[currLevel] into the current level.
      int64_t consume = std::min(dimRemaining, remainingLevels[currLevel]);
      // Check if the remaining strides can be consumed.
      if (currLevel == 1) {
        // thread level.
        if (strides[0] != 0 &&
            strides[0] * levels[currLevel] != remainingStrides[0]) {
          LDBG() << "cannot consume stride, aborting layout reshaping";
          // Cannot consume the stride, bail out.
          return VectorLayoutInterface();
        }
        strides[0] = (strides[0] == 0 ? remainingStrides[0] : strides[0]);
        remainingStrides[0] *= consume;
      }
      if (currLevel == 4) {
        // subgroup level.
        if (strides[1] != 0 &&
            strides[1] * levels[currLevel] != remainingStrides[1]) {
          // Cannot consume the stride, bail out.
          return VectorLayoutInterface();
        }
        strides[1] = (strides[1] == 0 ? remainingStrides[1] : strides[1]);
        remainingStrides[1] *= consume;
      }

      levels[currLevel] *= consume;
      dimRemaining /= consume;
      remainingLevels[currLevel] /= consume;
    } while (true);

    // This assert ensure that if we break the above do-while-loop because we
    // have completely depleted the layout that we have also fully distributed
    // the target shape.
    assert(dimRemaining == 1 && "cannot reshape, remaining dim not consumed");
    elementCount.push_back(levels[0]);
    threadCount.push_back(levels[1]);
    outerCount.push_back(levels[2]);
    batchCount.push_back(levels[3]);
    subgroupCount.push_back(levels[4]);
    threadStrides.push_back(strides[0]);
    subgroupStrides.push_back(strides[1]);

    if (llvm::all_of(remainingLevels, [](int64_t v) { return v == 1; })) {
      // This is the second case where we reset the minimum level: We have fully
      // consumed the current dimension of the target shape and at the same time
      // depleted the current dimension of the layout (all remaining levels are
      // unit). We will move on to the next dimension of the target shape and at
      // the same time also move on to the next dimension of the layout, which
      // allows us to reset the minimum level back to element level.
      minLevel = 0;
      currLevel = 5;
    }
  }
  // Reverse the counts and strides since we processed them in reverse order.
  std::reverse(subgroupCount.begin(), subgroupCount.end());
  std::reverse(batchCount.begin(), batchCount.end());
  std::reverse(outerCount.begin(), outerCount.end());
  std::reverse(threadCount.begin(), threadCount.end());
  std::reverse(elementCount.begin(), elementCount.end());
  std::reverse(threadStrides.begin(), threadStrides.end());
  std::reverse(subgroupStrides.begin(), subgroupStrides.end());

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
  if (!llvm::all_of(layouts, llvm::IsaPred<NestedLayoutAttr>)) {
    return NestedLayoutAttr();
  }
  MLIRContext *context = resultMap.getContext();

  SmallVector<NestedLayoutAttr> nestedLayouts;
  llvm::transform(layouts, std::back_inserter(nestedLayouts),
                  [&](VectorLayoutInterface layout) {
                    return cast<NestedLayoutAttr>(layout);
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
                                   subgroupBasis, subgroupDimToResult))) {
    return {};
  }
  if (failed(basisFromSizesStrides(getThreadTile(), getThreadStrides(),
                                   threadBasis, threadDimToResult))) {
    return {};
  }

  // Add the subgroup_size to the end of the subgroup delinearization basis.
  subgroupBasis.push_back(subgroupSize);

  auto subgroupSplit = affine::AffineDelinearizeIndexOp::create(
      rewriter, loc, threadId, subgroupBasis, /*hasOuterBound=*/false);
  auto threadSplit = affine::AffineDelinearizeIndexOp::create(
      rewriter, loc, threadId, threadBasis, /*hasOuterBound=*/false);

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
