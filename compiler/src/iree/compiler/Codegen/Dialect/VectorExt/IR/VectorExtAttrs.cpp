// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>

#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace mlir::iree_compiler::IREE::VectorExt {

using VectorValue = TypedValue<VectorType>;

bool PerDimLayoutAttr::contains(const LayoutDimension &dim) {
  for (LayoutDimensionAttr label : getLabels()) {
    if (label.getValue() == dim)
      return true;
  }
  return false;
}

std::optional<int64_t> PerDimLayoutAttr::getShape(const LayoutDimension &dim) {
  for (auto value : llvm::zip(getLabels(), getShapes())) {
    if (dim == std::get<0>(value).getValue())
      return std::get<1>(value);
  }
  return std::nullopt;
}

std::optional<int64_t> LayoutAttr::getShape(const LayoutDimension &dim) const {
  for (PerDimLayoutAttr layout : getLayouts()) {
    std::optional<int64_t> maybeShape = layout.getShape(dim);
    if (maybeShape)
      return maybeShape.value();
  }
  return std::nullopt;
}

// Get the SIMT Vector shape in the order specified by dims. If no dims are
// specified, then return an empty vector.
LogicalResult LayoutAttr::isValidLayout(ShapedType shapeTy,
                                        Location loc) const {
  ArrayRef<int64_t> shape = shapeTy.getShape();
  if (shape.size() != getRank()) {
    return emitError(loc, "Rank of vector (")
           << shape.size() << ") does not match rank of layout (" << getRank()
           << ").";
  }
  for (auto [idx, layout] : llvm::enumerate(getLayouts())) {
    ArrayRef<int64_t> layoutShape = layout.getShapes();
    int64_t expectedShape =
        std::reduce(layoutShape.begin(), layoutShape.end(),
                    static_cast<int64_t>(1), std::multiplies<int64_t>());
    if (expectedShape != shape[idx]) {
      std::string shapeStr;
      llvm::raw_string_ostream shapeOs(shapeStr);
      llvm::interleaveComma(shape, shapeOs);
      std::string layoutStr;
      llvm::raw_string_ostream layoutOs(layoutStr);
      printStripped(layoutOs);
      return emitError(loc, "Vector shape: [")
             << shapeStr << "] does not match the layout (" << layoutStr
             << ") at dim " << idx
             << ". Dimension expected by layout: " << expectedShape
             << " actual: " << shape[idx];
    }
  }
  return success();
}

// Project out the layout for the specified dimensions
// resulting in the layout for a lower dimensional vector.
VectorLayoutInterface LayoutAttr::project(ArrayRef<bool> droppedDims) const {
  assert(droppedDims.size() == getRank() &&
         "droppedDims size must match layout size");

  ArrayRef<PerDimLayoutAttr> layouts = getLayouts();
  SmallVector<PerDimLayoutAttr> newLayouts;
  for (auto pair : llvm::zip(droppedDims, layouts)) {
    if (!std::get<0>(pair))
      newLayouts.push_back(std::get<1>(pair));
  }
  return LayoutAttr::get(getContext(), newLayouts);
}

// Permute the layout according to the provided permutation
// vector. The dimensionality of the layout remains the same.
VectorLayoutInterface LayoutAttr::permute(ArrayRef<int64_t> permutation) const {
  assert(permutation.size() == getRank() &&
         "permutation size must match layout rank");

  ArrayRef<PerDimLayoutAttr> layouts = getLayouts();
  SmallVector<PerDimLayoutAttr> newLayouts;
  for (unsigned index : permutation) {
    assert(index >= 0 && index < getRank());
    newLayouts.push_back(layouts[index]);
  }
  return LayoutAttr::get(getContext(), newLayouts);
}

// This function returns the distributed shape of the SIMT
// vector and evaluates it in the following order:
// BATCHX, BATCHY, VECTORY, VECTORX
// The vector dimensions are combined into a single SIMT
// vector dimension.
SmallVector<int64_t> LayoutAttr::getDistributedShape() const {
  SmallVector<LayoutDimension> labels{
      LayoutDimension::BATCHX, LayoutDimension::BATCHY,
      LayoutDimension::VECTORY, LayoutDimension::VECTORX};
  SmallVector<int64_t> simtVectorShape;
  std::optional<int64_t> vectorShape;
  for (LayoutDimension dim : labels) {
    ArrayRef<PerDimLayoutAttr> layouts = getLayouts();
    for (PerDimLayoutAttr layout : layouts) {
      if (!layout.contains(dim))
        continue;
      int64_t shape = layout.getShape(dim).value();
      if (isVectorDimension(dim)) {
        vectorShape = shape * vectorShape.value_or(1);
        continue;
      }
      simtVectorShape.push_back(shape);
    }
  }
  if (vectorShape)
    simtVectorShape.push_back(vectorShape.value());
  return simtVectorShape;
}

PerDimLayoutAttr LayoutAttr::getDimLayout(int64_t dim) const {
  assert(dim >= 0 && dim < getRank());
  return getLayouts()[dim];
}

std::optional<int64_t> LayoutAttr::getBatchDim(int64_t dim) {
  assert(dim < getRank());
  PerDimLayoutAttr layout = getDimLayout(dim);
  for (auto [name, shape] :
       llvm::zip_equal(layout.getLabels(), layout.getShapes())) {
    if (isBatchDimension(name.getValue()))
      return shape;
  }
  return std::nullopt;
}

std::optional<int64_t> LayoutAttr::getLaneDim(int64_t dim) {
  assert(dim < getRank());
  PerDimLayoutAttr layout = getDimLayout(dim);
  for (auto [name, shape] :
       llvm::zip_equal(layout.getLabels(), layout.getShapes())) {
    if (isLaneDimension(name.getValue()))
      return shape;
  }
  return std::nullopt;
}

std::optional<LayoutDimension> LayoutAttr::getLane(int64_t dim) {
  assert(dim < getRank());
  PerDimLayoutAttr layout = getDimLayout(dim);
  for (auto [name, shape] :
       llvm::zip_equal(layout.getLabels(), layout.getShapes())) {
    if (isLaneDimension(name.getValue()))
      return name.getValue();
  }
  return std::nullopt;
}

int64_t LayoutAttr::getRank() const { return getLayouts().size(); }

std::tuple<int64_t, int64_t, int64_t> LayoutAttr::getLaneGrid() {
  int64_t laneX = 1;
  int64_t laneY = 1;
  int64_t laneZ = 1;
  for (PerDimLayoutAttr dimLayout : getLayouts()) {
    // Note that valid layouts only include at most one instance of each
    // dimension type, so this is simply doing assignment on the first instance
    // of each lane index, not an accumulative product.
    auto maybeXShape = dimLayout.getShape(LayoutDimension::LANEX);
    laneX *= maybeXShape.value_or(1);
    auto maybeYShape = dimLayout.getShape(LayoutDimension::LANEY);
    laneY *= maybeYShape.value_or(1);
    auto maybeZShape = dimLayout.getShape(LayoutDimension::LANEZ);
    laneZ *= maybeZShape.value_or(1);
  }
  return std::make_tuple(laneX, laneY, laneZ);
}

uint64_t LayoutAttr::getShuffleOffset(int64_t reductionDim) {
  uint64_t offset = 0;
  std::optional<LayoutDimension> laneDim = getLane(reductionDim);
  if (!laneDim)
    return offset;
  switch (laneDim.value()) {
  case LayoutDimension::LANEX:
    offset = 1;
    break;
  case LayoutDimension::LANEY:
    offset = getShape(LayoutDimension::LANEX).value_or(0);
    break;
  case LayoutDimension::LANEZ:
    offset = getShape(LayoutDimension::LANEX).value_or(0) *
             getShape(LayoutDimension::LANEY).value_or(0);
    break;
  default:
    assert(false && "Invalid dimension! Expected lane dimension");
    break;
  }
  return offset;
}

bool LayoutAttr::hasLaneConflictWith(const LayoutAttr &other) {
  SmallVector<LayoutDimension> laneDims{
      LayoutDimension::LANEX, LayoutDimension::LANEY, LayoutDimension::LANEZ};
  for (LayoutDimension dim : laneDims) {
    std::optional<int64_t> shape = getShape(dim);
    std::optional<int64_t> otherShape = other.getShape(dim);
    if ((shape && !otherShape) || (!shape && otherShape))
      return true;
    if (shape && otherShape) {
      if (shape.value() != otherShape.value())
        return true;
    }
  }
  return false;
}

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
      subgroupCount.push_back(getSubgroupsPerWorkgroup()[idx]);
      batchCount.push_back(getBatchesPerSubgroup()[idx]);
      outerCount.push_back(getOutersPerBatch()[idx]);
      threadCount.push_back(getThreadsPerOuter()[idx]);
      elementCount.push_back(getElementsPerThread()[idx]);
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

VectorLayoutInterface
NestedLayoutAttr::permute(ArrayRef<int64_t> permutation) const {
  SmallVector<int64_t> invPerm = invertPermutationVector(permutation);
  SmallVector<int64_t> subgroupCount =
      applyPermutation(getSubgroupsPerWorkgroup(), permutation);
  SmallVector<int64_t> batchCount =
      applyPermutation(getBatchesPerSubgroup(), permutation);
  SmallVector<int64_t> outerCount =
      applyPermutation(getOutersPerBatch(), permutation);
  SmallVector<int64_t> threadCount =
      applyPermutation(getThreadsPerOuter(), permutation);
  SmallVector<int64_t> elementCount =
      applyPermutation(getElementsPerThread(), permutation);
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
  shape.append(getBatchesPerSubgroup().begin(), getBatchesPerSubgroup().end());
  shape.append(getOutersPerBatch().begin(), getOutersPerBatch().end());
  shape.append(getElementsPerThread().begin(), getElementsPerThread().end());
  return shape;
}

// Gets the rank of the undistributed vector for this layout.
int64_t NestedLayoutAttr::getRank() const {
  // The layout requires that all size lists are the same length and match
  // the rank of the undistributed vector, so just return the length of one
  // of the fields.
  return getBatchesPerSubgroup().size();
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
    int64_t expectedShape = getSubgroupsPerWorkgroup()[i] *
                            getBatchesPerSubgroup()[i] *
                            getOutersPerBatch()[i] * getThreadsPerOuter()[i] *
                            getElementsPerThread()[i];
    if (expectedShape != shape[i]) {
      std::string shapeStr;
      llvm::raw_string_ostream shapeOs(shapeStr);
      llvm::interleaveComma(shape, shapeOs);
      std::string layoutStr;
      llvm::raw_string_ostream layoutOs(layoutStr);
      printStripped(layoutOs);
      return emitError(loc, "Vector shape: [")
             << shapeStr << "] does not match the layout ("
             << layoutStr + ") at dim " << i
             << ". Dimension expected by layout: " << expectedShape
             << " actual: " << shape[i];
    }
  }
  return success();
}
NestedLayoutAttr NestedLayoutAttr::getChecked(
    llvm::function_ref<InFlightDiagnostic()> emitError, MLIRContext *context,
    ArrayRef<int64_t> subgroupsPerWorkgroup,
    ArrayRef<int64_t> batchesPerSubgroup, ArrayRef<int64_t> outersPerBatch,
    ArrayRef<int64_t> threadsPerOuter, ArrayRef<int64_t> elementsPerThread,
    ArrayRef<int64_t> subgroupStrides, ArrayRef<int64_t> threadStrides) {
  if (failed(NestedLayoutAttr::verify(emitError, subgroupsPerWorkgroup,
                                      batchesPerSubgroup, outersPerBatch,
                                      threadsPerOuter, elementsPerThread,
                                      subgroupStrides, threadStrides))) {
    return NestedLayoutAttr();
  }

  return NestedLayoutAttr::get(
      context, subgroupsPerWorkgroup, batchesPerSubgroup, outersPerBatch,
      threadsPerOuter, elementsPerThread, subgroupStrides, threadStrides);
}

NestedLayoutAttr NestedLayoutAttr::get(
    MLIRContext *context, ArrayRef<int64_t> subgroupsPerWorkgroup,
    ArrayRef<int64_t> batchesPerSubgroup, ArrayRef<int64_t> outersPerBatch,
    ArrayRef<int64_t> threadsPerOuter, ArrayRef<int64_t> elementsPerThread,
    ArrayRef<int64_t> subgroupStrides, ArrayRef<int64_t> threadStrides) {

  SmallVector<int64_t> normalizedSubgroupStrides(subgroupStrides);
  SmallVector<int64_t> normalizedThreadStrides(threadStrides);

  // Dimension of size 1 only have one element to distribute, so stride can be
  // anything. We normalize the stride to be 0, to have consistency.

  for (auto [stride, size] :
       llvm::zip_equal(normalizedSubgroupStrides, subgroupsPerWorkgroup)) {
    if (size == 1) {
      stride = 0;
    }
  }

  for (auto [stride, size] :
       llvm::zip_equal(normalizedThreadStrides, threadsPerOuter)) {
    if (size == 1) {
      stride = 0;
    }
  }

  return Base::get(context, subgroupsPerWorkgroup, batchesPerSubgroup,
                   outersPerBatch, threadsPerOuter, elementsPerThread,
                   normalizedSubgroupStrides, normalizedThreadStrides);
}

LogicalResult NestedLayoutAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    ArrayRef<int64_t> subgroupsPerWorkgroup,
    ArrayRef<int64_t> batchesPerSubgroup, ArrayRef<int64_t> outersPerBatch,
    ArrayRef<int64_t> threadsPerOuter, ArrayRef<int64_t> elementsPerThread,
    ArrayRef<int64_t> subgroupStrides, ArrayRef<int64_t> threadStrides) {

  size_t rank = subgroupsPerWorkgroup.size();
  auto checkTile = [&](ArrayRef<int64_t> tileShape) {
    if (tileShape.size() != rank) {
      emitError() << "all fields must have the same rank as the layout";
      return failure();
    }
    return success();
  };

  if (failed(checkTile(subgroupsPerWorkgroup)) ||
      failed(checkTile(batchesPerSubgroup)) ||
      failed(checkTile(outersPerBatch)) || failed(checkTile(threadsPerOuter)) ||
      failed(checkTile(elementsPerThread)) ||
      failed(checkTile(subgroupStrides)) || failed(checkTile(threadStrides))) {
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

  AffineExpr tidExpr, size, stride;
  bindDims(rewriter.getContext(), tidExpr);
  bindSymbols(rewriter.getContext(), size, stride);

  // (tid floordiv stride) mod size
  AffineMap threadTidMap =
      AffineMap::get(/*dims=*/1, /*syms=*/2, tidExpr.floorDiv(stride) % size);

  // (tid floordiv (stride * subgroup_size)) mod size
  AffineMap subgroupTidMap = AffineMap::get(
      /*dims=*/1, /*syms=*/2, tidExpr.floorDiv(stride * subgroupSize) % size);

  for (auto [dimSize, dimStride] :
       llvm::zip_equal(getSubgroupsPerWorkgroup(), getSubgroupStrides())) {
    // Dimension is not distributed.
    if (dimStride == 0) {
      virtualTids.push_back(rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexAttr(dimStride)));
      continue;
    }

    auto sizeVal =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(dimSize));
    auto strideVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(dimStride));
    virtualTids.push_back(rewriter.create<affine::AffineApplyOp>(
        loc, subgroupTidMap, ValueRange{threadId, sizeVal, strideVal}));
  }

  for (auto [dimSize, dimStride] :
       llvm::zip_equal(getThreadsPerOuter(), getThreadStrides())) {
    // Dimension is not distributed.
    if (dimStride == 0) {
      virtualTids.push_back(rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexAttr(dimStride)));
      continue;
    }

    auto sizeVal =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(dimSize));
    auto strideVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(dimStride));
    virtualTids.push_back(rewriter.create<affine::AffineApplyOp>(
        loc, threadTidMap, ValueRange{threadId, sizeVal, strideVal}));
  }

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
