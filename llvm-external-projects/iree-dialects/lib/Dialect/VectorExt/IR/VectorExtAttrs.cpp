// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace mlir::iree_compiler::IREE::VectorExt {

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
bool LayoutAttr::isValidLayout(ArrayRef<int64_t> shape) const {
  for (auto perDimLayout : llvm::enumerate(getLayouts())) {
    ArrayRef<int64_t> layoutShape = perDimLayout.value().getShapes();
    int64_t computedShape =
        std::reduce(layoutShape.begin(), layoutShape.end(),
                    static_cast<int64_t>(1), std::multiplies<int64_t>());
    int64_t expectedShape = shape[perDimLayout.index()];
    if (computedShape != expectedShape) {
      return false;
    }
  }
  return true;
}

// Project out the layout for the specified dimensions
// resulting in the layout for a lower dimensional vector.
VectorLayoutInterface LayoutAttr::project(ArrayRef<bool> droppedDims) const {
  assert(droppedDims.size() == getLayouts().size() &&
         "droppedDims size must match layout size");

  ArrayRef<PerDimLayoutAttr> layouts = getLayouts();
  assert(droppedDims.size() == layouts.size());
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
  assert(permutation.size() == getLayouts().size() &&
         "permutation size must match layout size");

  ArrayRef<PerDimLayoutAttr> layouts = getLayouts();
  assert(permutation.size() == layouts.size());
  SmallVector<PerDimLayoutAttr> newLayouts;
  for (unsigned index : permutation) {
    assert(index >= 0 && index < layouts.size());
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
  assert(dim >= 0 && dim < getLayouts().size());
  return getLayouts()[dim];
}

std::optional<int64_t> LayoutAttr::getBatchDim(int64_t dim) {
  assert(dim < getLayouts().size());
  PerDimLayoutAttr layout = getDimLayout(dim);
  for (auto [name, shape] :
       llvm::zip_equal(layout.getLabels(), layout.getShapes())) {
    if (isBatchDimension(name.getValue()))
      return shape;
  }
  return std::nullopt;
}

std::optional<int64_t> LayoutAttr::getLaneDim(int64_t dim) {
  assert(dim < getLayouts().size());
  PerDimLayoutAttr layout = getDimLayout(dim);
  for (auto [name, shape] :
       llvm::zip_equal(layout.getLabels(), layout.getShapes())) {
    if (isLaneDimension(name.getValue()))
      return shape;
  }
  return std::nullopt;
}

std::optional<LayoutDimension> LayoutAttr::getLane(int64_t dim) {
  assert(dim < getLayouts().size());
  PerDimLayoutAttr layout = getDimLayout(dim);
  for (auto [name, shape] :
       llvm::zip_equal(layout.getLabels(), layout.getShapes())) {
    if (isLaneDimension(name.getValue()))
      return name.getValue();
  }
  return std::nullopt;
}

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
    assert("Invalid dimension! Expected lane dimension.");
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
  assert(droppedDims.size() == getBatchesPerSubgroup().size() &&
         "droppedDims size must match layout rank");

  // Projection for this layout simply means the sizes along the projected
  // are dropped.
  SmallVector<int64_t> subgroupCount;
  SmallVector<int64_t> batchCount;
  SmallVector<int64_t> outerCount;
  SmallVector<int64_t> threadCount;
  SmallVector<int64_t> elementCount;
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
      indexToRankReducedIndexMap[idx] = count++;
    }
  }
  // This layout is invalid for rank-0 vectors.
  assert(count >= 0 && "unimplemented rank-0 vector");

  auto getRankReducedPermutation =
      [&](ArrayRef<int64_t> perm) -> SmallVector<int64_t> {
    SmallVector<int64_t> newPerm;
    for (auto i : perm) {
      if (indexToRankReducedIndexMap.contains(i)) {
        newPerm.push_back(indexToRankReducedIndexMap[i]);
      }
    }
    return newPerm;
  };

  SmallVector<int64_t> subgroupOrder =
      getRankReducedPermutation(getSubgroupOrder());
  SmallVector<int64_t> batchOrder = getRankReducedPermutation(getBatchOrder());
  SmallVector<int64_t> outerOrder = getRankReducedPermutation(getOuterOrder());
  SmallVector<int64_t> threadOrder =
      getRankReducedPermutation(getThreadOrder());
  SmallVector<int64_t> elementOrder =
      getRankReducedPermutation(getElementOrder());

  // Compose the projected dims with the basis mask to get the new active
  // ids. Active ids indicates that we should use the ids marked as true, and
  // projected dims drop the dims marked as true. So to get the new mask, we
  // turn off all of the currently `true` ids marked as projected. For example:
  //
  // subgroup_active_ids = [true,  true,  false, true]
  // projected_dims =      [false, true,         false]
  //
  // new_active_ids =      [true,  false, false, true]
  auto composeMasks = [&](SmallVector<bool> &newMask, ArrayRef<bool> mask) {
    int64_t rankReducedIdx = 0;
    for (auto [i, active] : llvm::enumerate(newMask)) {
      if (active) {
        newMask[i] = !mask[rankReducedIdx];
        rankReducedIdx++;
      }
    }
  };
  SmallVector<bool> subgroupMask(getSubgroupActiveIds());
  SmallVector<bool> threadMask(getThreadActiveIds());
  composeMasks(subgroupMask, droppedDims);
  composeMasks(threadMask, droppedDims);

  return NestedLayoutAttr::get(getContext(), subgroupCount, subgroupOrder,
                               batchCount, batchOrder, outerCount, outerOrder,
                               threadCount, threadOrder, elementCount,
                               elementOrder, getSubgroupBasis(), subgroupMask,
                               getThreadBasis(), threadMask);
}

VectorLayoutInterface
NestedLayoutAttr::permute(ArrayRef<int64_t> permutation) const {
  llvm_unreachable("Not yet implemented");
}

/// We distribute to:
/// <BATCH x OUTER x ELEMENT>
SmallVector<int64_t> NestedLayoutAttr::getDistributedShape() const {
  SmallVector<int64_t> shape;
  shape.append(applyPermutation(getBatchesPerSubgroup(), getBatchOrder()));
  shape.append(applyPermutation(getOutersPerBatch(), getOuterOrder()));
  shape.append(applyPermutation(getElementsPerThread(), getElementOrder()));
  return shape;
}

bool NestedLayoutAttr::isValidLayout(ArrayRef<int64_t> shape) const {
  // Multiply all shapes in the layout.
  for (int i = 0, e = shape.size(); i < e; ++i) {
    int64_t expectedShape = getSubgroupsPerWorkgroup()[i] *
                            getBatchesPerSubgroup()[i] *
                            getOutersPerBatch()[i] * getThreadsPerOuter()[i] *
                            getElementsPerThread()[i];
    if (expectedShape != shape[i]) {
      return false;
    }
  }
  return true;
}

// TODO: These things should ideally go into the parser when we have a custom
// parser.
LogicalResult NestedLayoutAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    ArrayRef<int64_t> subgroupsPerWorkgroup, ArrayRef<int64_t> subgroupOrder,
    ArrayRef<int64_t> batchesPerSubgroup, ArrayRef<int64_t> batchOrder,
    ArrayRef<int64_t> outersPerBatch, ArrayRef<int64_t> outerOrder,
    ArrayRef<int64_t> threadsPerOuter, ArrayRef<int64_t> threadOrder,
    ArrayRef<int64_t> elementsPerThread, ArrayRef<int64_t> elementOrder,
    ArrayRef<int64_t> subgroupBasis, ArrayRef<bool> subgroupActiveIds,
    ArrayRef<int64_t> threadBasis, ArrayRef<bool> threadActiveIds) {

  size_t rank = subgroupsPerWorkgroup.size();
  auto checkTile = [&](ArrayRef<int64_t> tileShape, ArrayRef<int64_t> order) {
    if (tileShape.size() != rank || order.size() != rank) {
      emitError() << "all tiles must have the same rank as the layout";
      return failure();
    }
    if (!mlir::isPermutationVector(order)) {
      emitError() << "all orderings must be permutation vectors";
      return failure();
    }
    return success();
  };

  if (failed(checkTile(subgroupsPerWorkgroup, subgroupOrder)) ||
      failed(checkTile(batchesPerSubgroup, batchOrder)) ||
      failed(checkTile(outersPerBatch, outerOrder)) ||
      failed(checkTile(threadsPerOuter, threadOrder)) ||
      failed(checkTile(elementsPerThread, elementOrder))) {
    return failure();
  }

  auto checkBasis = [&](ArrayRef<int64_t> basis, ArrayRef<bool> mask) {
    if (basis.size() != mask.size()) {
      emitError() << "basis and active id mask must be the same length";
      return failure();
    }
    if (llvm::count(mask, true) != rank) {
      emitError()
          << "number of active basis ids must be equal to the layout rank";
    }
    return success();
  };
  if (failed(checkBasis(subgroupBasis, subgroupActiveIds)) ||
      failed(checkBasis(threadBasis, threadActiveIds))) {
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
NestedLayoutAttr::computeThreadIds(Value threadId,
                                   RewriterBase &rewriter) const {
  // The subgroup/thread bases tell us the ranges of the corresponding id from
  // slowest varying to fastest varying. Thus to get the correct basis ids, we
  // simply concatenate the sizes and delinearize the single thread id to those
  // sizes. For example:
  //
  // subgroup_basis = [3, 5]
  // thread_basis = [7, 9]
  //
  // subgroup_id(Y, X), thread_id(Y, X) = affine.delinearize_index(3, 5, 7, 9)
  auto basisSizes =
      llvm::concat<const int64_t>(getSubgroupBasis(), getThreadBasis());

  SmallVector<OpFoldResult> basisIndexAttr;
  for (int64_t basisIndex : basisSizes) {
    basisIndexAttr.push_back(rewriter.getIndexAttr(basisIndex));
  }

  SmallVector<Value> delinearized =
      rewriter
          .create<mlir::affine::AffineDelinearizeIndexOp>(
              threadId.getLoc(), threadId, basisIndexAttr)
          .getResults();

  // The subgroups_per_workgroup and threads_per_outer fields represent the
  // number of subgroups/threads along each vector dimension. To get the sizes
  // in the order of slowest varying to fastest varying (to match with the ids
  // delinearized based on the basis), we need to apply the subgroup/thread
  // permutation orders.
  auto tileSizes = llvm::concat<const int64_t>(
      applyPermutation(getSubgroupsPerWorkgroup(), getSubgroupOrder()),
      applyPermutation(getThreadsPerOuter(), getThreadOrder()));
  auto tileSizesIterator = tileSizes.begin();

  auto activeIdFilter =
      llvm::concat<const bool>(getSubgroupActiveIds(), getThreadActiveIds());

  // Modulo the active delinearized subgroup/thread ids by the number of unique
  // elements distributed to those ids. The only difference between subgroup
  // and thread dimensions is the order in which they are "divided out" of the
  // underlying vector (i.e. vector_shape /= subgroups -> batches -> outers ->
  // threads -> elements). There is no requirement that a subgroup id only
  // spans subgroups.
  //
  // thread_basis = [8, 4, 2]
  // active_thread_ids = [true, false, true]
  // threads_per_outer = [4, 2]
  //
  // To obtain the thread ids, we just delinearize based on the basis.
  //
  // i0, i1, i2 = affine.delinearize_inds %threadId (8, 4, 2)
  //
  // And then to get the thread id for the layout, we only consider the active
  // ids:
  //
  // layout_id0 = i0 % 4
  // layout_id1 = i2 % 2
  //
  // The typical way this is used it to implicitly broadcast data across
  // threads. For example, take a simpler case of the following:
  //
  // vector_shape = vector<2>
  // thread_basis = [2, 2]
  // active_thread_ids = [true, false]
  // threads_per_outer = [2]
  //
  // If we give the two elements in the vector labels, say s0 and s1, we can
  // see what this layout assigns as ids when doing a read of those two values
  // across 4 threads.
  //
  // %id = gpu.flat_thread_id   // In range [0, 4)
  // i0, i1 = affine.delinearize_index %id (2, 2)
  // %id = 0, 1, 2, 3
  // ----------------
  // i0  = 0, 0, 1, 1
  // i1  = 0, 1, 0, 1
  //
  // %0 = vector.load mem[i0]
  //
  // %id = 0, 1, 2, 3
  // ----------------
  // %0  = s0 s0 s1 s1
  //
  // If we instead had this layout:
  //
  // thread_basis = [4]
  // active_thread_ids = [true]
  // threads_per_outer = [2]
  //
  // With the modulus, we would get:
  //
  // %id = gpu.flat_thread_id   // In range [0, 4)
  // i0 = %id = affine.delinearize_index %id (4)
  // layout_i0 = i0 % 2
  //
  // %id        = 0, 1, 2, 3
  // ----------------
  // layout_i0  = 0, 1, 0, 1
  //
  // %0 = vector.load mem[layout_i0]
  //
  // %id = 0, 1, 2, 3
  // ----------------
  // %0  = s0 s1 s0 s1
  for (auto [delinearized, basis, isActive] :
       llvm::zip_equal(delinearized, basisSizes, activeIdFilter)) {
    if (!isActive) {
      continue;
    }
    int64_t tile = *tileSizesIterator;
    tileSizesIterator++;
    if (basis == tile) {
      continue;
    }

    AffineMap modMap =
        AffineMap::get(1, 0, rewriter.getAffineDimExpr(0) % tile);
    delinearized = rewriter.create<affine::AffineApplyOp>(threadId.getLoc(),
                                                          modMap, delinearized);
  }

  return delinearized;
}

//===----------------------------------------------------------------------===//
// Custom Parsers/Printers
//===----------------------------------------------------------------------===//

// Custom parser/printer to construct the permutation based on the rank of the
// sizes corresponding to this order.
static ParseResult parsePermutation(AsmParser &parser, StringRef baseName,
                                    ArrayRef<int64_t> sizes, bool parseComma,
                                    SmallVector<int64_t> &permutation) {
  if (failed(parser.parseOptionalKeyword(baseName))) {
    permutation = llvm::to_vector(llvm::seq<int64_t>(0, sizes.size()));
    return success();
  }
  if (failed(parser.parseEqual())) {
    return failure();
  }
  if (parser.parseLSquare()) {
    return failure();
  }
  auto arrayParser = FieldParser<SmallVector<int64_t>>::parse(parser);
  if (failed(arrayParser)) {
    parser.emitError(parser.getCurrentLocation(),
                     "failed to parse permutation parameter '")
        << baseName << "' which is to be a `::llvm::ArrayRef<int64_t>`";
  }
  if (parser.parseRSquare()) {
    return failure();
  }
  if (parseComma) {
    if (parser.parseComma()) {
      return failure();
    }
  }
  permutation = *arrayParser;
  return success();
}

static void printPermutation(AsmPrinter &p, StringRef baseName,
                             ArrayRef<int64_t> sizes, bool printComma,
                             ArrayRef<int64_t> permutation) {
  if (isIdentityPermutation(permutation)) {
    return;
  }
  p << baseName;
  // This is called without whitespace inserted by default for optionality.
  // Insert it explicitly instead.
  p << ' ';
  p << '=';
  p << ' ';
  p << '[';
  llvm::interleaveComma(permutation, p);
  p << ']';
  if (printComma) {
    p << ',' << ' ';
  }
}

// Custom parser/printer for a basis (array of i64 values) and a mask (array
// of boolean values).
static ParseResult parseBasis(AsmParser &parser, StringRef basisName,
                              StringRef maskName, bool parseComma,
                              SmallVector<int64_t> &basis,
                              SmallVector<bool> &mask) {
  if (failed(parser.parseKeyword(basisName)) || failed(parser.parseEqual()) ||
      failed(parser.parseLSquare())) {
    return failure();
  }
  auto arrayParser = FieldParser<SmallVector<int64_t>>::parse(parser);
  if (failed(arrayParser)) {
    parser.emitError(parser.getCurrentLocation(),
                     "failed to parse basis parameter '")
        << basisName << "' which is to be a `::llvm::ArrayRef<int64_t>`";
  }
  basis = *arrayParser;
  if (parser.parseRSquare()) {
    return failure();
  }
  // Optionally parse a comma between the basis and mask.
  if (parser.parseOptionalComma()) {
    // If we were supposed to find a comma, fail parsing.
    if (parseComma) {
      return failure();
    }
    // If it was fine not to find a comma, set the mask. If the comma was
    // missing this will fail to parse the closing angle bracket.
    mask = SmallVector<bool>(basis.size(), true);
    return success();
  }
  // There is a comma, meaning we either must find the mask, or we shouldn't
  // have expected a comma.
  if (failed(parser.parseOptionalKeyword(maskName))) {
    if (!parseComma) {
      return failure();
    }
    mask = SmallVector<bool>(basis.size(), true);
    return success();
  }

  if (failed(parser.parseEqual()) || failed(parser.parseLSquare())) {
    return failure();
  }
  auto maskParser = FieldParser<SmallVector<bool>>::parse(parser);
  if (failed(maskParser)) {
    parser.emitError(parser.getCurrentLocation(),
                     "failed to parse mask parameter '")
        << maskName << "' which is to be a `::llvm::ArrayRef<bool>`";
  }
  if (failed(parser.parseRSquare()) ||
      (parseComma && failed(parser.parseComma()))) {
    return failure();
  }
  mask = *maskParser;

  return success();
}

static void printBasis(AsmPrinter &p, StringRef basisName, StringRef maskName,
                       bool printComma, ArrayRef<int64_t> basis,
                       ArrayRef<bool> mask) {
  p << basisName;
  // This is called without whitespace inserted by default for optionality.
  // Insert it explicitly instead.
  p << ' ';
  p << '=';
  p << ' ';
  p << '[';
  llvm::interleaveComma(basis, p);
  p << ']';
  if (llvm::any_of(mask, [](bool b) { return !b; })) {
    p << ',' << ' ';
    p << maskName;
    p << '=';
    p << ' ';
    p << '[';
    llvm::interleaveComma(mask, p);
    p << ']';
  }
  if (printComma) {
    p << ',' << ' ';
  }
}

} // namespace mlir::iree_compiler::IREE::VectorExt

using namespace mlir::iree_compiler::IREE::VectorExt;

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtEnums.cpp.inc" // IWYU pragma: keep

#define GET_ATTRDEF_CLASSES
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtAttrs.cpp.inc" // IWYU pragma: keep

void IREEVectorExtDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtAttrs.cpp.inc"
      >();
}
