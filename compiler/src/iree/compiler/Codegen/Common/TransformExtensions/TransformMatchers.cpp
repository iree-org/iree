// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/TransformExtensions/TransformMatchers.h"

#include "mlir/Analysis/SliceAnalysis.h"

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::IREE;

//===---------------------------------------------------------------------===//
// StructuredOpMatcher and friends.
//===---------------------------------------------------------------------===//

bool transform_dialect::StructuredOpMatcher::match(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp) return false;

  if (!llvm::all_of(predicates, [linalgOp](const PredicateFn &fn) {
        return fn(linalgOp);
      })) {
    return false;
  }

  captured = linalgOp;
  return true;
}

transform_dialect::StructuredOpMatcher &
transform_dialect::StructuredOpMatcher::dim(int64_t dimension, ShapeKind kind) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    SmallVector<int64_t> shape = linalgOp.getStaticLoopRanges();
    int64_t transformedDimension =
        dimension >= 0 ? dimension : shape.size() + dimension;
    if (transformedDimension >= shape.size()) return false;
    return ShapedType::isDynamic(shape[transformedDimension]) ^
           (kind == ShapeKind::Static);
  });
  return *this;
}

transform_dialect::StructuredOpMatcher &
transform_dialect::StructuredOpMatcher::dim(AllDims tag, ShapeKind kind) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    SmallVector<int64_t> shape = linalgOp.getStaticLoopRanges();
    return llvm::all_of(shape, [=](int64_t dimension) {
      return ShapedType::isDynamic(dimension) ^ (kind == ShapeKind::Static);
    });
  });
  return *this;
}

transform_dialect::StructuredOpMatcher &
transform_dialect::StructuredOpMatcher::dim(int64_t dimension,
                                            utils::IteratorType kind) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    unsigned rank = linalgOp.getNumLoops();
    int64_t transformedDimension =
        dimension >= 0 ? dimension : rank + dimension;
    if (transformedDimension >= rank) return false;

    utils::IteratorType iteratorKind =
        linalgOp.getIteratorTypesArray()[transformedDimension];
    return iteratorKind == kind;
  });
  return *this;
}
transform_dialect::StructuredOpMatcher &
transform_dialect::StructuredOpMatcher::dim(AllDims tag,
                                            utils::IteratorType kind) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    return llvm::all_of(
        linalgOp.getIteratorTypesArray(),
        [=](utils::IteratorType iteratorType) { return iteratorType == kind; });
  });
  return *this;
}

transform_dialect::StructuredOpMatcher &
transform_dialect::StructuredOpMatcher::dim(int64_t dimension,
                                            DivisibleBy divisibleBy) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    unsigned rank = linalgOp.getNumLoops();
    int64_t transformedDimension =
        dimension >= 0 ? dimension : rank + dimension;
    if (transformedDimension >= rank) return false;

    int64_t size = linalgOp.getStaticLoopRanges()[transformedDimension];
    return !ShapedType::isDynamic(size) && (size % divisibleBy.value == 0);
  });
  return *this;
}

transform_dialect::StructuredOpMatcher &
transform_dialect::StructuredOpMatcher::input(AllOperands tag, IsPermutation) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    // all_of with a lambda requires const-casting dance, so using a loop.
    for (OpOperand *operand : linalgOp.getDpsInputOperands()) {
      if (!linalgOp.getMatchingIndexingMap(operand).isPermutation())
        return false;
    }
    return true;
  });
  return *this;
}

transform_dialect::StructuredOpMatcher &
transform_dialect::StructuredOpMatcher::output(AllOperands tag, IsPermutation) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    for (OpOperand *operand : linalgOp.getDpsInitOperands()) {
      if (!linalgOp.getMatchingIndexingMap(operand).isPermutation())
        return false;
    }
    return true;
  });
  return *this;
}

transform_dialect::StructuredOpMatcher &
transform_dialect::StructuredOpMatcher::output(int64_t position,
                                               ElementTypeBitWidth width) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    int64_t updatedPosition =
        position >= 0 ? position : linalgOp.getNumDpsInits() + position;
    if (updatedPosition >= linalgOp.getNumDpsInits()) return false;
    auto shapedType = linalgOp.getDpsInitOperand(updatedPosition)
                          ->get()
                          .getType()
                          .dyn_cast<ShapedType>();
    return shapedType && shapedType.getElementType().isIntOrFloat() &&
           shapedType.getElementType().getIntOrFloatBitWidth() == width.value;
  });
  return *this;
}

transform_dialect::StructuredOpMatcher &
transform_dialect::StructuredOpMatcher::output(int64_t position,
                                               SingleCombinerReduction tag) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    int64_t updatedPosition =
        position >= 0 ? position : linalgOp.getNumDpsInits() + position;
    if (updatedPosition >= linalgOp.getNumDpsInits()) return false;
    SmallVector<Operation *> combinerOps;
    return matchReduction(linalgOp.getRegionOutputArgs(), updatedPosition,
                          combinerOps) &&
           llvm::hasSingleElement(combinerOps);
  });
  return *this;
}

//===---------------------------------------------------------------------===//
// MatchCallbackResult.
//===---------------------------------------------------------------------===//

ArrayRef<Operation *> transform_dialect::MatchCallbackResult::getPayloadGroup(
    unsigned position) const {
  assert(position < payloadGroupLengths.size());
  int64_t start = 0;
  for (unsigned i = 0; i < position; ++i) {
    start += payloadGroupLengths[i];
  }
  return llvm::makeArrayRef(payloadOperations)
      .slice(start, payloadGroupLengths[position]);
}
