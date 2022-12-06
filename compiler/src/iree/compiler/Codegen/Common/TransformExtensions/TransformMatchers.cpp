// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/TransformExtensions/TransformMatchers.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

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

/// Traverses the transitive sources of `val` until it reaches an operation that
/// is not a known "subset-like" operation, i.e. `extract_slice` or
/// `foreach_thread`.
static Operation *traverseSubsetsBackwards(Value val) {
  do {
    Operation *op = val.getDefiningOp();
    if (!op) {
      // TODO: This should likely be done via RegionBranchOpInterface as a sort
      // of data flow analysis.
      auto bbArg = val.cast<BlockArgument>();
      Operation *blockOp = bbArg.getOwner()->getParentOp();
      assert(blockOp && "detached block");
      if (auto loop = dyn_cast<scf::ForeachThreadOp>(blockOp)) {
        val = loop.getTiedOpOperand(bbArg)->get();
        continue;
      }
      return blockOp;
    }

    // TODO: We may eventually want a "subset-like" interface that we can use to
    // traverse ops here and in post-canonicalization replacement
    // identification.
    if (auto extractSlice = dyn_cast<tensor::ExtractSliceOp>(op)) {
      val = extractSlice.getSource();
      continue;
    }
    return op;
  } while (true);
}

/// Greedily traverses the transitive uses of `val` until it reaches an
/// operation that is not a known "subset-like" operation, i.e. `extract_slice`
/// or `foreach_thread`.
static Operation *traverseSubsetsForwardAnyUse(Value val) {
  do {
    for (OpOperand &use : val.getUses()) {
      Operation *user = use.getOwner();
      if (auto loop = dyn_cast<scf::ForeachThreadOp>(user)) {
        auto range = loop.getOutputBlockArguments();
        auto it = llvm::find_if(range, [&](BlockArgument bbarg) {
          return loop.getTiedOpOperand(bbarg) != &use;
        });
        if (it == range.end()) return user;
        val = *it;
        continue;
      }
      if (auto slice = dyn_cast<tensor::ExtractSliceOp>(user)) {
        val = slice.getResult();
        continue;
      }
      return user;
    }
  } while (true);
}

transform_dialect::StructuredOpMatcher &
transform_dialect::StructuredOpMatcher::input(int64_t position,
                                              SubsetOf subset) {
  // Implementation note: SubsetOf must *not* be passed by-reference because
  // it is typically a temporary constructed within the argument of a function
  // call, but it will be used in the lambda that outlives the temporary. The
  // lambda itself must capture by value for the same reason.
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    int64_t transformedPosition =
        position >= 0 ? position : linalgOp.getNumDpsInputs() + position;
    if (transformedPosition >= linalgOp.getNumDpsInputs()) return false;

    Operation *producer = traverseSubsetsBackwards(
        linalgOp.getDpsInputOperand(transformedPosition)->get());
    return subset.matcher.match(producer);
  });
  recordNestedMatcher(subset.matcher);
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

transform_dialect::StructuredOpMatcher &
transform_dialect::StructuredOpMatcher::output(int64_t position,
                                               SubsetOf subset) {
  // Implementation note: SubsetOf must *not* be passed by-reference because
  // it is typically a temporary constructed within the argument of a function
  // call, but it will be used in the lambda that outlives the temporary. The
  // lambda itself must capture by value for the same reason.
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    int64_t transformedPosition =
        position >= 0 ? position : linalgOp.getNumDpsInputs() + position;
    if (transformedPosition >= linalgOp.getNumDpsInputs()) return false;

    Operation *producer = traverseSubsetsBackwards(
        linalgOp.getDpsInitOperand(transformedPosition)->get());
    return subset.matcher.match(producer);
  });
  recordNestedMatcher(subset.matcher);
  return *this;
}

transform_dialect::StructuredOpMatcher &
transform_dialect::StructuredOpMatcher::result(int64_t position, HasAnyUse tag,
                                               SubsetOf subset,
                                               OptionalMatch optional) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    int64_t transformedPosition =
        position >= 0 ? position : linalgOp->getNumResults() + position;
    if (transformedPosition >= linalgOp->getNumResults()) return false;

    Operation *user =
        traverseSubsetsForwardAnyUse(linalgOp->getResult(transformedPosition));
    return subset.matcher.match(user) || optional.value;
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

//===---------------------------------------------------------------------===//
// Case-specific matcher builders.
//===---------------------------------------------------------------------===//

static constexpr unsigned kCudaWarpSize = 32;

void transform_dialect::makeGPUReductionMatcher(
    transform_dialect::StructuredOpMatcher &reduction,
    transform_dialect::StructuredOpMatcher &fill,
    transform_dialect::StructuredOpMatcher &leading,
    transform_dialect::StructuredOpMatcher &trailing) {
  fill = m_StructuredOp<linalg::FillOp>();
  trailing = m_StructuredOp<linalg::GenericOp>()
                 .input(AllOperands(), IsPermutation())
                 .output(AllOperands(), IsPermutation())
                 .input(NumEqualsTo(1))
                 .output(NumEqualsTo(1));
  leading = trailing;
  reduction = m_StructuredOp()
                  .dim(AllDims(), ShapeKind::Static)
                  .dim(-1, utils::IteratorType::reduction)
                  .dim(-1, DivisibleBy(kCudaWarpSize))
                  // Can be extended to projected permutation with broadcast.
                  .input(AllOperands(), IsPermutation())
                  // TODO: we want to accept any input position here.
                  .input(0, leading, OptionalMatch())
                  .output(NumEqualsTo(1))
                  .output(0, fill)
                  // Only single combiner over 32 bits for now due to
                  // reduction distribution.
                  .output(0, ElementTypeBitWidth(32))
                  .output(0, SingleCombinerReduction())
                  .result(0, HasAnyUse(), trailing, OptionalMatch());
}

void transform_dialect::makeGPUSplitReductionMatcher(
    transform_dialect::StructuredOpMatcher &parallel_reduction,
    transform_dialect::StructuredOpMatcher &combiner_reduction,
    transform_dialect::StructuredOpMatcher &parallel_fill,
    transform_dialect::StructuredOpMatcher &original_fill,
    transform_dialect::StructuredOpMatcher &leading,
    transform_dialect::StructuredOpMatcher &trailing) {
  original_fill = m_StructuredOp<linalg::FillOp>();
  parallel_fill = m_StructuredOp<linalg::FillOp>();
  trailing = m_StructuredOp<linalg::GenericOp>()
                 .input(AllOperands(), IsPermutation())
                 .output(AllOperands(), IsPermutation())
                 .input(NumEqualsTo(1))
                 .output(NumEqualsTo(1));
  leading = m_StructuredOp<linalg::GenericOp>()
                .input(AllOperands(), IsPermutation())
                .output(AllOperands(), IsPermutation())
                .input(NumEqualsTo(1))
                .output(NumEqualsTo(1));
  parallel_reduction = m_StructuredOp()
                           .dim(AllDims(), ShapeKind::Static)
                           .dim(-1, utils::IteratorType::reduction)
                           .input(AllOperands(), IsPermutation())
                           // TODO: we want to accept any input position here.
                           .input(0, leading, OptionalMatch())
                           .output(NumEqualsTo(1))
                           .output(0, parallel_fill);
  combiner_reduction =
      m_StructuredOp()
          .dim(AllDims(), ShapeKind::Static)
          .dim(-1, utils::IteratorType::reduction)
          // Can be extended to projected permutation with broadcast.
          .input(AllOperands(), IsPermutation())
          .input(0, SubsetOf(parallel_reduction))
          .output(NumEqualsTo(1))
          .output(0, SubsetOf(original_fill))
          .output(0, ElementTypeBitWidth(32))
          .output(0, SingleCombinerReduction())
          .result(0, HasAnyUse(), SubsetOf(trailing), OptionalMatch());
}
