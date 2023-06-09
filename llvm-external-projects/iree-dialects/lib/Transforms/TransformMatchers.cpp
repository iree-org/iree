// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Transforms/TransformMatchers.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"
#include <mlir/Dialect/Tensor/IR/Tensor.h>

using namespace mlir;

#define DEBUG_TYPE "transform-matchers"
#define DBGS() llvm::dbgs() << "[" DEBUG_TYPE "] "
#define DBGSNL() llvm::dbgs() << "\n[" DEBUG_TYPE "] "

//===---------------------------------------------------------------------===//
// CapturingMatcherBase
//===---------------------------------------------------------------------===//

void transform_ext::CapturingMatcherBase::getAllNested(
    SmallVectorImpl<CapturingOpMatcher *> &nested) {

  SetVector<CapturingOpMatcher *> found;
  found.insert(nested.begin(), nested.end());
  int64_t start = found.size();

  auto appendOne = [&found](CapturingMatcherBase &one) {
    found.insert(one.nestedCapturingMatchers.begin(),
                 one.nestedCapturingMatchers.end());
    for (CapturingValueMatcher *valueMatcher :
         one.nestedCapturingValueMatchers) {
      found.insert(valueMatcher->nestedCapturingMatchers.begin(),
                   valueMatcher->nestedCapturingMatchers.end());
    }
  };

  appendOne(*this);
  for (int64_t position = start; position < found.size(); ++position) {
    appendOne(*found[position]);
  }

  llvm::append_range(nested, found.getArrayRef());
}

void transform_ext::CapturingMatcherBase::getAllNestedValueMatchers(
    SmallVectorImpl<CapturingValueMatcher *> &nested) {

  SetVector<CapturingValueMatcher *> found;
  found.insert(nested.begin(), nested.end());
  int64_t start = found.size();

  auto appendOne = [&found](CapturingMatcherBase &one) {
    found.insert(one.nestedCapturingValueMatchers.begin(),
                 one.nestedCapturingValueMatchers.end());
    for (CapturingOpMatcher *opMatcher : one.nestedCapturingMatchers) {
      found.insert(opMatcher->nestedCapturingValueMatchers.begin(),
                   opMatcher->nestedCapturingValueMatchers.end());
    }
  };

  appendOne(*this);
  for (int64_t position = start; position < found.size(); ++position) {
    appendOne(*found[position]);
  }

  llvm::append_range(nested, found.getArrayRef());
}

void transform_ext::CapturingMatcherBase::resetCapture() {
  SmallVector<CapturingOpMatcher *> nested;
  getAllNested(nested);
  for (CapturingOpMatcher *matcher : nested) {
    matcher->captured = nullptr;
  }
  SmallVector<CapturingValueMatcher *> nestedValue;
  getAllNestedValueMatchers(nestedValue);
  for (CapturingValueMatcher *matcher : nestedValue) {
    matcher->captured = nullptr;
  }
}

//===---------------------------------------------------------------------===//
// CapturingOpMatcher
//===---------------------------------------------------------------------===//

bool transform_ext::CapturingOpMatcher::checkAllTilableMatched(
    Operation *parent, Operation *op,
    ArrayRef<transform_ext::CapturingOpMatcher *> matchers) {
  LLVM_DEBUG(DBGS() << "all tilable ops captured");
  int64_t numTilableOps = 0;
  if (!parent)
    return false;
  parent->walk([&](TilingInterface Op) { ++numTilableOps; });

  llvm::SmallPtrSet<Operation *, 6> matched;
  for (CapturingOpMatcher *nested : matchers) {
    if (Operation *captured = nested->getCaptured()) {
      matched.insert(captured);
    }
  }

  // Don't forget to include the root matcher.
  matched.insert(op);
  return numTilableOps == matched.size();
}

bool transform_ext::CapturingOpMatcher::match(Operation *op) {
  auto debugRAII =
      llvm::make_scope_exit([] { LLVM_DEBUG(DBGS() << "-------\n"); });
  LLVM_DEBUG(DBGS() << "matching: " << *op << "\n");

  if (getCaptured()) {
    LLVM_DEBUG(DBGS() << "found an already captured op: ");
    if (getCaptured() == op) {
      LLVM_DEBUG(llvm::dbgs() << "same\n");
      return true;
    } else {
      LLVM_DEBUG(llvm::dbgs() << "different\n");
      return false;
    }
  }

  if (!llvm::all_of(predicates, [op](const PredicateFn &fn) {
        bool result = fn(op);
        LLVM_DEBUG(llvm::dbgs() << ": " << result << "\n");
        return result;
      })) {
    return false;
  }

  captured = op;
  return true;
}

void transform_ext::CapturingOpMatcher::debugOutputForCreate(
    ArrayRef<StringRef> opNames) {
  LLVM_DEBUG(DBGS() << "operation type is one of {";
             llvm::interleaveComma(opNames, llvm::dbgs()); llvm::dbgs() << "}");
}

/// Apply the given matcher to the given object, produce debug messages.
template <typename Matcher, typename Object = typename llvm::function_traits<
                                typename Matcher::match>::template args<0>>
static bool recursiveMatch(Matcher &matcher, Object &object,
                           StringRef extraMessage = "") {
  LLVM_DEBUG(llvm::dbgs() << "\n[" DEBUG_TYPE "] "
                          << "start recursive match (" << extraMessage
                          << ") {\n");
  bool result = matcher.match(object);
  LLVM_DEBUG(DBGS() << "} end recursive match");
  return result;
}

transform_ext::CapturingOpMatcher &
transform_ext::CapturingOpMatcher::alternatives(
    transform_ext::CapturingOpMatcher &first,
    transform_ext::CapturingOpMatcher &second) {
  addPredicate([&first, &second](Operation *op) {
    LLVM_DEBUG(DBGS() << "matching alternatives\n");
    return recursiveMatch(first, op, "alternative 1") ||
           recursiveMatch(second, op, "alternative 2");
  });
  return *this;
}

//---------------------------------------------------------------------------//
// Predicates for operands and results.
//---------------------------------------------------------------------------//

transform_ext::CapturingOpMatcher &
transform_ext::CapturingOpMatcher::operand(transform_ext::NumEqualsTo num) {
  addPredicate([=](Operation *op) {
    LLVM_DEBUG(DBGS() << "operation has exactly " << num.value << " operands");
    return num.value == op->getNumOperands();
  });
  return *this;
}

/// If `pos` is negative, returns the number of the operand in op starting from
/// the last. For example, -1 means the last operand, -2 means the
/// second-to-last, etc. Returns nullopt if pos is out-of-bounds, both positive
/// and negative.
static std::optional<int64_t> remapNegativeOperandNumber(int64_t pos,
                                                         Operation *op) {
  int64_t updated = pos < 0 ? op->getNumOperands() + pos : pos;
  if (updated < 0 || updated >= op->getNumOperands()) {
    LLVM_DEBUG(DBGS() << "match operand #" << pos
                      << "that does not exist in the operation");
    return std::nullopt;
  }
  return updated;
}

transform_ext::CapturingOpMatcher &
transform_ext::CapturingOpMatcher::operand(int64_t pos,
                                           CapturingOpMatcher &nested) {
  addPredicate([pos, &nested](Operation *op) {
    std::optional<int64_t> operandNo = remapNegativeOperandNumber(pos, op);
    if (!operandNo)
      return false;
    LLVM_DEBUG(DBGS() << "operand #" << pos << " is defined by an operation");
    Operation *definingOp = op->getOperand(*operandNo).getDefiningOp();
    if (!definingOp)
      return false;
    return recursiveMatch(nested, definingOp);
  });
  recordNestedMatcher(nested);
  return *this;
}

transform_ext::CapturingOpMatcher &
transform_ext::CapturingOpMatcher::operand(int64_t pos,
                                           CapturingValueMatcher &nested) {
  addPredicate([pos, &nested](Operation *op) {
    std::optional<int64_t> operandNo = remapNegativeOperandNumber(pos, op);
    if (!operandNo)
      return false;
    LLVM_DEBUG(DBGS() << "operand #" << pos << " is");
    Value operand = op->getOperand(*operandNo);
    return recursiveMatch(nested, operand);
  });
  recordNestedMatcher(nested);
  return *this;
}

transform_ext::CapturingOpMatcher &transform_ext::CapturingOpMatcher::operand(
    int64_t position, std::function<bool(llvm::APFloat)> floatValueFn) {
  addPredicate([position,
                floatValueFn = std::move(floatValueFn)](Operation *op) -> bool {
    std::optional<int64_t> operandNo = remapNegativeOperandNumber(position, op);
    if (!operandNo)
      return false;

    LLVM_DEBUG(DBGS() << "operand #" << *operandNo
                      << " is a special floating point constant");
    auto cstOp =
        op->getOperand(*operandNo).getDefiningOp<arith::ConstantFloatOp>();
    if (!cstOp)
      return false;
    return floatValueFn(cstOp.value());
  });

  return *this;
}

transform_ext::CapturingOpMatcher &
transform_ext::CapturingOpMatcher::operand(int64_t position, ConstantFloatOne) {
  return operand(position,
                 [](llvm::APFloat value) { return value.isExactlyValue(1.0); });
}

transform_ext::CapturingOpMatcher &
transform_ext::CapturingOpMatcher::result(transform_ext::NumEqualsTo num) {
  addPredicate([=](Operation *op) {
    LLVM_DEBUG(DBGS() << "operation has exactly " << num.value << " results");
    return num.value == op->getNumResults();
  });
  return *this;
}

transform_ext::CapturingOpMatcher &
transform_ext::CapturingOpMatcher::result(int64_t pos,
                                          CapturingValueMatcher &nested) {
  addPredicate([pos, &nested](Operation *op) {
    int64_t updated = pos < 0 ? op->getNumResults() + pos : pos;
    if (updated < 0 || updated >= op->getNumResults()) {
      LLVM_DEBUG(DBGS() << "matching result #" << pos
                        << " that does not exist in the operation");
      return false;
    }
    LLVM_DEBUG(DBGS() << "result #" << pos << " is");
    Value result = op->getResult(updated);
    return recursiveMatch(nested, result);
  });
  recordNestedMatcher(nested);
  return *this;
}

//===---------------------------------------------------------------------===//
// CapturingValueMatcher
//===---------------------------------------------------------------------===//

namespace {
struct DebugPrintValueWrapper {
  Value value;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const DebugPrintValueWrapper &wrapper) {
  if (auto opResult = wrapper.value.dyn_cast<OpResult>()) {
    return os << "op result #" << opResult.getResultNumber() << " in "
              << wrapper.value;
  }

  auto blockArg = wrapper.value.cast<BlockArgument>();
  os << "block argument #" << blockArg.getArgNumber();
  Block *parentBlock = blockArg.getParentBlock();
  Region *parentRegion = parentBlock->getParent();
  if (!parentRegion) {
    os << " of a detached block:\n";
    parentBlock->print(os);
    return os;
  }

  os << " of block #"
     << std::distance(parentRegion->begin(), parentBlock->getIterator());
  Operation *parentOp = parentRegion->getParentOp();
  if (!parentOp) {
    os << " of a detached region:\n";
    for (Block &b : *parentRegion)
      b.print(os);
    return os;
  }

  os << " in region #" << parentRegion->getRegionNumber() << " of "
     << *parentOp;
  return os;
}
} // namespace

bool transform_ext::CapturingValueMatcher::match(Value value) {
  auto debugRAII =
      llvm::make_scope_exit([] { LLVM_DEBUG(DBGS() << "-------\n"); });
  LLVM_DEBUG(DBGS() << "matching " << DebugPrintValueWrapper{value} << "\n");

  if (getCaptured()) {
    LLVM_DEBUG(DBGS() << "found an already captured value: ");
    if (getCaptured() == value) {
      LLVM_DEBUG(llvm::dbgs() << "same\n");
      return true;
    } else {
      LLVM_DEBUG(llvm::dbgs() << "different\n");
      return false;
    }
  }

  for (const PredicateFn &fn : predicates) {
    bool result = fn(value);
    LLVM_DEBUG(llvm::dbgs() << ": " << result << "\n");
    if (!result)
      return false;
  }

  captured = value;
  return true;
}

transform_ext::ShapedValueMatcher::ShapedValueMatcher()
    : CapturingValueMatcher() {
  addPredicate([](Value value) {
    LLVM_DEBUG(DBGS() << "value is of shaped type");
    return value && value.getType().isa<ShapedType>();
  });
}

transform_ext::ShapedValueMatcher &
transform_ext::ShapedValueMatcher::rank(transform_ext::CaptureRank capture) {
  addPredicate([=](Value value) {
    LLVM_DEBUG(DBGS() << "capturing shaped value rank");
    capture.value = value.getType().cast<ShapedType>().getRank();
    return true;
  });
  return *this;
}

transform_ext::ShapedValueMatcher &
transform_ext::ShapedValueMatcher::dim(int64_t dimension, CaptureDim capture) {
  addPredicate([=](Value value) {
    LLVM_DEBUG(DBGS() << "capturing shaped value dimension " << dimension);
    capture.value = value.getType().cast<ShapedType>().getDimSize(dimension);
    return true;
  });
  return *this;
}

transform_ext::ShapedValueMatcher &
transform_ext::ShapedValueMatcher::dim(AllDims tag, CaptureDims captures) {
  (void)tag;
  addPredicate([=](Value value) {
    LLVM_DEBUG(DBGS() << "capturing all shaped value dimensions");
    ArrayRef<int64_t> shape = value.getType().cast<ShapedType>().getShape();
    captures.value.assign(shape.begin(), shape.end());
    return true;
  });
  return *this;
}

transform_ext::ShapedValueMatcher &
transform_ext::ShapedValueMatcher::elementType(CaptureElementType captures) {
  addPredicate([=](Value value) {
    LLVM_DEBUG(DBGS() << "capturing elementType");
    captures.value = value.getType().cast<ShapedType>().getElementType();
    return true;
  });
  return *this;
}

//===---------------------------------------------------------------------===//
// StructuredOpMatcher and friends.
//===---------------------------------------------------------------------===//

transform_ext::StructuredOpMatcher::StructuredOpMatcher() {
  addPredicate([](Operation *op) {
    LLVM_DEBUG(DBGS() << "is a structured op");
    return isa<linalg::LinalgOp>(op);
  });
}

//===---------------------------------------------------------------------===//
// Constraints on op rank and dims.
//===---------------------------------------------------------------------===//

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::rank(NumGreaterEqualTo minRank) {
  addPredicate([=](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "rank >= " << minRank.value);
    return linalgOp.getNumLoops() >= minRank.value;
  });
  return *this;
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::rank(NumLowerEqualTo maxRank) {
  addPredicate([=](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "rank <= " << maxRank.value);
    return linalgOp.getNumLoops() <= maxRank.value;
  });
  return *this;
}

StringRef stringifyShapeKind(transform_ext::ShapeKind kind) {
  switch (kind) {
  case transform_ext::ShapeKind::Static:
    return "static";
  case transform_ext::ShapeKind::Dynamic:
    return "dynamic";
  }
  llvm_unreachable("unhandled shape kind");
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::dim(SmallVector<int64_t> &&dimensions,
                                        ShapeKind kind) {
  addPredicate([dimensions = std::move(dimensions),
                kind](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "dimensions [";
               llvm::interleaveComma(dimensions, llvm::dbgs());
               llvm::dbgs() << "] are " << stringifyShapeKind(kind));
    SmallVector<int64_t> shape = linalgOp.getStaticLoopRanges();
    for (auto dimension : dimensions) {
      int64_t transformedDimension =
          dimension >= 0 ? dimension : shape.size() + dimension;
      if (transformedDimension < 0 || transformedDimension >= shape.size())
        return false;
      if (ShapedType::isDynamic(shape[transformedDimension]) ^
          (kind == ShapeKind::Static))
        continue;
      return false;
    }
    return true;
  });
  return *this;
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::dim(AllDims tag, ShapeKind kind) {
  addPredicate([=](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "all dimensions are " << stringifyShapeKind(kind));
    SmallVector<int64_t> shape = linalgOp.getStaticLoopRanges();
    return llvm::all_of(shape, [=](int64_t dimension) {
      return ShapedType::isDynamic(dimension) ^ (kind == ShapeKind::Static);
    });
  });
  return *this;
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::dim(SmallVector<int64_t> &&dimensions,
                                        utils::IteratorType kind) {
  addPredicate([dimensions = std::move(dimensions),
                kind](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "dimensions [";
               llvm::interleaveComma(dimensions, llvm::dbgs());
               llvm::dbgs() << "] are " << utils::stringifyIteratorType(kind));
    int64_t rank = linalgOp.getNumLoops();
    for (auto dimension : dimensions) {
      int64_t transformedDimension =
          dimension >= 0 ? dimension : rank + dimension;
      if (transformedDimension < 0 || transformedDimension >= rank)
        return false;
      utils::IteratorType iteratorKind =
          linalgOp.getIteratorTypesArray()[transformedDimension];
      if (iteratorKind == kind)
        continue;
      return false;
    }
    return true;
  });
  return *this;
}
transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::dim(AllDims tag, utils::IteratorType kind) {
  return dim(AllDimsExcept({}), kind);
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::dim(AllDimsExcept &&dims,
                                        utils::IteratorType kind) {
  addPredicate([dimensions = std::move(dims),
                kind](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "all dimensions except [";
               llvm::interleaveComma(dimensions.getExcluded(), llvm::dbgs());
               llvm::dbgs() << "] are " << utils::stringifyIteratorType(kind));
    int64_t rank = linalgOp.getNumLoops();
    llvm::SmallDenseSet<int64_t> excludedDims;
    for (int64_t dim : dimensions.getExcluded()) {
      excludedDims.insert(dim >= 0 ? dim : rank + dim);
    }

    for (auto [index, type] :
         llvm::enumerate(linalgOp.getIteratorTypesArray())) {
      if (excludedDims.contains(index))
        continue;
      if (type == kind)
        continue;
      return false;
    }
    return true;
  });
  return *this;
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::dim(int64_t dimension,
                                        DivisibleBy divisibleBy) {
  addPredicate([=](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "dimension " << dimension << " is divisible by "
                      << divisibleBy.value);
    int64_t rank = linalgOp.getNumLoops();
    int64_t transformedDimension =
        dimension >= 0 ? dimension : rank + dimension;
    if (transformedDimension >= rank)
      return false;

    int64_t size = linalgOp.getStaticLoopRanges()[transformedDimension];
    return !ShapedType::isDynamic(size) && (size % divisibleBy.value == 0);
  });
  return *this;
}

//===---------------------------------------------------------------------===//
// Capture directives.
//===---------------------------------------------------------------------===//
transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::rank(CaptureRank capture) {
  addPredicate([=](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "capture rank");
    capture.value = linalgOp.getNumLoops();
    return true;
  });
  return *this;
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::dim(int64_t dimension, CaptureDim capture) {
  addPredicate([=](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "capture dimension");
    int64_t rank = linalgOp.getNumLoops();
    int64_t transformedDimension =
        dimension >= 0 ? dimension : rank + dimension;
    if (transformedDimension >= rank)
      return false;

    capture.value = linalgOp.getStaticLoopRanges()[transformedDimension];
    return true;
  });
  return *this;
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::dim(AllDims tag, CaptureDims captures) {
  addPredicate([=](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "capture all dimensions");
    captures.value = linalgOp.getStaticLoopRanges();
    return true;
  });
  return *this;
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::convolutionDims(CaptureConvDims convDims) {
  addPredicate([=](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "capture convolution dimensions\n");
    StringRef convMessage = linalg::detail::getMatchConvolutionMessage(
        mlir::linalg::detail::isConvolutionInterfaceImpl(linalgOp,
                                                         &convDims.value));
    if (convMessage.empty())
      return true;
    LLVM_DEBUG(DBGS() << "capture convolution dimensions failed: "
                      << convMessage << "\n");
    return false;
  });
  return *this;
}

transform_ext::StructuredOpMatcher::StructuredOpMatcher(
    StructuredOpMatcher &A, StructuredOpMatcher &B) {

  addPredicate([&A, &B](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "start recursive lhs OR match {\n");
    {
      auto debugRAII = llvm::make_scope_exit(
          [] { LLVM_DEBUG(DBGS() << "} end recursive match"); });
      if (A.match(linalgOp))
        return true;
    }
    LLVM_DEBUG(DBGS() << "start recursive rhs OR match {\n");
    {
      auto debugRAII = llvm::make_scope_exit(
          [] { LLVM_DEBUG(DBGS() << "} end recursive match"); });
      if (B.match(linalgOp))
        return true;
    }
    return false;
  });
  recordNestedMatcher(A);
  recordNestedMatcher(B);
}

//===---------------------------------------------------------------------===//
// Constraints on input operands.
//===---------------------------------------------------------------------===//

void transform_ext::StructuredOpMatcher::addInputMatcher(
    int64_t position, std::function<bool(Operation *)> matcher,
    OptionalMatch optional) {
  addInputMatcher(
      position,
      // No need to handle optional inside the lambda, the wrapper will do that.
      [matcher = std::move(matcher)](Value value) {
        Operation *definingOp = value.getDefiningOp();
        return definingOp && matcher(definingOp);
      },
      optional);
}

void transform_ext::StructuredOpMatcher::addInputMatcher(
    int64_t position, std::function<bool(Value)> matcher,
    OptionalMatch optional) {
  addPredicate([position, optional, matcher = std::move(matcher)](
                   linalg::LinalgOp linalgOp) -> bool {
    int64_t transformedPosition =
        position >= 0 ? position : linalgOp.getNumDpsInputs() + position;
    if (transformedPosition >= linalgOp.getNumDpsInputs()) {
      LLVM_DEBUG(DBGS() << "input operand #" << position
                        << " does not exist but match required");
      return false;
    }

    LLVM_DEBUG(DBGS() << "input operand #" << position
                      << (optional.value ? " (optional match) " : " ")
                      << "is\n");

    // We MUST run the matcher at this point, even if the match is optional,
    // to allow for capture.
    LLVM_DEBUG(DBGS() << "start recursive match {\n");
    auto debugRAII = llvm::make_scope_exit(
        [] { LLVM_DEBUG(DBGS() << "} end recursive match"); });
    if (matcher(linalgOp.getDpsInputOperand(transformedPosition)->get()))
      return true;
    return optional.value;
  });
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::input(AllOperands tag, IsPermutation) {
  addPredicate([=](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "all input operands have permutation maps");
    // all_of with a lambda requires const-casting dance, so using a loop.
    for (OpOperand *operand : linalgOp.getDpsInputOperands()) {
      if (!linalgOp.getMatchingIndexingMap(operand).isPermutation())
        return false;
    }
    return true;
  });
  return *this;
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::input(AllOperands tag,
                                          IsProjectedPermutation) {
  addPredicate([=](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "all input operands have projected permutation maps");
    // all_of with a lambda requires const-casting dance, so using a loop.
    for (OpOperand *operand : linalgOp.getDpsInputOperands()) {
      if (!linalgOp.getMatchingIndexingMap(operand).isProjectedPermutation())
        return false;
    }
    return true;
  });
  return *this;
}

/// Helper to check if the map is an identity map with a projected dim.
static bool isProjectedMap(AffineMap map, int64_t projectedDim) {
  if (!map.isProjectedPermutation())
    return false;
  int64_t dimCounter = 0;
  for (unsigned i = 0, e = map.getNumResults(); i < e; i++) {
    // Skip the project dim.
    if (dimCounter == projectedDim)
      dimCounter++;
    if (map.getDimPosition(i) != dimCounter++) {
      return false;
    }
  }
  return true;
}

/// Helper to turn a potentially negative index to positive within the range
/// [0, ub) and indicate whether the transformed index is in bounds.
static bool makeValidPositiveIndex(int64_t &index, int64_t ub) {
  int64_t positiveIndex = index >= 0 ? index : ub + index;
  if (positiveIndex < 0 || ub < positiveIndex) {
    LLVM_DEBUG(DBGSNL() << "  index out of range");
    return false;
  }
  index = positiveIndex;
  return true;
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::input(SmallVector<int64_t> &&positions,
                                          IsProjected dim) {
  addPredicate([=](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "operands ";
               llvm::interleaveComma(positions, llvm::dbgs());
               llvm::dbgs() << " have a permutation maps with " << dim.value
                            << " projected");
    int64_t updatedDim = dim.value;
    if (!makeValidPositiveIndex(updatedDim, linalgOp.getNumLoops()))
      return false;
    for (int64_t position : positions) {
      int64_t updatedPosition = position;
      if (!makeValidPositiveIndex(updatedPosition, linalgOp.getNumDpsInputs()))
        return false;
      OpOperand *operand = linalgOp.getDpsInputOperand(updatedPosition);
      if (!isProjectedMap(linalgOp.getMatchingIndexingMap(operand), updatedDim))
        return false;
    }
    return true;
  });
  return *this;
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::input(AllOperands tag, IsIdentity) {
  addPredicate([=](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "all input operands have identity maps");
    // all_of with a lambda requires const-casting dance, so using a loop.
    for (OpOperand *operand : linalgOp.getDpsInputOperands()) {
      if (!linalgOp.getMatchingIndexingMap(operand).isIdentity())
        return false;
    }
    return true;
  });
  return *this;
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::input(SmallVector<int64_t> &&positions,
                                          IsIdentity) {
  addPredicate([=](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "input operands ";
               llvm::interleaveComma(positions, llvm::dbgs());
               llvm::dbgs() << " have identity maps");
    // all_of with a lambda requires const-casting dance, so using a loop.
    for (int64_t position : positions) {
      int64_t updatedPosition = position;
      if (!makeValidPositiveIndex(updatedPosition, linalgOp.getNumDpsInputs()))
        return false;
      OpOperand *operand = linalgOp.getDpsInputOperand(updatedPosition);
      if (!linalgOp.getMatchingIndexingMap(operand).isIdentity())
        return false;
    }
    return true;
  });
  return *this;
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::input(int64_t position,
                                          ElementTypeBitWidth width) {
  addPredicate([=](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "input operand #" << position
                      << " has elemental type with bit width " << width.value);
    int64_t updatedPosition = position;
    if (!makeValidPositiveIndex(updatedPosition, linalgOp.getNumDpsInputs()))
      return false;
    auto shapedType = linalgOp.getDpsInputOperand(updatedPosition)
                          ->get()
                          .getType()
                          .dyn_cast<ShapedType>();
    return shapedType && shapedType.getElementType().isIntOrFloat() &&
           shapedType.getElementType().getIntOrFloatBitWidth() == width.value;
  });
  return *this;
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::input(int64_t position,
                                          CaptureElementTypeBitWidth width) {
  addPredicate([=](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "input operand #" << position << " capture bitwidth");
    int64_t updatedPosition = position;
    if (!makeValidPositiveIndex(updatedPosition, linalgOp.getNumDpsInputs()))
      return false;
    auto shapedType = linalgOp.getDpsInputOperand(updatedPosition)
                          ->get()
                          .getType()
                          .dyn_cast<ShapedType>();
    if (!shapedType || !shapedType.getElementType().isIntOrFloat())
      return false;
    width.value = shapedType.getElementType().getIntOrFloatBitWidth();
    return true;
  });
  return *this;
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::input(int64_t position,
                                          CaptureElementType elem) {
  addPredicate([=](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "input operand #" << position
                      << " capture element type");
    int64_t updatedPosition = position;
    if (!makeValidPositiveIndex(updatedPosition, linalgOp.getNumDpsInputs()))
      return false;
    auto shapedType = linalgOp.getDpsInputOperand(updatedPosition)
                          ->get()
                          .getType()
                          .dyn_cast<ShapedType>();
    if (!shapedType) {
      LLVM_DEBUG(DBGSNL() << "  not a shaped type");
      return false;
    }
    elem.value = shapedType.getElementType();
    return true;
  });
  return *this;
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::input(NumEqualsTo num) {
  addPredicate([=](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "number of input operands == " << num.value);
    return linalgOp.getNumDpsInputs() == num.value;
  });
  return *this;
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::input(int64_t position,
                                          ConstantFloatMinOrMinusInf) {
  return input(position, [](llvm::APFloat f) {
    return (f.isLargest() || f.isInfinity()) && f.isNegative();
  });
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::input(int64_t position, ConstantFloatZero) {
  return input(position, [](llvm::APFloat f) { return f.isZero(); });
}

transform_ext::StructuredOpMatcher &transform_ext::StructuredOpMatcher::input(
    int64_t position, std::function<bool(llvm::APFloat)> floatValueFn) {
  addPredicate([=](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "input operand #" << position
                      << " is a special floating point constant");
    int64_t updatedPosition = position;
    if (!makeValidPositiveIndex(updatedPosition, linalgOp.getNumDpsInputs()))
      return false;
    auto cstOp = linalgOp.getDpsInputOperand(updatedPosition)
                     ->get()
                     .getDefiningOp<arith::ConstantFloatOp>();
    if (!cstOp)
      return false;
    return floatValueFn(cstOp.value());
  });
  return *this;
}

//===---------------------------------------------------------------------===//
// Constraints on output operands.
//===---------------------------------------------------------------------===//

void transform_ext::StructuredOpMatcher::addOutputMatcher(
    int64_t position, std::function<bool(Operation *)> matcher,
    OptionalMatch optional) {
  addPredicate([position, optional, matcher = std::move(matcher)](
                   linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "output operand #" << position
                      << (optional.value ? " (optional match) "
                                         : " (mandatory match) ")
                      << "is produced by\n");
    int64_t updatedPosition = position;
    if (!makeValidPositiveIndex(updatedPosition, linalgOp.getNumDpsInits()))
      return false;
    Operation *definingOp =
        linalgOp.getDpsInitOperand(updatedPosition)->get().getDefiningOp();
    if (!definingOp)
      return optional.value;
    // We MUST run the matcher at this point, even if the match is optional,
    // to allow for capture.
    LLVM_DEBUG(DBGS() << "start recursive match {\n");
    auto debugRAII = llvm::make_scope_exit(
        [] { LLVM_DEBUG(DBGS() << "} end recursive match"); });
    if (matcher(definingOp))
      return true;
    return optional.value;
  });
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::output(AllOperands tag, IsPermutation) {
  addPredicate([=](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "all output operands have permutation maps");
    for (OpOperand *operand : linalgOp.getDpsInitOperands()) {
      if (!linalgOp.getMatchingIndexingMap(operand).isPermutation())
        return false;
    }
    return true;
  });
  return *this;
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::output(AllOperands tag,
                                           IsProjectedPermutation) {
  addPredicate([=](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "all output operands have projected permutation maps");
    for (OpOperand *operand : linalgOp.getDpsInitOperands()) {
      if (!linalgOp.getMatchingIndexingMap(operand).isProjectedPermutation())
        return false;
    }
    return true;
  });
  return *this;
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::output(AllOperands tag, IsProjected dim) {
  addPredicate([=](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "all output operands have a maps with projected");
    int64_t updatedDim = dim.value;
    if (!makeValidPositiveIndex(updatedDim, linalgOp.getNumLoops()))
      return false;
    // all_of with a lambda requires const-casting dance, so using a loop.
    for (OpOperand *operand : linalgOp.getDpsInitOperands()) {
      if (!isProjectedMap(linalgOp.getMatchingIndexingMap(operand), updatedDim))
        return false;
    }
    return true;
  });
  return *this;
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::output(AllOperands tag, IsIdentity) {
  addPredicate([=](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "all output operands have identity permutation maps");
    for (OpOperand *operand : linalgOp.getDpsInitOperands()) {
      if (!linalgOp.getMatchingIndexingMap(operand).isIdentity())
        return false;
    }
    return true;
  });
  return *this;
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::output(int64_t position,
                                           ElementTypeBitWidth width) {
  addPredicate([=](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "output operand #" << position
                      << " has elemental type with bit width " << width.value);
    int64_t updatedPosition = position;
    if (!makeValidPositiveIndex(updatedPosition, linalgOp.getNumDpsInits()))
      return false;
    auto shapedType = linalgOp.getDpsInitOperand(updatedPosition)
                          ->get()
                          .getType()
                          .dyn_cast<ShapedType>();
    return shapedType && shapedType.getElementType().isIntOrFloat() &&
           shapedType.getElementType().getIntOrFloatBitWidth() == width.value;
  });
  return *this;
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::output(int64_t position,
                                           CaptureElementTypeBitWidth width) {
  addPredicate([=](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "output operand #" << position << " capture bitwidth");
    int64_t updatedPosition = position;
    if (!makeValidPositiveIndex(updatedPosition, linalgOp.getNumDpsInits()))
      return false;
    auto shapedType = linalgOp.getDpsInitOperand(updatedPosition)
                          ->get()
                          .getType()
                          .dyn_cast<ShapedType>();
    if (!shapedType || !shapedType.getElementType().isIntOrFloat()) {
      LLVM_DEBUG(DBGSNL() << "  could not infer element type");
      return false;
    }
    width.value = shapedType.getElementType().getIntOrFloatBitWidth();
    return true;
  });
  return *this;
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::output(int64_t position,
                                           CaptureElementType elem) {
  addPredicate([=](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "output operand #" << position
                      << " capture element type");
    int64_t updatedPosition = position;
    if (!makeValidPositiveIndex(updatedPosition, linalgOp.getNumDpsInits()))
      return false;
    auto shapedType = linalgOp.getDpsInitOperand(updatedPosition)
                          ->get()
                          .getType()
                          .dyn_cast<ShapedType>();
    if (!shapedType) {
      LLVM_DEBUG(DBGSNL() << "  not a shaped type");
      return false;
    }
    elem.value = shapedType.getElementType();
    return true;
  });
  return *this;
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::output(int64_t position,
                                           SingleCombinerReduction tag) {
  addPredicate([=](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "output operand #" << position
                      << " is populated by a single-combiner reduction");
    int64_t updatedPosition = position;
    if (!makeValidPositiveIndex(updatedPosition, linalgOp.getNumDpsInits()))
      return false;
    SmallVector<Operation *> combinerOps;
    return matchReduction(linalgOp.getRegionOutputArgs(), updatedPosition,
                          combinerOps) &&
           llvm::hasSingleElement(combinerOps);
  });
  return *this;
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::output(NumEqualsTo num) {
  addPredicate([=](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "number of output operands == " << num.value);
    return linalgOp.getNumDpsInits() == num.value;
  });
  return *this;
}

//===---------------------------------------------------------------------===//
// Constraints on results.
//===---------------------------------------------------------------------===//

void transform_ext::StructuredOpMatcher::addResultMatcher(
    int64_t position, HasAnyUse tag, std::function<bool(Operation *)> matcher,
    OptionalMatch optional) {
  addPredicate([matcher = std::move(matcher), optional,
                position](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "result #" << position
                      << (optional.value ? " (optional match) "
                                         : " (mandatory match) ")
                      << "has a use\n");
    int64_t updatedPosition = position;
    if (!makeValidPositiveIndex(updatedPosition, linalgOp->getNumResults()))
      return false;

    // We MUST run the matcher at this point, even if the match is optional,
    // to allow for capture.
    LLVM_DEBUG(DBGS() << "start recursive match {\n");
    auto debugRAII = llvm::make_scope_exit(
        [] { LLVM_DEBUG(DBGS() << "} end recursive match"); });
    if (llvm::any_of(linalgOp->getResult(updatedPosition).getUsers(),
                     [&matcher](Operation *op) { return matcher(op); })) {
      return true;
    }
    return optional.value;
  });
}

//===-------------------------------------------------------------------===//
// Constraints on op region.
//===-------------------------------------------------------------------===//

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::singleOpWithCanonicaleArgs(
    StringRef opcode, bool commutative) {
  addPredicate([=](linalg::LinalgOp linalgOp) {
    if (linalgOp.getBlock()->getOperations().size() != 2)
      return false;
    Operation *innerOp = &(*linalgOp.getBlock()->getOperations().begin());
    if (innerOp->getName().getStringRef() != opcode ||
        innerOp->getNumResults() != 1)
      return false;
    Operation *yieldOp = linalgOp.getBlock()->getTerminator();
    if (yieldOp->getNumOperands() != 1)
      return false;
    if (yieldOp->getOperand(0).getDefiningOp() != innerOp)
      return false;
    if (commutative && innerOp->getNumOperands() == 2) {
      auto arg0 = dyn_cast<BlockArgument>(innerOp->getOperand(0));
      auto arg1 = dyn_cast<BlockArgument>(innerOp->getOperand(1));
      if (!arg0 || !arg1)
        return false;
      if (arg0.getParentBlock() != linalgOp.getBlock() ||
          arg1.getParentBlock() != linalgOp.getBlock())
        return false;
      if (!((arg0.getArgNumber() == 0 && arg1.getArgNumber() == 1) ||
            (arg1.getArgNumber() == 0 && arg0.getArgNumber() == 1)))
        return false;
    } else {
      for (auto [index, operand] : llvm::enumerate(innerOp->getOperands())) {
        auto arg = dyn_cast<BlockArgument>(operand);
        if (!arg || arg.getParentBlock() != linalgOp.getBlock() ||
            arg.getArgNumber() != index)
          return false;
      }
    }
    return true;
  });
  return *this;
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::isFloatReciprocal() {
  addPredicate([=](linalg::LinalgOp linalgOp) {
    LLVM_DEBUG(DBGS() << "op region represents a reciprocal operation");
    if (linalgOp.getBlock()->getOperations().size() != 2)
      return false;
    Operation *innerOp = &(*linalgOp.getBlock()->getOperations().begin());
    if (!isa<arith::DivFOp>(innerOp) || innerOp->getNumResults() != 1)
      return false;
    Operation *yieldOp = linalgOp.getBlock()->getTerminator();
    if (yieldOp->getNumOperands() != 1)
      return false;
    if (yieldOp->getOperand(0).getDefiningOp() != innerOp)
      return false;
    auto cst = innerOp->getOperand(0).getDefiningOp<arith::ConstantFloatOp>();
    if (!cst || cst.value().convertToDouble() != 1.0)
      return false;
    auto arg = dyn_cast<BlockArgument>(innerOp->getOperand(1));
    if (!arg || arg.getParentBlock() != linalgOp.getBlock() ||
        arg.getArgNumber() != 0)
      return false;
    return true;
  });
  return *this;
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::passThroughOp() {
  addPredicate([=](linalg::LinalgOp linalgOp) {
    if (linalgOp.getBlock()->getOperations().size() != 1)
      return false;
    Operation *yieldOp = linalgOp.getBlock()->getTerminator();
    for (auto [index, operand] : llvm::enumerate(yieldOp->getOperands())) {
      auto arg = dyn_cast<BlockArgument>(operand);
      if (!arg || arg.getParentBlock() != linalgOp.getBlock() ||
          arg.getArgNumber() != index)
        return false;
    }
    return true;
  });
  return *this;
}

void transform_ext::detail::debugOutputForConcreteOpMatcherConstructor(
    StringRef name) {
  LLVM_DEBUG(DBGS() << "op is a " << name << "'");
}

//===---------------------------------------------------------------------===//
// TensorPadOpMatcher
//===---------------------------------------------------------------------===//

transform_ext::TensorPadOpMatcher &
transform_ext::TensorPadOpMatcher::low(ArrayRef<int64_t> sizes) {
  addPredicate([=](tensor::PadOp tensorPad) {
    LLVM_DEBUG({
      DBGS() << "low pad sizes are ";
      llvm::interleaveComma(sizes, llvm::dbgs());
    });
    return tensorPad.getStaticLow() == sizes;
  });
  return *this;
}

transform_ext::TensorPadOpMatcher &
transform_ext::TensorPadOpMatcher::low(AllDims tag, int64_t size) {
  addPredicate([=](tensor::PadOp tensorPad) {
    LLVM_DEBUG(DBGS() << "all low pad sizes are " << size);
    return llvm::all_of(tensorPad.getStaticLow(),
                        [&](int64_t v) { return v == size; });
  });
  return *this;
}

transform_ext::TensorPadOpMatcher &
transform_ext::TensorPadOpMatcher::high(ArrayRef<int64_t> sizes) {
  addPredicate([=](tensor::PadOp tensorPad) {
    LLVM_DEBUG({
      DBGS() << "high pad sizes are ";
      llvm::interleaveComma(sizes, llvm::dbgs());
    });
    return tensorPad.getStaticHigh() == sizes;
  });
  return *this;
}

transform_ext::TensorPadOpMatcher &
transform_ext::TensorPadOpMatcher::high(AllDims tag, int64_t size) {
  addPredicate([=](tensor::PadOp tensorPad) {
    LLVM_DEBUG(DBGS() << "all high pad sizes are " << size);
    return llvm::all_of(tensorPad.getStaticHigh(),
                        [&](int64_t v) { return v == size; });
  });
  return *this;
}

transform_ext::TensorPadOpMatcher &
transform_ext::TensorPadOpMatcher::yieldsExternalValue() {
  addPredicate([=](tensor::PadOp tensorPad) {
    LLVM_DEBUG(DBGS() << "pad body yields an externally-defined value");
    Block *body = tensorPad.getBody();
    if (!llvm::hasSingleElement(*body))
      return false;
    return llvm::all_of(body->getTerminator()->getOperands(),
                        [body](Value operand) {
                          auto arg = dyn_cast<BlockArgument>(operand);
                          return !arg || arg.getOwner() != body;
                        });
  });
  return *this;
}

//===---------------------------------------------------------------------===//
// MatchCallbackResult.
//===---------------------------------------------------------------------===//

ArrayRef<Operation *>
transform_ext::MatchCallbackResult::getPayloadGroup(int64_t position) const {
  assert(position < payloadGroupLengths.size());
  int64_t start = 0;
  for (int64_t i = 0; i < position; ++i) {
    start += payloadGroupLengths[i];
  }
  return llvm::ArrayRef(payloadOperations)
      .slice(start, payloadGroupLengths[position]);
}

//===---------------------------------------------------------------------===//
// Case-specific matcher builders.
//===---------------------------------------------------------------------===//

static constexpr int64_t kCudaWarpSize = 32;

void transform_ext::makeReductionMatcher(
    transform_ext::MatcherContext &matcherContext,
    transform_ext::StructuredOpMatcher *&reductionCapture,
    transform_ext::StructuredOpMatcher *&fillCapture,
    transform_ext::StructuredOpMatcher *&leadingCapture,
    transform_ext::StructuredOpMatcher *&trailingCapture,
    MatchedReductionCaptures &captures, bool mustMatchEntireFunc) {
  // The core part of the matcher is anchored on a particular reduction op.
  auto &reduction =
      m_StructuredOp(matcherContext)
          // Op has at least a parallel a reduction dimension and at
          // most 3 parallel dimensions.
          // TODO: relax once we have global collapse/expand_shape.
          //
          .rank(NumGreaterEqualTo(2))
          .rank(NumLowerEqualTo(4))
          .rank(CaptureRank(captures.reductionRank))
          // Op has a single most-minor reduction.
          .dim(-1, utils::IteratorType::reduction)
          // Capture op sizes.
          .dim(AllDims(), CaptureDims(captures.reductionOpSizes))
          // All other dimensions are parallel.
          .dim(AllDimsExcept({-1}), utils::IteratorType::parallel)
          // Single input for now, can be arbitrary projected permutations.
          // TODO: Multiple inputs, can be arbitrary projected permutations.
          // TODO: Watch out for multiple inputs though as a reduction turns
          //       into a contraction when mixed with projected
          //       permutations. A reduction is often bandwidth bound but
          //       contraction is a different beast that is compute bound
          //       and has a very different schedule.
          //
          .input(NumEqualsTo(1))
          .input(AllOperands(), IsProjectedPermutation())
          // Single output supported atm.
          // TODO: Multiple outputs.
          //
          .output(NumEqualsTo(1))
          // A reduction output must be a projected permutation, match it but we
          // could also drop this technically.
          .output(AllOperands(), IsProjectedPermutation())
          // Only single combiner for now due to reduction warp
          // distribution.
          // TODO: relax this once reduction distribution is more powerful.
          //
          .output(0, CaptureElementTypeBitWidth(
                         captures.reductionOutputElementalTypeBitWidth))
          .output(0, SingleCombinerReduction());
  reductionCapture = &reduction;

  // Mandatory FillOp must create the unique output of the reduction.
  // TODO: Relax this, as any map, broadcast, transpose should also work.
  //
  auto &fill = m_StructuredOp<linalg::FillOp>(matcherContext);
  reduction = reduction.output(NumEqualsTo(1)).output(0, fill);
  fillCapture = &fill;

  // Optional leading or trailing op can be any map, transpose, broadcast but
  // not reduce or windowing operation for now.
  // It must create the unique input for the reduction.
  // TODO: match more optional leading ops, one per input of the reduction.
  // TODO: careful about multi-output and turning into a contraction.
  //
  transform_ext::StructuredOpMatcher commonLeadingOrTrailing =
      m_StructuredOp<linalg::GenericOp>(matcherContext)
          // All parallel dimensions.
          .dim(AllDims(), utils::IteratorType::parallel)
          // All inputs are any projected permutation.
          .input(AllOperands(), IsProjectedPermutation())
          .output(AllOperands(), IsPermutation())
          // leading and trailing may have 0, 1 or more input as long as they do
          // not come from unmatched ops. This extra constraint is taken care of
          // separately. This is also a noop but we document it.
          // TODO: Base and derived classes, atm this does not compile.
          // .input(NumGreaterEqualTo(0))
          // Single output supported atm.
          // TODO: extend this.
          //
          .output(NumEqualsTo(1));
  // TODO: match more optional leading ops, one per input of the reduction.
  // TODO: careful about multi-output and turning into a contraction.
  //
  auto &leading =
      m_StructuredOp(matcherContext, commonLeadingOrTrailing)
          .rank(CaptureRank(captures.maybeLeadingRank))
          // Capture op sizes.
          .dim(AllDims(), CaptureDims(captures.leadingOpSizes))
          // Capture output elemental type.
          .output(0, CaptureElementTypeBitWidth(
                         captures.maybeLeadingOutputElementalTypeBitWidth));
  reduction = reduction.input(0, leading, OptionalMatch());
  leadingCapture = &leading;

  // Optional trailing can be any map, transpose, broadcast but not reduce or
  // windowing operation for now.
  // It must be fed by the unique input for the reduction.
  // TODO: match more optional leading ops, one per input of the reduction.
  // TODO: careful about multi-output and turning into a contraction.
  //
  auto &trailing =
      m_StructuredOp(matcherContext, commonLeadingOrTrailing)
          .rank(CaptureRank(captures.maybeTrailingRank))
          // Capture op sizes.
          .dim(AllDims(), CaptureDims(captures.trailingOpSizes))
          // Capture output elemental type.
          .output(0, CaptureElementTypeBitWidth(
                         captures.maybeTrailingOutputElementalTypeBitWidth));
  reduction = reduction.result(0, HasAnyUse(), trailing, OptionalMatch());
  if (mustMatchEntireFunc)
    reduction = reduction.allTilableOpsCaptured<func::FuncOp>();
  trailingCapture = &trailing;
}

void transform_ext::makeReductionMatcher(transform_ext::MatcherContext &context,
                                         StructuredOpMatcher *&reductionCapture,
                                         MatchedReductionCaptures &captures,
                                         bool mustMatchEntireFunc) {
  StructuredOpMatcher *fill;
  StructuredOpMatcher *leading;
  StructuredOpMatcher *trailing;
  makeReductionMatcher(context, reductionCapture, fill, leading, trailing,
                       captures, mustMatchEntireFunc);
}

void transform_ext::makeMatmulMatcher(
    transform_ext::MatcherContext &matcherContext,
    transform_ext::StructuredOpMatcher *&matmulCapture,
    transform_ext::StructuredOpMatcher *&fillCapture,
    transform_ext::StructuredOpMatcher *&trailingCapture,
    transform_ext::MatchedMatmulCaptures &captures, bool mustMatchEntireFunc) {
  auto &matmul = transform_ext::m_StructuredOp<linalg::MatmulOp>(matcherContext)
                     // Capture op sizes.
                     .dim(AllDims(), CaptureDims(captures.matmulOpSizes))
                     // Capture input/output element types.
                     .input(0, CaptureElementType(captures.lhsElementType))
                     .input(1, CaptureElementType(captures.rhsElementType))
                     .output(0, CaptureElementType(captures.outputElementType));
  matmulCapture = &matmul;
  // Mandatory FillOp must create the unique output of the reduction.
  auto &fill = transform_ext::m_StructuredOp<linalg::FillOp>(matcherContext);
  matmul = matmul.output(transform_ext::NumEqualsTo(1)).output(0, fill);
  fillCapture = &fill;

  auto &trailing = m_StructuredOp<linalg::GenericOp>(matcherContext);
  matmul = matmul.result(0, HasAnyUse(), trailing, OptionalMatch());
  if (mustMatchEntireFunc)
    matmul = matmul.allTilableOpsCaptured<func::FuncOp>();
  trailingCapture = &trailing;
}

/// Match sum(%src, broadcast(%reduction))
static void
matchSubBroadcast(transform_ext::MatcherContext &matcherContext,
                  transform_ext::StructuredOpMatcher &maxReduction,
                  transform_ext::CapturingValueMatcher &softmaxSourceOperand,
                  transform_ext::StructuredOpMatcher *&sub) {
  using namespace transform_ext;

  auto &broadcast =
      transform_ext::m_StructuredOp<linalg::GenericOp>(matcherContext)
          .passThroughOp()
          .dim(AllDims(), utils::IteratorType::parallel)
          .input(NumEqualsTo(1))
          .input(0, IsProjected(-1))
          .output(NumEqualsTo(1))
          .output(AllOperands(), IsIdentity());
  broadcast = broadcast.input(0, maxReduction);

  auto &subParallel =
      transform_ext::m_StructuredOp<linalg::GenericOp>(matcherContext)
          .singleOpWithCanonicaleArgs<arith::SubFOp>()
          .dim(AllDims(), utils::IteratorType::parallel)
          .input(NumEqualsTo(2))
          .input(0, IsIdentity())
          .input(1, IsIdentity())
          .output(NumEqualsTo(1))
          .output(AllOperands(), IsIdentity());
  subParallel = subParallel.input(0, softmaxSourceOperand);
  subParallel = subParallel.input(1, broadcast);

  auto &subBroadcast =
      transform_ext::m_StructuredOp<linalg::GenericOp>(matcherContext)
          .singleOpWithCanonicaleArgs<arith::SubFOp>()
          .dim(AllDims(), utils::IteratorType::parallel)
          .input(NumEqualsTo(2))
          .input(0, IsIdentity())
          .input(1, IsProjected(-1))
          .output(NumEqualsTo(1))
          .output(AllOperands(), IsIdentity());
  subBroadcast = subBroadcast.input(0, softmaxSourceOperand);
  subBroadcast = subBroadcast.input(1, maxReduction);
  auto &subOr = transform_ext::m_StructuredOp_Or(matcherContext, subBroadcast,
                                                 subParallel);
  sub = &subOr;
}

/// Match sum(%exp, broadcast(%sum))
static void matchdivBroadcast(transform_ext::MatcherContext &matcherContext,
                              transform_ext::StructuredOpMatcher &expOperand,
                              transform_ext::StructuredOpMatcher &sum,
                              transform_ext::StructuredOpMatcher *&div) {
  using namespace transform_ext;

  auto &broadcast =
      transform_ext::m_StructuredOp<linalg::GenericOp>(matcherContext)
          .passThroughOp()
          .dim(AllDims(), utils::IteratorType::parallel)
          .input(NumEqualsTo(1))
          .input(0, IsProjected(-1))
          .output(NumEqualsTo(1))
          .output(AllOperands(), IsIdentity());
  broadcast = broadcast.input(0, sum);

  auto &divNoBroadcast =
      transform_ext::m_StructuredOp<linalg::GenericOp>(matcherContext)
          .singleOpWithCanonicaleArgs<arith::DivFOp>()
          .dim(AllDims(), utils::IteratorType::parallel)
          .input(NumEqualsTo(2))
          .input(0, IsIdentity())
          .input(1, IsIdentity())
          .output(NumEqualsTo(1))
          .output(AllOperands(), IsIdentity());

  divNoBroadcast = divNoBroadcast.input(0, expOperand);
  divNoBroadcast = divNoBroadcast.input(1, broadcast);

  auto &divBroadcast =
      transform_ext::m_StructuredOp<linalg::GenericOp>(matcherContext)
          .singleOpWithCanonicaleArgs<arith::DivFOp>()
          .dim(AllDims(), utils::IteratorType::parallel)
          .input(NumEqualsTo(2))
          .input(0, IsIdentity())
          .input(1, IsProjected(-1))
          .output(NumEqualsTo(1))
          .output(AllOperands(), IsIdentity());

  divBroadcast = divBroadcast.input(0, expOperand);
  divBroadcast = divBroadcast.input(1, sum);

  auto &divMerge = transform_ext::m_StructuredOp_Or(
      matcherContext, divNoBroadcast, divBroadcast);
  div = &divMerge;
}

void transform_ext::makeSoftmaxMatcher(
    transform_ext::MatcherContext &matcherContext,
    transform_ext::StructuredOpMatcher *&maxReductionCapture,
    transform_ext::StructuredOpMatcher *&softmaxRootCapture) {
  auto &softmaxSourceOperand = m_Value(matcherContext);

  auto &fillMinusInf = m_StructuredOp<linalg::FillOp>(matcherContext)
                           .input(0, ConstantFloatMinOrMinusInf());
  auto &maxReduction =
      transform_ext::m_StructuredOp<linalg::GenericOp>(matcherContext)
          .singleOpWithCanonicaleArgs<arith::MaxFOp>(/*commutative=*/true)
          // Only handle most inner reduction for now.
          .dim(-1, utils::IteratorType::reduction)
          .dim(AllDimsExcept({-1}), utils::IteratorType::parallel)
          .input(NumEqualsTo(1))
          .input(AllOperands(), IsIdentity())
          .output(NumEqualsTo(1))
          .output(AllOperands(), IsProjected(-1));
  maxReduction = maxReduction.input(0, softmaxSourceOperand);
  maxReduction = maxReduction.output(0, fillMinusInf);
  maxReductionCapture = &maxReduction;

  transform_ext::StructuredOpMatcher *subOperand;
  matchSubBroadcast(matcherContext, maxReduction, softmaxSourceOperand,
                    subOperand);

  auto &expOperand = m_StructuredOp<linalg::GenericOp>(matcherContext)
                         .singleOpWithCanonicaleArgs<math::ExpOp>()
                         .dim(AllDims(), utils::IteratorType::parallel)
                         .input(NumEqualsTo(1))
                         .input(AllOperands(), IsIdentity())
                         .output(AllOperands(), IsIdentity())
                         .output(NumEqualsTo(1));
  expOperand = expOperand.input(0, *subOperand);

  auto &fillZero = m_StructuredOp<linalg::FillOp>(matcherContext)
                       .input(0, ConstantFloatZero());
  auto &sum =
      m_StructuredOp<linalg::GenericOp>(matcherContext)
          .singleOpWithCanonicaleArgs<arith::AddFOp>(/*commutative=*/true)
          // Only handle most inner reduction for now.
          .dim(-1, utils::IteratorType::reduction)
          .dim(AllDimsExcept({-1}), utils::IteratorType::parallel)
          .input(NumEqualsTo(1))
          .input(AllOperands(), IsIdentity())
          .output(AllOperands(), IsProjected(-1))
          .output(NumEqualsTo(1));
  sum = sum.input(0, expOperand);
  sum = sum.output(0, fillZero);

  auto &rcpOperand = m_StructuredOp<linalg::GenericOp>(matcherContext)
                         .isFloatReciprocal()
                         .dim(AllDims(), utils::IteratorType::parallel)
                         .input(NumEqualsTo(1))
                         .input(AllOperands(), IsIdentity())
                         .output(AllOperands(), IsIdentity())
                         .output(NumEqualsTo(1));
  rcpOperand = rcpOperand.input(0, sum);

  auto &mulOperand =
      transform_ext::m_StructuredOp<linalg::GenericOp>(matcherContext)
          .singleOpWithCanonicaleArgs<arith::MulFOp>(/*commutative=*/true)
          .dim(AllDims(), utils::IteratorType::parallel)
          .input(NumEqualsTo(2))
          .input(0, IsIdentity())
          .input(1, IsProjected(-1))
          .output(NumEqualsTo(1))
          .output(AllOperands(), IsIdentity());

  mulOperand = mulOperand.input(0, expOperand);
  mulOperand = mulOperand.input(1, rcpOperand);

  transform_ext::StructuredOpMatcher *divOperand;
  matchdivBroadcast(matcherContext, expOperand, sum, divOperand);

  auto &softmaxRoot =
      transform_ext::m_StructuredOp_Or(matcherContext, mulOperand, *divOperand);
  softmaxRootCapture = &softmaxRoot;
}

/// Matcher for convolutions.
void transform_ext::makeConvolutionMatcher(
    transform_ext::MatcherContext &matcherContext,
    transform_ext::StructuredOpMatcher *&convolutionCapture,
    transform_ext::StructuredOpMatcher *&fillCapture,
    transform_ext::StructuredOpMatcher *&trailingCapture,
    MatchedConvolutionCaptures &captures, bool mustMatchEntireFunc) {
  // The core part of the matcher is anchored on a particular convolution op.
  auto &convolution =
      m_StructuredOp<linalg::Conv2DNchwFchwOp, linalg::Conv2DNhwcHwcfOp>(
          matcherContext)
          // Capture convolution dim classifications.
          .convolutionDims(CaptureConvDims(captures.convolutionDims))
          // Capture op sizes.
          .dim(AllDims(), CaptureDims(captures.convolutionOpSizes))
          // Capture convolution element type.
          .output(0, CaptureElementTypeBitWidth(
                         captures.convolutionOutputElementalTypeBitWidth));
  convolutionCapture = &convolution;

  // Optional FillOp to create the unique output of the convolution.
  auto &fill = m_StructuredOp<linalg::FillOp>(matcherContext)
                   .output(0, CaptureElementTypeBitWidth(
                                  captures.maybeFillElementalTypeBitWidth));
  convolution =
      convolution.output(NumEqualsTo(1)).output(0, fill, OptionalMatch());
  fillCapture = &fill;

  // Optional trailing op can be any map, transpose, broadcast but
  // not reduce or windowing operation for now.
  // It must create the unique input for the reduction.
  auto &trailing =
      m_StructuredOp<linalg::GenericOp>(matcherContext)
          // All parallel dimensions.
          .dim(AllDims(), utils::IteratorType::parallel)
          // All inputs are any projected permutation.
          .input(AllOperands(), IsProjectedPermutation())
          .output(AllOperands(), IsPermutation())
          .output(NumEqualsTo(1))
          .dim(AllDims(), CaptureDims(captures.trailingOpSizes))
          // Capture output elemental type.
          .output(0, CaptureElementTypeBitWidth(
                         captures.maybeTrailingOutputElementalTypeBitWidth));

  // Optional trailing can be any map, transpose, broadcast but not reduce or
  // windowing operation for now.
  convolution = convolution.result(0, HasAnyUse(), trailing, OptionalMatch());
  if (mustMatchEntireFunc)
    convolution = convolution.allTilableOpsCaptured<func::FuncOp>();
  trailingCapture = &trailing;
}

void transform_ext::makeConvolutionMatcher(
    transform_ext::MatcherContext &context,
    StructuredOpMatcher *&convolutionCapture,
    MatchedConvolutionCaptures &captures, bool mustMatchEntireFunc) {
  StructuredOpMatcher *fill;
  StructuredOpMatcher *trailing;
  makeConvolutionMatcher(context, convolutionCapture, fill, trailing, captures,
                         mustMatchEntireFunc);
}

void transform_ext::makePadMatcher(MatcherContext &context,
                                   CapturingOpMatcher *&padCapture,
                                   MatchedPadCaptures &captures,
                                   bool mustMatchEntireFunc) {
  auto &value = transform_ext::m_ShapedValue(context);
  value.rank(transform_ext::CaptureRank(captures.rank))
      .dim(transform_ext::AllDims(), transform_ext::CaptureDims(captures.dims))
      .elementType(CaptureElementType(captures.elementType));
  auto &opMatcher = transform_ext::m_tensorPad(context)
                        .low(AllDims(), 0)
                        .yieldsExternalValue()
                        .result(0, value);
  if (mustMatchEntireFunc)
    opMatcher = opMatcher.allTilableOpsCaptured<func::FuncOp>();
  padCapture = &opMatcher;
}
