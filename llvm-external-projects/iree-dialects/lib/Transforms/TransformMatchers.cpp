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

using namespace mlir;

#define DEBUG_TYPE "transform-matchers"
#define DBGS() llvm::dbgs() << "[" DEBUG_TYPE "] "

//===---------------------------------------------------------------------===//
// CapturingMatcherBase
//===---------------------------------------------------------------------===//

void transform_ext::CapturingMatcherBase::getAllNested(
    SmallVectorImpl<CapturingMatcherBase *> &nested) {
  SetVector<CapturingMatcherBase *> found;
  found.insert(nested.begin(), nested.end());
  for (int64_t position = 0; position < found.size(); ++position) {
    found.insert(found[position]->nested.begin(),
                 found[position]->nested.end());
  }

  llvm::append_range(nested, found.takeVector());
}

void transform_ext::CapturingMatcherBase::resetCapture() {
  SmallVector<CapturingMatcherBase *> nested;
  getAllNested(nested);
  for (CapturingMatcherBase *n : nested)
    n->resetThisCapture();
}

//===---------------------------------------------------------------------===//
// CapturingOpMatcher
//===---------------------------------------------------------------------===//

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
transform_ext::CapturingOpMatcher::operand(int64_t pos, ValueMatcher &nested) {
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

transform_ext::CapturingOpMatcher &transform_ext::CapturingOpMatcher::operand(
    transform_ext::AllOperands tag,
    transform_ext::CapturingValuePackMatcher &nested) {
  addPredicate([&nested](Operation *op) {
    LLVM_DEBUG(DBGS() << "all operands are");
    ValueRange operands = op->getOperands();
    return recursiveMatch(nested, operands);
  });
  recordNestedMatcher(nested);
  return *this;
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
transform_ext::CapturingOpMatcher::result(int64_t pos, ValueMatcher &nested) {
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
// BlockBodyMatcher
//===---------------------------------------------------------------------===//

transform_ext::BlockBodyMatcher::BlockBodyMatcher(
    transform_ext::BlockBodyMatcher &first,
    transform_ext::BlockBodyMatcher &second) {
  addPredicate([&first, &second](Block &block) {
    LLVM_DEBUG(DBGS() << "matching block alternatives");
    return recursiveMatch(first, block, "first") ||
           recursiveMatch(second, block, "second");
  });
}

namespace {
struct WrapBlockForPrinting {
  Block &block;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const WrapBlockForPrinting &wrap) {
  wrap.block.print(os);
  return os;
}
} // end namespace

bool transform_ext::BlockBodyMatcher::match(Block &block) {
  auto debugRAII =
      llvm::make_scope_exit([] { LLVM_DEBUG(DBGS() << "-------\n"); });
  LLVM_DEBUG(DBGS() << "matching: " << WrapBlockForPrinting{block} << "\n");
  if (getCaptured()) {
    LLVM_DEBUG(DBGS() << "found an already captured block: ");
    if (getCaptured() == &block) {
      LLVM_DEBUG(llvm::dbgs() << "same\n");
      return true;
    } else {
      LLVM_DEBUG(llvm::dbgs() << "different\n");
      return false;
    }
  }
  return llvm::all_of(predicates, [&](const PredicateFn &fn) {
    bool result = fn(block);
    LLVM_DEBUG(llvm::dbgs() << ": " << result << "\n");
    return result;
  });
}

transform_ext::BlockBodyMatcher &transform_ext::BlockBodyMatcher::argument(
    transform_ext::NumGreaterEqualTo num) {
  addPredicate([=](Block &block) {
    LLVM_DEBUG(DBGS() << "block has >= " << num.value << " arguments");
    return block.getNumArguments() >= num.value;
  });
  return *this;
}

transform_ext::BlockBodyMatcher &
transform_ext::BlockBodyMatcher::argument(int64_t pos, ValueMatcher &nested) {
  addPredicate([pos, &nested](Block &block) {
    int64_t updatedPos = pos < 0 ? block.getNumArguments() + pos : pos;
    if (updatedPos < 0 || updatedPos >= block.getNumArguments()) {
      LLVM_DEBUG(DBGS() << "matching block argument #" << pos
                        << " that does not exist");
      return false;
    }
    LLVM_DEBUG(DBGS() << "block argument #" << pos << " is\n");
    Value argument = block.getArgument(updatedPos);
    return recursiveMatch(nested, argument);
  });
  recordNestedMatcher(nested);
  return *this;
}

transform_ext::BlockBodyMatcher &
transform_ext::BlockBodyMatcher::operations(transform_ext::NumEqualsTo num) {
  addPredicate([=](Block &block) {
    LLVM_DEBUG(DBGS() << "block has exactly " << num.value << " operations");
    return num.value == std::distance(block.begin(), block.end());
  });
  return *this;
}

transform_ext::BlockBodyMatcher &transform_ext::BlockBodyMatcher::terminator(
    transform_ext::CapturingOpMatcher &nested) {
  addPredicate([&nested](Block &block) {
    Operation *terminator = block.getTerminator();
    LLVM_DEBUG(DBGS() << "has terminator");
    if (!terminator) {
      return false;
    }
    return recursiveMatch(nested, terminator);
  });
  recordNestedMatcher(nested);
  return *this;
}

//===---------------------------------------------------------------------===//
// ValueMatcher
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

bool transform_ext::ValueMatcher::match(Value value) {
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

//===---------------------------------------------------------------------===//
// ValueMatcher
//===---------------------------------------------------------------------===//

std::optional<ValueRange>
transform_ext::CapturingValuePackMatcher::getCaptured() const {
  if (!capturedValuesSet)
    return std::nullopt;

  return capturedValues;
}

bool transform_ext::CapturingValuePackMatcher::match(ValueRange values) {
  if (capturedValuesSet) {
    return llvm::equal(values, capturedValues);
  }

  capturedValuesSet = true;
  capturedValues = llvm::to_vector(values);
  return true;
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
// Body constraints.
//===---------------------------------------------------------------------===//

transform_ext::StructuredOpMatcher &transform_ext::StructuredOpMatcher::body(
    transform_ext::BlockBodyMatcher &nested) {
  addPredicate([&nested](linalg::LinalgOp linalgOp) {
    LLVM_DEBUG(DBGS() << "body block is");
    return recursiveMatch(nested, *linalgOp.getBlock());
  });
  recordNestedMatcher(nested);
  return *this;
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::bodyInputArguments(
    transform_ext::CapturingValuePackMatcher &nested) {
  addPredicate([&nested](linalg::LinalgOp linalgOp) {
    SmallVector<Value> arguments = llvm::to_vector(
        llvm::map_range(linalgOp.getDpsInputOperands(),
                        [&linalgOp](OpOperand *operand) -> Value {
                          return linalgOp.getMatchingBlockArgument(operand);
                        }));
    LLVM_DEBUG(DBGS() << "body block input arguments are");
    return recursiveMatch(nested, arguments);
  });
  recordNestedMatcher(nested);
  return *this;
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

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::input(SmallVector<int64_t> &&positions,
                                          IsProjected dim) {
  addPredicate([=](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "operands ";
               llvm::interleaveComma(positions, llvm::dbgs());
               llvm::dbgs() << " have a permutation maps with " << dim.value
                            << " projected");
    int64_t updatedDim =
        dim.value >= 0 ? dim.value : linalgOp.getNumLoops() + dim.value;
    for (int64_t position : positions) {
      OpOperand *operand = linalgOp.getDpsInputOperand(position);
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
      int64_t updatedPosition =
          position >= 0 ? position : linalgOp.getNumDpsInputs() + position;
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
    int64_t updatedPosition =
        position >= 0 ? position : linalgOp.getNumDpsInputs() + position;
    if (0 < updatedPosition || updatedPosition >= linalgOp.getNumDpsInputs())
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
    int64_t updatedPosition =
        position >= 0 ? position : linalgOp.getNumDpsInputs() + position;
    if (0 < updatedPosition || updatedPosition >= linalgOp.getNumDpsInputs())
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
    int64_t updatedPosition =
        position >= 0 ? position : linalgOp.getNumDpsInputs() + position;
    if (0 > updatedPosition || updatedPosition >= linalgOp.getNumDpsInputs())
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
                      << (optional.value ? " (optional match) " : " ")
                      << "is produced by\n");
    int64_t transformedPosition =
        position >= 0 ? position : linalgOp.getNumDpsInits() + position;
    if (transformedPosition >= linalgOp.getNumDpsInits())
      return false;

    Operation *definingOp =
        linalgOp.getDpsInitOperand(transformedPosition)->get().getDefiningOp();
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
    int64_t updatedDim =
        dim.value >= 0 ? dim.value : linalgOp.getNumLoops() + dim.value;
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
    int64_t updatedPosition =
        position >= 0 ? position : linalgOp.getNumDpsInits() + position;
    if (0 < updatedPosition || updatedPosition >= linalgOp.getNumDpsInits())
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
    int64_t updatedPosition =
        position >= 0 ? position : linalgOp.getNumDpsInits() + position;
    if (0 < updatedPosition || updatedPosition >= linalgOp.getNumDpsInits())
      return false;
    auto shapedType = linalgOp.getDpsInitOperand(updatedPosition)
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
transform_ext::StructuredOpMatcher::output(int64_t position,
                                           SingleCombinerReduction tag) {
  addPredicate([=](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "output operand #" << position
                      << " is populated by a single-combiner reduction");
    int64_t updatedPosition =
        position >= 0 ? position : linalgOp.getNumDpsInits() + position;
    if (0 < updatedPosition || updatedPosition >= linalgOp.getNumDpsInits())
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
                      << (optional.value ? " (optional match) " : " ")
                      << "has a use\n");
    int64_t transformedPosition =
        position >= 0 ? position : linalgOp->getNumResults() + position;
    if (transformedPosition >= linalgOp->getNumResults())
      return false;

    // We MUST run the matcher at this point, even if the match is optional,
    // to allow for capture.
    LLVM_DEBUG(DBGS() << "start recursive match {\n");
    auto debugRAII = llvm::make_scope_exit(
        [] { LLVM_DEBUG(DBGS() << "} end recursive match"); });
    if (llvm::any_of(linalgOp->getResult(transformedPosition).getUsers(),
                     [&matcher](Operation *op) { return matcher(op); })) {
      return true;
    }
    return optional.value;
  });
}

bool transform_ext::StructuredOpMatcher::checkAllTilableMatched(
    Operation *parent, linalg::LinalgOp linalgOp,
    ArrayRef<transform_ext::CapturingMatcherBase *> matchers) {
  LLVM_DEBUG(DBGS() << "all tilable ops captured");
  int64_t numTilableOps = 0;
  if (!parent)
    return false;
  parent->walk([&](TilingInterface Op) { ++numTilableOps; });

  llvm::SmallPtrSet<Operation *, 6> matched;
  for (CapturingMatcherBase *nested : matchers) {
    auto *opMatcher = dyn_cast<CapturingOpMatcher>(nested);
    if (!opMatcher)
      continue;
    if (Operation *captured = opMatcher->getCaptured()) {
      matched.insert(captured);
    }
  }

  // Don't forget to include the root matcher.
  matched.insert(linalgOp);
  return numTilableOps == matched.size();
}

//===-------------------------------------------------------------------===//
// Constraints on op region.
//===-------------------------------------------------------------------===//

/// Configures the `body` block matcher to match a block with one operation
/// feeding exactly one value to the terminator. Returns the matcher of this
/// operation for further chaining.
template <typename OpTy>
static transform_ext::CapturingOpMatcher &
singleOpBody(transform_ext::MatcherContext &matcherContext, size_t numArguments,
             transform_ext::BlockBodyMatcher &body) {
  using namespace transform_ext;

  auto &inner = m_Operation<OpTy>(matcherContext)
                    .operand(NumEqualsTo(numArguments))
                    .result(NumEqualsTo(1));

  auto &terminator =
      m_Operation(matcherContext).operand(NumEqualsTo(1)).operand(0, inner);

  body = body.operations(NumEqualsTo(2))
             .argument(NumGreaterEqualTo(numArguments))
             .terminator(terminator);
  return inner;
}

/// Adds a predicate to `op` that matches its body to be a float reciprocal
/// (1.0/x) computation. Returns the updated `op`.
static transform_ext::StructuredOpMatcher &
floatReciprocalBody(transform_ext::MatcherContext &matcherContext,
                    transform_ext::StructuredOpMatcher &op) {
  using namespace transform_ext;

  BlockBodyMatcher &body = m_Block(matcherContext);
  CapturingOpMatcher &inner =
      singleOpBody<arith::DivFOp>(matcherContext, /*numArguments=*/2, body);

  auto &arg = m_Value(matcherContext);
  body = body.argument(NumGreaterEqualTo(1)).argument(0, arg);
  inner = inner.operand(0, ConstantFloatOne()).operand(1, arg);
  return op.body(body);
}

/// Adds a predicate to `op` that matches its body to just yield its arguments
/// that correspond to structured op inputs.
static transform_ext::StructuredOpMatcher &
passThroughBody(transform_ext::MatcherContext &matcherContext,
                transform_ext::StructuredOpMatcher &op) {
  using namespace transform_ext;

  CapturingValuePackMatcher &args = m_ValuePack(matcherContext);
  CapturingOpMatcher &terminator =
      m_Operation<linalg::YieldOp>(matcherContext).operand(AllOperands(), args);
  return op.bodyInputArguments(args).body(
      m_Block(matcherContext).terminator(terminator));
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
    MatchedReductionCaptures &captures) {
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
  reduction = reduction.result(0, HasAnyUse(), trailing, OptionalMatch())
                  .allTilableOpsCaptured<func::FuncOp>();
  trailingCapture = &trailing;
}

void transform_ext::makeReductionMatcher(transform_ext::MatcherContext &context,
                                         StructuredOpMatcher *&reductionCapture,
                                         MatchedReductionCaptures &captures) {
  StructuredOpMatcher *fill;
  StructuredOpMatcher *leading;
  StructuredOpMatcher *trailing;
  makeReductionMatcher(context, reductionCapture, fill, leading, trailing,
                       captures);
}

/// Adds a predicate to `op` checking that its body has one operation of type
/// OpTy feeding the terminator. The operation has `numArguments` operands that
/// correspond to block arguments and is commutative if indicated so. In the
/// latter case, the block arguments may be swapped in the operand list.
/// Otherwise they are checked to appear in the same order as in the block
/// argument list.
template <typename OpTy>
static transform_ext::StructuredOpMatcher &
singleOpBodyWithCanonicalArgs(transform_ext::MatcherContext &matcherContext,
                              transform_ext::StructuredOpMatcher &op,
                              size_t numArguments, bool isCommutative) {
  using namespace transform_ext;
  assert((!isCommutative || numArguments == 2) &&
         "commutative ops are expected to have 2 arguments");

  BlockBodyMatcher &body = m_Block(matcherContext);
  if (isCommutative) {
    auto &lhs = m_Value(matcherContext);
    auto &rhs = m_Value(matcherContext);

    // It is important for the argument predicates to be listed first here:
    // the nested value matchers will capture the values that will be later
    // checked against in the two alternatives. Otherwise, the values will be
    // captured by the first of the alternatives and fail for the second.
    // TODO: consider explicit indication of when a nested matcher is allowed to
    // capture.
    body = body.argument(0, lhs).argument(1, rhs);
    CapturingOpMatcher &inner =
        singleOpBody<OpTy>(matcherContext, /*numArguments=*/numArguments, body);

    inner = inner.alternatives(
        m_Operation(matcherContext).operand(0, lhs).operand(1, rhs),
        m_Operation(matcherContext).operand(0, rhs).operand(1, lhs));
  } else {
    CapturingOpMatcher &inner =
        singleOpBody<OpTy>(matcherContext, /*numArguments=*/numArguments, body);
    for (size_t i = 0; i < numArguments; ++i) {
      auto &operand = m_Value(matcherContext);
      body = body.argument(i, operand);
      inner = inner.operand(i, operand);
    }
  }

  return op.body(body);
}

/// Match sum(%src, broadcast(%reduction))
static void matchSubBroadcast(transform_ext::MatcherContext &matcherContext,
                              transform_ext::StructuredOpMatcher &maxReduction,
                              transform_ext::ValueMatcher &softmaxSourceOperand,
                              transform_ext::StructuredOpMatcher *&sub) {
  using namespace transform_ext;

  auto &broadcast =
      transform_ext::m_StructuredOp<linalg::GenericOp>(matcherContext)
          .dim(AllDims(), utils::IteratorType::parallel)
          .input(NumEqualsTo(1))
          .input(0, IsProjected(-1))
          .output(NumEqualsTo(1))
          .output(AllOperands(), IsIdentity());
  broadcast = broadcast.input(0, maxReduction);
  broadcast = passThroughBody(matcherContext, broadcast);

  auto &subParallel =
      transform_ext::m_StructuredOp<linalg::GenericOp>(matcherContext)
          .dim(AllDims(), utils::IteratorType::parallel)
          .input(NumEqualsTo(2))
          .input(0, IsIdentity())
          .input(1, IsIdentity())
          .output(NumEqualsTo(1))
          .output(AllOperands(), IsIdentity());
  subParallel = subParallel.input(0, softmaxSourceOperand);
  subParallel = subParallel.input(1, broadcast);
  subParallel = singleOpBodyWithCanonicalArgs<arith::SubFOp>(
      matcherContext, subParallel, /*numArguments=*/2, /*isCommutative=*/false);

  auto &subBroadcast =
      transform_ext::m_StructuredOp<linalg::GenericOp>(matcherContext)
          .dim(AllDims(), utils::IteratorType::parallel)
          .input(NumEqualsTo(2))
          .input(0, IsIdentity())
          .input(1, IsProjected(-1))
          .output(NumEqualsTo(1))
          .output(AllOperands(), IsIdentity());
  subBroadcast = subBroadcast.input(0, softmaxSourceOperand);
  subBroadcast = subBroadcast.input(1, maxReduction);
  subBroadcast = singleOpBodyWithCanonicalArgs<arith::SubFOp>(
      matcherContext, subBroadcast, /*numArguments=*/2,
      /*isCommutative=*/false);
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
          .dim(AllDims(), utils::IteratorType::parallel)
          .input(NumEqualsTo(1))
          .input(0, IsProjected(-1))
          .output(NumEqualsTo(1))
          .output(AllOperands(), IsIdentity());
  broadcast = broadcast.input(0, sum);
  broadcast = passThroughBody(matcherContext, broadcast);

  auto &divNoBroadcast =
      transform_ext::m_StructuredOp<linalg::GenericOp>(matcherContext)
          .dim(AllDims(), utils::IteratorType::parallel)
          .input(NumEqualsTo(2))
          .input(0, IsIdentity())
          .input(1, IsIdentity())
          .output(NumEqualsTo(1))
          .output(AllOperands(), IsIdentity());

  divNoBroadcast = divNoBroadcast.input(0, expOperand);
  divNoBroadcast = divNoBroadcast.input(1, broadcast);
  divNoBroadcast = singleOpBodyWithCanonicalArgs<arith::DivFOp>(
      matcherContext, divNoBroadcast, /*numArguments=*/2,
      /*isCommutative=*/false);

  auto &divBroadcast =
      transform_ext::m_StructuredOp<linalg::GenericOp>(matcherContext)
          .dim(AllDims(), utils::IteratorType::parallel)
          .input(NumEqualsTo(2))
          .input(0, IsIdentity())
          .input(1, IsProjected(-1))
          .output(NumEqualsTo(1))
          .output(AllOperands(), IsIdentity());

  divBroadcast = divBroadcast.input(0, expOperand);
  divBroadcast = divBroadcast.input(1, sum);
  divBroadcast = singleOpBodyWithCanonicalArgs<arith::DivFOp>(
      matcherContext, divBroadcast, /*numArguments=*/2,
      /*isCommutative=*/false);

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
          // Only handle most inner reduction for now.
          .dim(-1, utils::IteratorType::reduction)
          .dim(AllDimsExcept({-1}), utils::IteratorType::parallel)
          .input(NumEqualsTo(1))
          .input(AllOperands(), IsIdentity())
          .output(NumEqualsTo(1))
          .output(AllOperands(), IsProjected(-1));
  maxReduction = maxReduction.input(0, softmaxSourceOperand);
  maxReduction = maxReduction.output(0, fillMinusInf);
  maxReduction = singleOpBodyWithCanonicalArgs<arith::MaxFOp>(
      matcherContext, maxReduction, /*numArguments=*/2, /*isCommutative=*/true);
  maxReductionCapture = &maxReduction;

  transform_ext::StructuredOpMatcher *subOperand;
  matchSubBroadcast(matcherContext, maxReduction, softmaxSourceOperand,
                    subOperand);

  auto &expOperand = m_StructuredOp<linalg::GenericOp>(matcherContext)
                         .dim(AllDims(), utils::IteratorType::parallel)
                         .input(NumEqualsTo(1))
                         .input(AllOperands(), IsIdentity())
                         .output(AllOperands(), IsIdentity())
                         .output(NumEqualsTo(1));
  expOperand = expOperand.input(0, *subOperand);
  expOperand = singleOpBodyWithCanonicalArgs<math::ExpOp>(
      matcherContext, expOperand, /*numArguments=*/1, /*isCommutative=*/false);

  auto &fillZero = m_StructuredOp<linalg::FillOp>(matcherContext)
                       .input(0, ConstantFloatZero());
  auto &sum = m_StructuredOp<linalg::GenericOp>(matcherContext)
                  // Only handle most inner reduction for now.
                  .dim(-1, utils::IteratorType::reduction)
                  .dim(AllDimsExcept({-1}), utils::IteratorType::parallel)
                  .input(NumEqualsTo(1))
                  .input(AllOperands(), IsIdentity())
                  .output(AllOperands(), IsProjected(-1))
                  .output(NumEqualsTo(1));
  sum = sum.input(0, expOperand);
  sum = sum.output(0, fillZero);
  sum = singleOpBodyWithCanonicalArgs<arith::AddFOp>(
      matcherContext, sum, /*numArguments=*/2, /*isCommutative=*/true);

  auto &rcpOperand = m_StructuredOp<linalg::GenericOp>(matcherContext)
                         //  .isFloatReciprocal()
                         .dim(AllDims(), utils::IteratorType::parallel)
                         .input(NumEqualsTo(1))
                         .input(AllOperands(), IsIdentity())
                         .output(AllOperands(), IsIdentity())
                         .output(NumEqualsTo(1));
  rcpOperand = rcpOperand.input(0, sum);
  rcpOperand = floatReciprocalBody(matcherContext, rcpOperand);

  auto &mulOperand =
      transform_ext::m_StructuredOp<linalg::GenericOp>(matcherContext)
          .dim(AllDims(), utils::IteratorType::parallel)
          .input(NumEqualsTo(2))
          .input(0, IsIdentity())
          .input(1, IsProjected(-1))
          .output(NumEqualsTo(1))
          .output(AllOperands(), IsIdentity());

  mulOperand = mulOperand.input(0, expOperand);
  mulOperand = mulOperand.input(1, rcpOperand);
  mulOperand = singleOpBodyWithCanonicalArgs<arith::MulFOp>(
      matcherContext, mulOperand, /*numArguments=*/2, /*isCommutative=*/true);

  transform_ext::StructuredOpMatcher *divOperand;
  matchdivBroadcast(matcherContext, expOperand, sum, divOperand);

  auto &softmaxRoot =
      transform_ext::m_StructuredOp_Or(matcherContext, mulOperand, *divOperand);
  softmaxRootCapture = &softmaxRoot;
}

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform_ext::BlockBodyMatcher);
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform_ext::CapturingOpMatcher);
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform_ext::CapturingValueMatcher);
