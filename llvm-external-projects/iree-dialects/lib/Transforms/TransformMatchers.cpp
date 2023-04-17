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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

#define DEBUG_TYPE "transform-matchers"
#define DBGS() llvm::dbgs() << "[" DEBUG_TYPE "] "

//===---------------------------------------------------------------------===//
// CapturingOpMatcher
//===---------------------------------------------------------------------===//

void transform_ext::CapturingOpMatcher::getAllNested(
    SmallVectorImpl<CapturingOpMatcher *> &nested) {
  int64_t start = nested.size();
  llvm::append_range(nested, nestedCapturingMatchers);
  for (int64_t position = start; position < nested.size(); ++position) {
    llvm::append_range(nested, nested[position]->nestedCapturingMatchers);
  }
}

void transform_ext::CapturingOpMatcher::getAllNestedValueMatchers(
    SmallVectorImpl<CapturingValueMatcher *> &nested) {
  llvm::append_range(nested, nestedCapturingValueMatchers);
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
// StructuredOpMatcher and friends.
//===---------------------------------------------------------------------===//

void transform_ext::StructuredOpMatcher::debugOutputForCreate(
    ArrayRef<StringRef> opNames) {
  LLVM_DEBUG(DBGS() << "operation type is one of {";
             llvm::interleaveComma(opNames, llvm::dbgs()); llvm::dbgs() << "}");
}

bool transform_ext::StructuredOpMatcher::match(Operation *op) {
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

  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp) {
    LLVM_DEBUG(DBGS() << "not a structured op\n");
    return false;
  }

  if (!llvm::all_of(predicates, [linalgOp](const PredicateFn &fn) {
        bool result = fn(linalgOp);
        LLVM_DEBUG(llvm::dbgs() << ": " << result << "\n");
        return result;
      })) {
    return false;
  }

  captured = linalgOp;
  return true;
}

//===---------------------------------------------------------------------===//
// Constraints on op rank and dims.
//===---------------------------------------------------------------------===//

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::rank(NumGreaterEqualTo minRank) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "rank >= " << minRank.value);
    return linalgOp.getNumLoops() >= minRank.value;
  });
  return *this;
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::rank(NumLowerEqualTo maxRank) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
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
  predicates.push_back([dimensions = std::move(dimensions),
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
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
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
  predicates.push_back([dimensions = std::move(dimensions),
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
  predicates.push_back([dimensions = std::move(dims),
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
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
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
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "capture rank");
    capture.value = linalgOp.getNumLoops();
    return true;
  });
  return *this;
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::dim(int64_t dimension, CaptureDim capture) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
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
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "capture all dimensions");
    captures.value = linalgOp.getStaticLoopRanges();
    return true;
  });
  return *this;
}

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::convolutionDims(CaptureConvDims convDims) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
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

  predicates.push_back([&A, &B](linalg::LinalgOp linalgOp) -> bool {
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
  predicates.push_back([position, optional, matcher = std::move(matcher)](
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
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
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
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
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
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
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
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
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
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
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
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
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
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
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
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
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
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    LLVM_DEBUG(DBGS() << "input operands " << position
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
  predicates.push_back([position, optional, matcher = std::move(matcher)](
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
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
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
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
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
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
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
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
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
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
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
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
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
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
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
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
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
  predicates.push_back([matcher = std::move(matcher), optional,
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
  matched.insert(linalgOp);
  return numTilableOps == matched.size();
}

//===-------------------------------------------------------------------===//
// Constraints on op region.
//===-------------------------------------------------------------------===//

transform_ext::StructuredOpMatcher &
transform_ext::StructuredOpMatcher::singleOpWithCanonicaleArgs(
    StringRef opcode, bool commutative) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) {
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
  predicates.push_back([=](linalg::LinalgOp linalgOp) {
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
  predicates.push_back([=](linalg::LinalgOp linalgOp) {
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

/// Match sum(%src, broadcast(%reduction))
static void matchSubBroadcat(transform_ext::MatcherContext &matcherContext,
                             transform_ext::StructuredOpMatcher &maxReduction,
                             transform_ext::ValueMatcher &softmaxSourceOperand,
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
  matchSubBroadcat(matcherContext, maxReduction, softmaxSourceOperand,
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
    MatchedConvolutionCaptures &captures) {
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
  convolution = convolution.result(0, HasAnyUse(), trailing, OptionalMatch())
                    .allTilableOpsCaptured<func::FuncOp>();
  trailingCapture = &trailing;
}

void transform_ext::makeConvolutionMatcher(
    transform_ext::MatcherContext &context,
    StructuredOpMatcher *&convolutionCapture,
    MatchedConvolutionCaptures &captures) {
  StructuredOpMatcher *fill;
  StructuredOpMatcher *trailing;
  makeConvolutionMatcher(context, convolutionCapture, fill, trailing, captures);
}
