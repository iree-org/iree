// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <numeric>

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOpUtils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

//===----------------------------------------------------------------------===//
// Dispatch regions
//===----------------------------------------------------------------------===//

namespace {

struct DceDispatchRegion : public OpRewritePattern<DispatchRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DispatchRegionOp op,
                                PatternRewriter &rewriter) const override {
    if (op.body().empty()) return failure();
    ClosureOpDce dce(op, op.body().front(), /*variadicOffset=*/1);
    if (!dce.needsOptimization()) return failure();

    bool newOperation = dce.needsNewOperation();
    if (!newOperation) {
      rewriter.startRootUpdate(op);
      dce.optimize(rewriter);
      rewriter.finalizeRootUpdate(op);
    } else {
      dce.optimize(rewriter, /*eraseOriginal=*/false);
      rewriter.eraseOp(op);
    }
    return success();
  }
};

}  // namespace

void DispatchRegionOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<DceDispatchRegion>(context);
}

//===----------------------------------------------------------------------===//
// Streams
//===----------------------------------------------------------------------===//

namespace {

// Optimizes stream fragment arguments by:
//   - Removing any that are not used in the body
//   - Deduping arguments that refer to the same Value
struct DceStreamFragment : public OpRewritePattern<ExStreamFragmentOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExStreamFragmentOp op,
                                PatternRewriter &rewriter) const override {
    if (op.body().empty()) return failure();
    ClosureOpDce dce(op, op.body().front(), /*variadicOffset=*/0);
    if (!dce.needsOptimization()) return failure();

    bool newOperation = dce.needsNewOperation();
    if (!newOperation) {
      rewriter.startRootUpdate(op);
      dce.optimize(rewriter);
      rewriter.finalizeRootUpdate(op);
    } else {
      dce.optimize(rewriter, /*eraseOriginal=*/false);
      rewriter.eraseOp(op);
    }
    return success();
  }
};

}  // namespace

void ExStreamFragmentOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<DceStreamFragment>(context);
}

//===----------------------------------------------------------------------===//
// Variables
//===----------------------------------------------------------------------===//

namespace {

/// Converts variable initializer functions that evaluate to a constant to a
/// specified initial value.
struct InlineConstVariableOpInitializer : public OpRewritePattern<VariableOp> {
  using OpRewritePattern<VariableOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(VariableOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.initializer()) return failure();
    auto *symbolOp =
        SymbolTable::lookupNearestSymbolFrom(op, op.initializer().getValue());
    auto initializer = cast<FuncOp>(symbolOp);
    if (initializer.getBlocks().size() == 1 &&
        initializer.getBlocks().front().getOperations().size() == 2 &&
        isa<mlir::ReturnOp>(
            initializer.getBlocks().front().getOperations().back())) {
      auto &primaryOp = initializer.getBlocks().front().getOperations().front();
      Attribute constResult;
      if (matchPattern(primaryOp.getResult(0), m_Constant(&constResult))) {
        rewriter.replaceOpWithNewOp<VariableOp>(
            op, op.sym_name(), op.is_mutable(), op.type(), constResult);
        return success();
      }
    }
    return failure();
  }
};

}  // namespace

void VariableOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  results.insert<InlineConstVariableOpInitializer>(context);
}

OpFoldResult VariableLoadOp::fold(ArrayRef<Attribute> operands) {
  auto variableOp = dyn_cast_or_null<VariableOp>(
      SymbolTable::lookupNearestSymbolFrom(*this, variable()));
  if (!variableOp) return {};
  if (variableOp.getAttr("noinline")) {
    // Inlining of the constant has been disabled.
    return {};
  } else if (variableOp.is_mutable()) {
    // We can't inline mutable variables as they may be changed at any time.
    // There may still be other folders/canonicalizers that can help (such as
    // store-forwarding).
    return {};
  } else if (!variableOp.initial_value()) {
    // Uninitialized variables (or those with initializers) can't be folded as
    // we don't yet know the value. InlineConstVariableOpInitializer may help.
    return {};
  }
  return variableOp.initial_value().getValue();
}

namespace {

class PropagateVariableLoadAddress
    : public OpRewritePattern<VariableLoadIndirectOp> {
  using OpRewritePattern::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(VariableLoadIndirectOp op,
                                PatternRewriter &rewriter) const override {
    if (auto addressOp = dyn_cast_or_null<VariableAddressOp>(
            op.variable().getDefiningOp())) {
      rewriter.replaceOpWithNewOp<VariableLoadOp>(op, op.result().getType(),
                                                  addressOp.variable());
      return success();
    }
    return failure();
  }
};

}  // namespace

void VariableLoadIndirectOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<PropagateVariableLoadAddress>(context);
}

namespace {

/// Erases flow.variable.store ops that are no-ops.
/// This can happen if there was a variable load, some DCE'd usage, and a
/// store back to the same variable: we want to be able to elide the entire load
/// and store.
struct EraseUnusedVariableStoreOp : public OpRewritePattern<VariableStoreOp> {
  using OpRewritePattern<VariableStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(VariableStoreOp op,
                                PatternRewriter &rewriter) const override {
    if (auto loadOp =
            dyn_cast_or_null<VariableLoadOp>(op.value().getDefiningOp())) {
      if (loadOp.variable() == op.variable()) {
        rewriter.eraseOp(op);
        return success();
      }
    }
    return failure();
  }
};

}  // namespace

void VariableStoreOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<EraseUnusedVariableStoreOp>(context);
}

namespace {

class PropagateVariableStoreAddress
    : public OpRewritePattern<VariableStoreIndirectOp> {
  using OpRewritePattern::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(VariableStoreIndirectOp op,
                                PatternRewriter &rewriter) const override {
    if (auto addressOp = dyn_cast_or_null<VariableAddressOp>(
            op.variable().getDefiningOp())) {
      rewriter.replaceOpWithNewOp<VariableStoreOp>(op, op.value(),
                                                   addressOp.variable());
      return success();
    }
    return failure();
  }
};

}  // namespace

void VariableStoreIndirectOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<PropagateVariableStoreAddress>(context);
}

//===----------------------------------------------------------------------===//
// Tensor ops
//===----------------------------------------------------------------------===//

/// Reduces the provided multidimensional index into a flattended 1D row-major
/// index. The |type| is expected to be statically shaped (as all constants
/// are).
static uint64_t getFlattenedIndex(ShapedType type, ArrayRef<uint64_t> index) {
  assert(type.hasStaticShape() && "for use on statically shaped types only");
  auto rank = type.getRank();
  auto shape = type.getShape();
  uint64_t valueIndex = 0;
  uint64_t dimMultiplier = 1;
  for (int i = rank - 1; i >= 0; --i) {
    valueIndex += index[i] * dimMultiplier;
    dimMultiplier *= shape[i];
  }
  return valueIndex;
}

OpFoldResult TensorReshapeOp::fold(ArrayRef<Attribute> operands) {
  auto sourceType = source().getType().cast<ShapedType>();
  auto resultType = result().getType().cast<ShapedType>();
  if (sourceType.hasStaticShape() && sourceType == resultType) {
    // No-op.
    return source();
  }

  // Skip intermediate reshapes.
  if (auto definingOp =
          dyn_cast_or_null<TensorReshapeOp>(source().getDefiningOp())) {
    setOperand(definingOp.getOperand());
    return result();
  }

  return {};
}

OpFoldResult TensorLoadOp::fold(ArrayRef<Attribute> operands) {
  if (auto source = operands[0].dyn_cast_or_null<ElementsAttr>()) {
    // Load directly from the constant source tensor.
    auto indices = operands.drop_front();
    if (llvm::count(indices, nullptr) == 0) {
      return source.getValue(
          llvm::to_vector<4>(llvm::map_range(indices, [](Attribute value) {
            return value.cast<IntegerAttr>().getValue().getZExtValue();
          })));
    }
  }
  return {};
}

OpFoldResult TensorStoreOp::fold(ArrayRef<Attribute> operands) {
  if (!operands[0]) return {};
  auto &value = operands[0];
  if (auto target = operands[1].dyn_cast_or_null<ElementsAttr>()) {
    // Store into the constant target tensor.
    if (target.getType().getRank() == 0) {
      return DenseElementsAttr::get(target.getType(), {value});
    }
    auto indices = operands.drop_front(2);
    if (llvm::count(indices, nullptr) == 0) {
      uint64_t offset = getFlattenedIndex(
          target.getType(),
          llvm::to_vector<4>(llvm::map_range(indices, [](Attribute value) {
            return value.cast<IntegerAttr>().getValue().getZExtValue();
          })));
      SmallVector<Attribute, 16> newContents(target.getValues<Attribute>());
      newContents[offset] = value;
      return DenseElementsAttr::get(target.getType(), newContents);
    }
  }
  return {};
}

OpFoldResult TensorSplatOp::fold(ArrayRef<Attribute> operands) {
  // TODO(benvanik): only fold when shape is constant.
  if (operands[0]) {
    // Splat value is constant and we can fold the operation.
    return SplatElementsAttr::get(result().getType().cast<ShapedType>(),
                                  operands[0]);
  }
  return {};
}

OpFoldResult TensorCloneOp::fold(ArrayRef<Attribute> operands) {
  if (operands[0]) {
    return operands[0];
  }
  // TODO(benvanik): fold if clone device placements differ.
  return operand();
}

// Slices tensor from start to (start + length) exclusively at dim.
static ElementsAttr tensorSlice(ElementsAttr tensor, uint64_t dim,
                                uint64_t start, uint64_t length) {
  auto shape = llvm::to_vector<4>(tensor.getType().getShape());
  if (length == shape[dim]) {
    // No need to slice.
    return tensor;
  }
  auto outputShape = shape;
  outputShape[dim] = length;
  auto outputType =
      RankedTensorType::get(outputShape, getElementTypeOrSelf(tensor));
  llvm::SmallVector<Attribute, 4> newContents;
  newContents.reserve(outputType.getNumElements());
  auto valuesBegin = tensor.getValues<Attribute>().begin();
  int64_t step =
      std::accumulate(shape.rbegin(), shape.rbegin() + shape.size() - dim,
                      /*init=*/1, /*op=*/std::multiplies<int64_t>());
  int64_t num = length * step / shape[dim];
  for (int64_t offset = step / shape[dim] * start,
               numElements = tensor.getType().getNumElements();
       offset < numElements; offset += step) {
    newContents.append(valuesBegin + offset, valuesBegin + offset + num);
  }
  return DenseElementsAttr::get(outputType, newContents);
}

OpFoldResult TensorSliceOp::fold(ArrayRef<Attribute> operands) {
  if (llvm::count(operands, nullptr) == 0) {
    // Fully constant arguments so we can perform the slice here.
    auto tensor = operands[0].cast<ElementsAttr>();
    int64_t rank = source().getType().cast<ShapedType>().getRank();
    // start = operands[1:1+rank), and length = operands[1+rank:].
    auto start = llvm::to_vector<4>(llvm::map_range(
        operands.drop_front(1).drop_back(rank), [](Attribute value) {
          return value.cast<IntegerAttr>().getValue().getZExtValue();
        }));
    auto length = llvm::to_vector<4>(
        llvm::map_range(operands.drop_front(1 + rank), [](Attribute value) {
          return value.cast<IntegerAttr>().getValue().getZExtValue();
        }));
    for (int64_t dim = 0; dim < rank; ++dim) {
      tensor = tensorSlice(tensor, dim, start[dim], length[dim]);
    }
    return tensor;
  }
  return {};
}

static ElementsAttr tensorUpdate(ElementsAttr update, ElementsAttr target,
                                 ArrayRef<Attribute> startIndicesAttrs) {
  auto updateType = update.getType().cast<ShapedType>();
  auto targetType = target.getType().cast<ShapedType>();
  // If either target or update has zero element, then no update happens.
  if (updateType.getNumElements() == 0 || targetType.getNumElements() == 0) {
    return target;
  }

  int64_t rank = targetType.getRank();
  // If target is scalar, update is also scalar and is the new content.
  if (rank == 0) {
    return update;
  }

  auto startIndex = llvm::to_vector<4>(
      llvm::map_range(startIndicesAttrs, [](Attribute value) {
        return value.cast<IntegerAttr>().getValue().getZExtValue();
      }));
  auto targetValues = llvm::to_vector<4>(target.getValues<Attribute>());
  // target indices start from startIndicesAttrs and update indices start from
  // all zeros.
  llvm::SmallVector<uint64_t, 4> targetIndex(startIndex);
  llvm::SmallVector<uint64_t, 4> updateIndex(rank, 0);
  int64_t numElements = updateType.getNumElements();
  while (numElements--) {
    targetValues[getFlattenedIndex(targetType, targetIndex)] =
        update.getValue<Attribute>(updateIndex);
    // Increment the index at last dim.
    ++updateIndex.back();
    ++targetIndex.back();
    // If the index in dim j exceeds dim size, reset dim j and
    // increment dim (j-1).
    for (int64_t j = rank - 1;
         j >= 0 && updateIndex[j] >= updateType.getDimSize(j); --j) {
      updateIndex[j] = 0;
      targetIndex[j] = startIndex[j];
      if (j - 1 >= 0) {
        ++updateIndex[j - 1];
        ++targetIndex[j - 1];
      }
    }
  }
  return DenseElementsAttr::get(targetType, targetValues);
}

OpFoldResult TensorUpdateOp::fold(ArrayRef<Attribute> operands) {
  auto indices = operands.drop_front(2);
  bool allIndicesConstant = llvm::count(indices, nullptr) == 0;
  if (operands[0] && operands[1] && allIndicesConstant) {
    // Fully constant arguments so we can perform the update here.
    return tensorUpdate(operands[0].cast<ElementsAttr>(),
                        operands[1].cast<ElementsAttr>(), indices);
  } else {
    // Replace the entire tensor when the sizes match.
    auto updateType = update().getType().cast<ShapedType>();
    auto targetType = target().getType().cast<ShapedType>();
    if (updateType.hasStaticShape() && targetType.hasStaticShape() &&
        updateType == targetType) {
      return update();
    }
  }
  return {};
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
