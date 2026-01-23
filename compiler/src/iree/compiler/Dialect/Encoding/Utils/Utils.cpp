// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/Utils/Utils.h"

#include <variant>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Dominance.h"

#define DEBUG_TYPE "iree-encoding-utils"

namespace mlir::iree_compiler::IREE::Encoding {

SerializableAttr getSerializableAttr(RankedTensorType type) {
  return dyn_cast_if_present<SerializableAttr>(type.getEncoding());
}

EncodingAttr getEncodingAttr(RankedTensorType type) {
  return dyn_cast_if_present<EncodingAttr>(type.getEncoding());
}

bool hasPackedStorageAttr(RankedTensorType type) {
  return dyn_cast_if_present<PackedStorageAttr>(type.getEncoding()) != nullptr;
}

FailureOr<linalg::ContractionDimensions>
getEncodingContractionDims(EncodingAttr encoding) {
  ArrayAttr indexingMapsAttr = encoding.getUserIndexingMaps();
  if (!indexingMapsAttr) {
    return failure();
  }
  // Derive the contraction dims from the first maps in every entry of the
  // `user_indexing_maps` as these contain the layout information about the
  // originally encoded operation.
  SmallVector<AffineMap> indexingMaps = encoding.getRootMaps();
  return linalg::inferContractionDims(indexingMaps);
}

FailureOr<IREE::LinalgExt::ScaledContractionDimensions>
getEncodingScaledContractionDims(EncodingAttr encoding) {
  ArrayAttr indexingMapsAttr = encoding.getUserIndexingMaps();
  if (!indexingMapsAttr) {
    return failure();
  }
  // Derive the contraction dims from the first maps in every entry of the
  // `user_indexing_maps` as these contain the layout information about the
  // originally encoded operation.
  SmallVector<AffineMap> indexingMaps = encoding.getRootMaps();
  return IREE::LinalgExt::inferScaledContractionDims(
      ArrayRef<AffineMap>(indexingMaps));
}

FailureOr<BxMxNxKxKb> getEncodingContractionLikeSizes(EncodingAttr encoding) {
  SmallVector<int64_t> iterationSizes = encoding.getIterationSizesArray();
  if (iterationSizes.empty()) {
    return failure();
  }
  // Try to get scaled contraction dimensions first (which includes kB),
  // otherwise fall back to regular contraction dimensions.
  FailureOr<IREE::LinalgExt::ScaledContractionDimensions> maybeScaledCDims =
      getEncodingScaledContractionDims(encoding);
  IREE::LinalgExt::ScaledContractionDimensions cDims;
  if (succeeded(maybeScaledCDims)) {
    cDims = *maybeScaledCDims;
  } else {
    // Fall back to regular contraction dimensions and convert to scaled format
    // with an empty kB dimension.
    FailureOr<linalg::ContractionDimensions> maybeCDims =
        getEncodingContractionDims(encoding);
    if (failed(maybeCDims)) {
      return failure();
    }
    // Convert ContractionDimensions to ScaledContractionDimensions.
    cDims.m = maybeCDims->m;
    cDims.n = maybeCDims->n;
    cDims.k = maybeCDims->k;
    cDims.kB = {}; // Empty for non-scaled matmuls
    cDims.batch = maybeCDims->batch;
  }
  // The following expects M, N, K, KB, and Batch sizes of at most 1 for now.
  // TODO: Extend this to multiple M/N/K/KB/Batch dims.
  assert(cDims.m.size() <= 1 && cDims.n.size() <= 1 && cDims.k.size() == 1 &&
         cDims.kB.size() <= 1 && cDims.batch.size() <= 1 &&
         "Expected at most one M, N, K, KB, and Batch dimension");
  const int64_t k = iterationSizes[cDims.k[0]];
  // M, N, or Batch can be empty instead of having an explicit dim size of 1
  // for matvec and vecmat, so set to 1 if empty.
  const int64_t m = cDims.m.empty() ? 1 : iterationSizes[cDims.m[0]];
  const int64_t n = cDims.n.empty() ? 1 : iterationSizes[cDims.n[0]];
  const int64_t batch =
      cDims.batch.empty() ? 1 : iterationSizes[cDims.batch[0]];
  // KB can be empty if the encoding is not a scaled matmul.
  const int64_t kb = cDims.kB.empty() ? 1 : iterationSizes[cDims.kB[0]];
  return BxMxNxKxKb{batch, m, n, k, kb};
}

MatmulNarrowDim getPo2MatmulNarrowDim(EncodingAttr encoding) {
  // Get the matmul sizes for the given encoding.
  FailureOr<BxMxNxKxKb> matmulSizes = getEncodingContractionLikeSizes(encoding);
  if (failed(matmulSizes)) {
    return {};
  }
  const int64_t m = matmulSizes->M;
  const int64_t n = matmulSizes->N;

  // If both dimensions are dynamic, return empty.
  if (ShapedType::isDynamic(m) && ShapedType::isDynamic(n)) {
    return {};
  }
  // If only one dimension is dynamic, pick the other as the narrow dimension.
  if (ShapedType::isDynamic(m)) {
    return {MatmulNarrowDim::Dim::N,
            static_cast<int64_t>(llvm::PowerOf2Ceil(n))};
  }
  if (ShapedType::isDynamic(n)) {
    return {MatmulNarrowDim::Dim::M,
            static_cast<int64_t>(llvm::PowerOf2Ceil(m))};
  }
  // If Both dimensions are static, pick the smaller one.
  if (n < m) {
    return {MatmulNarrowDim::Dim::N,
            static_cast<int64_t>(llvm::PowerOf2Ceil(n))};
  }
  if (m < n) {
    return {MatmulNarrowDim::Dim::M,
            static_cast<int64_t>(llvm::PowerOf2Ceil(m))};
  }
  // If dimensions are static and equal, return empty.
  return {};
}

bool isNarrowNResult(EncodingAttr encoding) {
  if (encoding.getOperandIndex().getValue() != IREE::Encoding::MATMUL_RESULT) {
    return false;
  }

  return IREE::Encoding::getPo2MatmulNarrowDim(encoding).isN();
}

namespace {

/// Action types for rematerialization plan.
/// Value already dominates, use as-is.
struct UseExistingAction {
  Value originalValue;
};

/// Create new tensor.dim from newSource.
struct CreateDimOpAction {
  Value originalValue;
  LocationAttr dimLoc;
  Value dimIndex;
};

/// Clone operation with rematerialized operands.
struct CloneOpAction {
  OpResult originalValue;
};

using RematerializationAction =
    std::variant<UseExistingAction, CreateDimOpAction, CloneOpAction>;

/// Helper for std::visit with multiple lambdas (explicit overload set).
template <class... Ts>
struct OverloadedVisit : Ts... {
  using Ts::operator()...;
};
// Deduction guide for aggregate initialization (required in C++17).
template <class... Ts>
OverloadedVisit(Ts...) -> OverloadedVisit<Ts...>;

/// Helper class for analyzing and executing rematerialization of encoding dims.
/// Split into two phases to avoid creating operations on failure:
/// 1. analyze() - builds a plan without creating any operations.
/// 2. apply() - executes the plan and creates operations.
class RematerializationHelper {
public:
  RematerializationHelper(Operation *insertionPoint, Value propagationSource,
                          Value newSource)
      : insertionPoint(insertionPoint), propagationSource(propagationSource),
        newSource(newSource), domInfo(insertionPoint->getParentOp()) {}

  /// Phase 1: Analyze what needs to be rematerialized.
  /// Returns failure if rematerialization is not possible.
  FailureOr<SmallVector<RematerializationAction>> analyze(ValueRange values) {
    SmallVector<RematerializationAction> plan;
    for (Value value : values) {
      if (!analyzeValue(value, plan)) {
        return failure();
      }
    }
    return plan;
  }

  /// Phase 2: Apply the plan and create operations.
  /// Returns rematerialized values only for the requested values.
  SmallVector<Value> apply(RewriterBase &builder,
                           ArrayRef<RematerializationAction> plan,
                           ValueRange requestedValues) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(insertionPoint);

    // Apply all actions to populate the applied map.
    for (const RematerializationAction &action : plan) {
      applyAction(builder, action);
    }

    // Return only the requested values.
    return llvm::map_to_vector(
        requestedValues, [&](Value value) { return applied.lookup(value); });
  }

private:
  /// Analyze a single value, adding actions to the plan.
  /// Returns false if rematerialization is not possible.
  bool analyzeValue(Value value, SmallVector<RematerializationAction> &plan) {
    // Check if already analyzed.
    if (analyzed.contains(value)) {
      return true;
    }

    // If the value already dominates, use it directly.
    if (domInfo.properlyDominates(value, insertionPoint)) {
      analyzed.insert(value);
      plan.push_back(UseExistingAction{value});
      return true;
    }

    Operation *definingOp = value.getDefiningOp();
    if (!definingOp) {
      LDBG() << "Cannot rematerialize: block argument does not dominate";
      return false;
    }

    // Special case: tensor.dim on the propagation source.
    if (auto dimOp = dyn_cast<tensor::DimOp>(definingOp)) {
      if (dimOp.getSource() == propagationSource) {
        analyzed.insert(value);
        plan.push_back(
            CreateDimOpAction{value, dimOp.getLoc(), dimOp.getIndex()});
        return true;
      }
    }

    // General case: pure operations can be cloned.
    if (!isPure(definingOp)) {
      LDBG() << "Cannot rematerialize: operation is not pure";
      return false;
    }

    // Recursively analyze operands first (they must be rematerialized before
    // this op).
    for (Value operand : definingOp->getOperands()) {
      if (!analyzeValue(operand, plan)) {
        LDBG() << "Cannot rematerialize operand: " << operand;
        return false;
      }
    }

    analyzed.insert(value);
    plan.push_back(CloneOpAction{cast<OpResult>(value)});
    return true;
  }

  /// Get the original value from any action type.
  static Value getOriginalValue(const RematerializationAction &action) {
    return std::visit([](const auto &a) -> Value { return a.originalValue; },
                      action);
  }

  /// Apply a single action and return the resulting value.
  Value applyAction(RewriterBase &builder,
                    const RematerializationAction &action) {
    Value originalValue = getOriginalValue(action);

    // Check if already applied.
    if (applied.contains(originalValue)) {
      return applied[originalValue];
    }

    Value result = std::visit(
        OverloadedVisit{
            [&](const UseExistingAction &a) -> Value {
              return a.originalValue;
            },
            [&](const CreateDimOpAction &a) -> Value {
              return tensor::DimOp::create(builder, a.dimLoc, newSource,
                                           a.dimIndex);
            },
            [&](const CloneOpAction &a) -> Value {
              Operation *originalOp = a.originalValue.getOwner();
              Operation *cloned = builder.clone(*originalOp);
              // Update operands to use rematerialized values.
              for (auto [idx, operand] :
                   llvm::enumerate(originalOp->getOperands())) {
                if (applied.contains(operand)) {
                  cloned->setOperand(idx, applied[operand]);
                }
              }
              return cloned->getResult(a.originalValue.getResultNumber());
            },
        },
        action);

    applied[originalValue] = result;
    return result;
  }

  Operation *insertionPoint;
  Value propagationSource;
  Value newSource;
  DominanceInfo domInfo;
  // Tracks values that have already been analyzed.
  DenseSet<Value> analyzed;
  // Maps original value to rematerialized value (for apply phase).
  DenseMap<Value, Value> applied;
};
} // namespace

FailureOr<SmallVector<Value>>
rematerializeEncodingDims(RewriterBase &builder, Operation *insertionPoint,
                          ValueRange encodingDims, Value propagationSource,
                          Value newSource) {
  if (encodingDims.empty()) {
    return SmallVector<Value>{};
  }

  RematerializationHelper helper(insertionPoint, propagationSource, newSource);

  // Phase 1: Analyze without creating operations.
  FailureOr<SmallVector<RematerializationAction>> plan =
      helper.analyze(encodingDims);
  if (failed(plan)) {
    LDBG() << "Cannot rematerialize encoding dims: analysis failed";
    return failure();
  }

  // Phase 2: Apply the plan and create operations.
  return helper.apply(builder, *plan, encodingDims);
}

} // namespace mlir::iree_compiler::IREE::Encoding
