// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/Attributes/Range.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Solver.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/State.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"

using llvm::SmallPtrSet;

namespace mlir {
namespace iree_compiler {
namespace GlobalOptimization {

namespace {

IntegerType deriveIntegerTypeFromRange(MLIRContext *context, int64_t minValue,
                                       int64_t maxValue) {
  // Clamp min/max to span 0.
  const int64_t zero = 0;
  minValue = std::min(zero, minValue);
  maxValue = std::max(zero, maxValue);
  bool isSigned;
  if (minValue < 0) {
    // For signed, make symmetric from -N:N-1
    isSigned = true;
    maxValue = std::max(std::abs(minValue) - 1, maxValue);
    minValue = std::min(-maxValue - 1, minValue);
  } else {
    isSigned = false;
  }
  int64_t n = maxValue - minValue + 1;
  int64_t numBits = std::ceil(std::log2(n));

  return IntegerType::get(context, numBits,
                          isSigned
                              ? IntegerType::SignednessSemantics::Signed
                              : IntegerType::SignednessSemantics::Unsigned);
}

class InferNumericNarrowingPass
    : public InferNumericNarrowingBase<InferNumericNarrowingPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    auto probePoints = collectProbePoints();

    Explorer explorer(getOperation(), TraversalAction::SHALLOW);
    llvm::BumpPtrAllocator allocator;
    DFX::Solver solver(explorer, allocator);

    // Prime with probe points.
    for (Value probePoint : probePoints) {
      solver.getOrCreateElementFor<IREE::Util::FloatRangeValueElement>(
          Position::forValue(probePoint));
    }

    // Solve.
    if (failed(solver.run())) {
      return signalPassFailure();
    }

    // Annotate.
    for (Value probePoint : probePoints) {
      auto *elt = solver.lookupElementFor<IREE::Util::FloatRangeValueElement>(
          Position::forValue(probePoint));
      if (!elt) {
        // Not valid analysis.
        continue;
      }

      applyAnnotation(probePoint, elt->getKnown());
    }
  }

  SmallPtrSet<Value, 8> collectProbePoints() {
    SmallPtrSet<Value, 8> probePoints;
    getOperation()->walk([&](Operation *op) {
      if (auto linalgOp = llvm::dyn_cast<linalg::LinalgOp>(op)) {
        for (Value input : linalgOp.getDpsInputs()) {
          probePoints.insert(input);
        }
        for (Value output : linalgOp.getDpsInits()) {
          probePoints.insert(output);
        }
      }
    });
    return probePoints;
  }

  void applyAnnotation(Value probePoint, IREE::Util::FloatRangeStats stats) {
    if (stats.isTruncated() && stats.isFinite()) {
      // Integer annotation.
      applyIntegerAnnotation(probePoint, stats);
    }
  }

  void applyIntegerAnnotation(Value probePoint,
                              IREE::Util::FloatRangeStats stats) {
    auto context = probePoint.getContext();
    auto minValue = static_cast<int64_t>(stats.minValue);
    auto maxValue = static_cast<int64_t>(stats.maxValue);
    IntegerType type =
        deriveIntegerTypeFromRange(probePoint.getContext(), minValue, maxValue);

    // Insert the annotation.
    OpBuilder builder(context);
    builder.setInsertionPointAfterValue(probePoint);
    std::optional<std::pair<int64_t, int64_t>> range;
    // i0 values cannot parse any values so omit.
    if (type.getWidth() != 0) {
      range = std::make_pair(minValue, maxValue);
    }
    auto annotationOp = builder.create<IREE::Util::NumericOptionalNarrowOp>(
        probePoint.getLoc(), probePoint, type, range);
    probePoint.replaceAllUsesExcept(annotationOp, annotationOp);
  }
};

} // namespace

std::unique_ptr<Pass> createInferNumericNarrowingPass() {
  return std::make_unique<InferNumericNarrowingPass>();
}

} // namespace GlobalOptimization
} // namespace iree_compiler
} // namespace mlir
