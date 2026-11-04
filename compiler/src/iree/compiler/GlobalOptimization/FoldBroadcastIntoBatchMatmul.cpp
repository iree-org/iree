// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Folds a broadcast feeding a `linalg.batch_matmul` input into the op's
// indexing maps. `batch_matmul(A, broadcast(B))` with a weight shared across
// the batch becomes a `batch_matmul` over `A` and `B` whose RHS map simply
// drops the batch dim. The contraction then reads its operands directly, which
// keeps producer patterns such as dequantization visible to later rewrites, and
// no broadcast is materialized. Where the named op cannot express the folded
// map, the contraction is carried by a `linalg.generic` with the same body.

#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_FOLDBROADCASTINTOBATCHMATMULPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

namespace {

/// A producer that only moves data: its result reads `source` through `map`,
/// which takes result indices to source indices and drops at least one dim.
struct BroadcastProducer {
  Value source;
  AffineMap map;
};

/// Matches `value` as the result of a `linalg.broadcast`, or of a generic that
/// only broadcasts (and possibly permutes) its single input.
static FailureOr<BroadcastProducer> matchBroadcastProducer(Value value) {
  Operation *op = value.getDefiningOp();
  if (!op) {
    return failure();
  }
  if (auto broadcastOp = dyn_cast<linalg::BroadcastOp>(op)) {
    return BroadcastProducer{broadcastOp.getInput(),
                             broadcastOp.getIndexingMapsArray()[0]};
  }
  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp || !genericOp.hasPureTensorSemantics() ||
      genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1 ||
      genericOp.getNumParallelLoops() != genericOp.getNumLoops()) {
    return failure();
  }
  Block *body = genericOp.getBody();
  auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());
  if (yieldOp.getNumOperands() != 1 ||
      yieldOp.getOperand(0) != body->getArgument(0)) {
    return failure();
  }
  AffineMap inputMap = genericOp.getIndexingMapsArray()[0];
  AffineMap outputMap = genericOp.getIndexingMapsArray()[1];
  if (!inputMap.isProjectedPermutation() || !outputMap.isPermutation() ||
      inputMap.getNumResults() == inputMap.getNumDims()) {
    return failure();
  }
  // Take result indices to iteration indices, then to input indices.
  return BroadcastProducer{genericOp.getDpsInputs()[0],
                           inputMap.compose(inversePermutation(outputMap))};
}

/// Returns the default map of the given batch_matmul input.
static AffineMap getDefaultInputMap(linalg::BatchMatmulOp op, bool isLhs) {
  SmallVector<AffineMap> defaultMaps =
      linalg::BatchMatmulOp::getDefaultIndexingMaps(op.getContext());
  return defaultMaps[isLhs ? 0 : 1];
}

/// Returns true if `map` still reads a contraction operand: every result is
/// one of the input's default dims and the reduction dim is among them. The
/// batch_matmul loops are (batch, m, n, k).
static bool isContractionInputMap(linalg::BatchMatmulOp op, AffineMap map,
                                  bool isLhs) {
  AffineMap defaultMap = getDefaultInputMap(op, isLhs);
  if (map.getNumResults() > defaultMap.getNumResults() ||
      !map.isFunctionOfDim(/*position=*/3)) {
    return false;
  }
  return llvm::all_of(map.getResults(), [&](AffineExpr expr) {
    return isa<AffineDimExpr>(expr) &&
           llvm::is_contained(defaultMap.getResults(), expr);
  });
}

/// Mirrors the batch_matmul verifier's rules on top of
/// `isContractionInputMap`: a broadcast (fewer results) is limited to the
/// operand orders upstream accepts, and a full-rank map leads with the batch.
static bool isNamedInputMap(linalg::BatchMatmulOp op, AffineMap map,
                            bool isLhs) {
  AffineMap defaultMap = getDefaultInputMap(op, isLhs);
  if (map.getNumResults() < defaultMap.getNumResults()) {
    return op.isValidLhsRhsBroadcastMap(map, isLhs);
  }
  return map.getResult(0).isFunctionOfDim(0);
}

/// Folds broadcast producers of either batch_matmul input into the op's
/// indexing maps. The broadcast is left for its other users, if any.
struct FoldBroadcastIntoBatchMatmul
    : public OpRewritePattern<linalg::BatchMatmulOp> {
  using OpRewritePattern<linalg::BatchMatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BatchMatmulOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }
    SmallVector<AffineMap> maps = op.getIndexingMapsArray();
    SmallVector<Value> inputs(op.getDpsInputs());
    bool changed = false;
    bool needsGeneric = false;
    for (auto [idx, input] : llvm::enumerate(inputs)) {
      FailureOr<BroadcastProducer> producer = matchBroadcastProducer(input);
      if (failed(producer)) {
        continue;
      }
      // Take contraction loops to operand indices, then to source indices.
      bool isLhs = idx == 0;
      AffineMap foldedMap = producer->map.compose(maps[idx]);
      if (!isContractionInputMap(op, foldedMap, isLhs)) {
        continue;
      }
      // The named op accepts a single operand order per broadcast, e.g. a
      // batch-broadcast RHS must be [K, N]. Any other folded map is carried
      // by a generic with the same body, so no data is moved to satisfy the
      // verifier.
      if (!isNamedInputMap(op, foldedMap, isLhs)) {
        needsGeneric = true;
      }
      input = producer->source;
      maps[idx] = foldedMap;
      changed = true;
    }
    if (!changed) {
      return failure();
    }
    if (!needsGeneric) {
      rewriter.modifyOpInPlace(op, [&] {
        op.getInputsMutable().assign(inputs);
        op.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(maps));
      });
      return success();
    }
    FailureOr<linalg::LinalgOp> generalized =
        linalg::generalizeNamedOp(rewriter, op);
    if (failed(generalized)) {
      return failure();
    }
    auto genericOp = cast<linalg::GenericOp>(generalized->getOperation());
    rewriter.modifyOpInPlace(genericOp, [&] {
      genericOp.getInputsMutable().assign(inputs);
      genericOp.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(maps));
    });
    return success();
  }
};

struct FoldBroadcastIntoBatchMatmulPass final
    : impl::FoldBroadcastIntoBatchMatmulPassBase<
          FoldBroadcastIntoBatchMatmulPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<FoldBroadcastIntoBatchMatmul>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::GlobalOptimization
