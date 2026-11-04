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
//
// The same fold applies through an affine dequantization of the broadcast,
// `batch_matmul(A, dequantize(broadcast(W_q)))`, when the quantization
// parameters do not vary along the broadcast dims: the dequantize is rebuilt
// at the rank of `W_q` and the broadcast is folded into the contraction map in
// the same step, so no broadcast ever sits between the dequantize and the
// contraction and the weights are dequantized once instead of once per batch.

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "llvm/ADT/SmallBitVector.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_FOLDBROADCASTINTOBATCHMATMULPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

using IREE::LinalgExt::DequantizeAffineOp;

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

/// A contraction input whose data movement folds into the contraction map:
/// the contraction reads `source` through `map` (operand indices to source
/// indices) after the fold. When `dequantize` is set, the input is a
/// dequantize of a broadcast of `source`; the fold replaces it with a
/// dequantize of `source` itself with `dequantizeMaps` as indexing maps.
struct FoldableProducer {
  Value source;
  AffineMap map;
  DequantizeAffineOp dequantize;
  SmallVector<AffineMap> dequantizeMaps;
};

/// Matches `value` as `dequantize(broadcast(source))` whose quantization
/// parameters do not vary along the broadcast dims, so that the dequantize can
/// run on `source` at its own rank. The dequantize must have no other user:
/// otherwise the fold would add a second dequantize instead of shrinking the
/// existing one. The broadcast may have other users; they keep it.
static FailureOr<FoldableProducer> matchDequantizedBroadcast(Value value) {
  auto dequantizeOp = value.getDefiningOp<DequantizeAffineOp>();
  if (!dequantizeOp || !dequantizeOp.hasPureTensorSemantics() ||
      !value.hasOneUse()) {
    return failure();
  }
  FailureOr<BroadcastProducer> broadcast =
      matchBroadcastProducer(dequantizeOp.getInput());
  if (failed(broadcast)) {
    return failure();
  }
  // The dequantize's iteration dims to source indices. Its value maps are
  // permutations, so every iteration dim missing from this map is a dim the
  // broadcast introduced.
  AffineMap iterationToSource =
      broadcast->map.compose(dequantizeOp.getInputMap());
  int64_t numLoops = iterationToSource.getNumDims();
  int64_t sourceRank = iterationToSource.getNumResults();
  MLIRContext *context = value.getContext();
  // Each iteration dim replaced by the source dim it reads. Broadcast dims
  // get a placeholder; no rebased map may refer to them.
  SmallVector<AffineExpr> iterationToSourceDim(
      numLoops, getAffineConstantExpr(0, context));
  llvm::SmallBitVector broadcastDims(numLoops, true);
  for (auto [position, expr] :
       llvm::enumerate(iterationToSource.getResults())) {
    unsigned dim = cast<AffineDimExpr>(expr).getPosition();
    iterationToSourceDim[dim] = getAffineDimExpr(position, context);
    broadcastDims.reset(dim);
  }

  // Rebase the parameter maps onto the source dims. A parameter indexed by a
  // broadcast dim genuinely varies across the broadcast, so the dequantized
  // values are not a broadcast of one shared tensor and nothing folds.
  SmallVector<AffineMap> maps = dequantizeOp.getIndexingMapsArray();
  SmallVector<AffineMap> dequantizeMaps;
  AffineMap identity = AffineMap::getMultiDimIdentityMap(sourceRank, context);
  dequantizeMaps.push_back(identity);
  for (AffineMap parameterMap : ArrayRef(maps).drop_front().drop_back()) {
    if (llvm::any_of(broadcastDims.set_bits(), [&](unsigned dim) {
          return parameterMap.isFunctionOfDim(dim);
        })) {
      return failure();
    }
    dequantizeMaps.push_back(parameterMap.replaceDimsAndSymbols(
        iterationToSourceDim, /*symReplacements=*/{}, sourceRank,
        /*numResultSyms=*/0));
  }
  dequantizeMaps.push_back(identity);

  // Result indices to iteration indices, then to source indices.
  AffineMap resultToSource = iterationToSource.compose(
      inversePermutation(dequantizeOp.getOutputMap()));
  return FoldableProducer{broadcast->source, resultToSource, dequantizeOp,
                          std::move(dequantizeMaps)};
}

/// Matches a contraction input whose data movement folds into the contraction
/// map: a broadcast, or a dequantize of a broadcast.
static FailureOr<FoldableProducer> matchFoldableProducer(Value value) {
  FailureOr<BroadcastProducer> broadcast = matchBroadcastProducer(value);
  if (succeeded(broadcast)) {
    return FoldableProducer{broadcast->source, broadcast->map,
                            /*dequantize=*/nullptr, /*dequantizeMaps=*/{}};
  }
  return matchDequantizedBroadcast(value);
}

/// Creates the value the contraction reads after folding `producer`: the
/// broadcast source itself, or a dequantize of it at the source's rank that
/// carries the original dequantize's parameters and attributes verbatim.
static Value materializeFoldedInput(RewriterBase &rewriter,
                                    const FoldableProducer &producer) {
  DequantizeAffineOp dequantizeOp = producer.dequantize;
  if (!dequantizeOp) {
    return producer.source;
  }
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(dequantizeOp);
  Location loc = dequantizeOp.getLoc();
  Type elementType = dequantizeOp.getOutputType().getElementType();
  Value init = tensor::EmptyOp::create(
      rewriter, loc, tensor::getMixedSizes(rewriter, loc, producer.source),
      elementType);
  SmallVector<Value> operands = dequantizeOp->getOperands();
  operands.front() = producer.source;
  operands.back() = init;
  Operation *folded =
      mlir::clone(rewriter, dequantizeOp, TypeRange{init.getType()}, operands);
  rewriter.modifyOpInPlace(folded, [&] {
    cast<DequantizeAffineOp>(folded).setIndexingMapsAttr(
        rewriter.getAffineMapArrayAttr(producer.dequantizeMaps));
  });
  return folded->getResult(0);
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

/// Folds broadcast producers of either batch_matmul input, direct or under a
/// dequantize, into the op's indexing maps. The broadcast is left for its
/// other users, if any.
struct FoldBroadcastIntoBatchMatmul
    : public OpRewritePattern<linalg::BatchMatmulOp> {
  using OpRewritePattern<linalg::BatchMatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BatchMatmulOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }
    SmallVector<AffineMap> maps = op.getIndexingMapsArray();
    SmallVector<std::optional<FoldableProducer>> producers(
        op.getNumDpsInputs());
    bool changed = false;
    bool needsGeneric = false;
    for (auto [idx, input] : llvm::enumerate(op.getDpsInputs())) {
      FailureOr<FoldableProducer> producer = matchFoldableProducer(input);
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
      maps[idx] = foldedMap;
      producers[idx] = std::move(*producer);
      changed = true;
    }
    if (!changed) {
      return failure();
    }
    // Nothing is created before the rewrite is certain, so a failure leaves
    // the IR untouched. Generalization erases `op`, so its inputs are read
    // first.
    SmallVector<Value> inputs(op.getDpsInputs());
    auto materializeInputs = [&]() {
      for (auto [input, producer] : llvm::zip_equal(inputs, producers)) {
        if (producer) {
          input = materializeFoldedInput(rewriter, *producer);
        }
      }
    };
    if (!needsGeneric) {
      materializeInputs();
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
    materializeInputs();
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
