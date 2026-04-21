// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/Transforms/LoopMappingUtils.h"

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

llvm::SmallBitVector getOuterParallelLoops(Operation *op) {
  // Try TilingInterface first - this properly handles linalg ops with mixed
  // parallel/reduction loops by iterating through iterator types.
  if (auto interfaceOp = dyn_cast<TilingInterface>(op)) {
    SmallVector<utils::IteratorType> loopIteratorTypes =
        interfaceOp.getLoopIteratorTypes();
    llvm::SmallBitVector parallelLoops(loopIteratorTypes.size());
    for (auto iteratorType : llvm::enumerate(loopIteratorTypes)) {
      if (iteratorType.value() != utils::IteratorType::parallel) {
        break;
      }
      parallelLoops.set(iteratorType.index());
    }
    return parallelLoops;
  }

  // For ops that implement LinalgFusionOpInterface but not TilingInterface
  // (like SetEncodingOp/UnsetEncodingOp), use the interface to determine
  // parallel loops. These ops have all dimensions as parallel.
  if (auto fusionOp = dyn_cast<LinalgFusionOpInterface>(op)) {
    unsigned numLoops = fusionOp.getNumLoops();
    unsigned numParallelLoops = fusionOp.getNumParallelLoops();
    llvm::SmallBitVector parallelLoops(numLoops);
    for (unsigned i = 0; i < numParallelLoops; ++i) {
      parallelLoops.set(i);
    }
    return parallelLoops;
  }

  // For ops that don't implement either interface, return empty.
  return llvm::SmallBitVector{};
}

/// Combines two composed iteration-space maps from the same reference space
/// into the same op. Broadcast-zero results yield to concrete expressions;
/// returns failure only on a genuine conflict. A null `a` is treated as the
/// identity — merging it with `b` returns `b`.
static FailureOr<AffineMap> mergeComposedMaps(AffineMap a, AffineMap b) {
  if (!a || a == b) {
    return b;
  }
  assert(a.getNumDims() == b.getNumDims() &&
         a.getNumSymbols() == b.getNumSymbols() &&
         a.getNumResults() == b.getNumResults() &&
         "composed maps must share arity");

  auto isZero = [](AffineExpr e) {
    auto c = dyn_cast<AffineConstantExpr>(e);
    return c && c.getValue() == 0;
  };

  SmallVector<AffineExpr> merged;
  merged.reserve(a.getNumResults());
  for (auto [ea, eb] : llvm::zip_equal(a.getResults(), b.getResults())) {
    if (ea == eb || isZero(eb)) {
      merged.push_back(ea);
    } else if (isZero(ea)) {
      merged.push_back(eb);
    } else {
      return failure();
    }
  }
  return AffineMap::get(a.getNumDims(), a.getNumSymbols(), merged,
                        a.getContext());
}

static FailureOr<AffineMap>
computeIterationSpaceMapping(AffineMap producerResultMap,
                             AffineMap consumerOperandMap,
                             AffineMap sourceMap) {
  if (!producerResultMap || !consumerOperandMap ||
      !producerResultMap.isProjectedPermutation() ||
      !consumerOperandMap.isProjectedPermutation()) {
    return failure();
  }

  // `consumerOperandMap` is a mapping from the consumer's iteration space to
  // the operand's data space.
  // `inverseMap` is the mapping from the operand data space to the consumer's
  // iteration space.
  AffineMap inverseMap =
      inverseAndBroadcastProjectedPermutation(consumerOperandMap);

  // `producerResultMap` maps the producer's iteration space to the same
  // operand's data space. Composing the two yields a mapping from the
  // producer's iteration space to the consumer's iteration space.
  AffineMap composedMap = inverseMap.compose(producerResultMap);

  // `sourceMap` is the mapping from some reference space (e.g., root's outer
  // parallel loops) to the producer's iteration space.
  // `composedMap.compose(sourceMap)` computes the mapping from the reference
  // space to the consumer's iteration space.
  return composedMap.compose(sourceMap);
}

FailureOr<AffineMap> getRootParallelLoopToOpMap(
    Operation *candidateOp,
    const llvm::MapVector<Operation *, AffineMap> &loopMaps) {
  auto fusionOp = dyn_cast<LinalgFusionOpInterface>(candidateOp);
  if (!fusionOp) {
    return failure();
  }

  bool isConsumer = llvm::any_of(candidateOp->getOperands(), [&](Value v) {
    return loopMaps.contains(v.getDefiningOp());
  });
  assert(isConsumer != llvm::any_of(candidateOp->getUsers(),
                                    [&](Operation *op) {
                                      return loopMaps.contains(op);
                                    }) &&
         "op must not be both a producer and consumer");

  AffineMap resultMap;
  if (isConsumer) {
    // Compute mapping by examining producer operands.
    for (OpOperand &operand : candidateOp->getOpOperands()) {
      auto producer = operand.get().getDefiningOp<LinalgFusionOpInterface>();
      if (!producer || !loopMaps.contains(producer)) {
        continue;
      }

      AffineMap producerMap =
          producer.getIndexingMapMatchingResult(cast<OpResult>(operand.get()));
      AffineMap consumerMap = fusionOp.getMatchingIndexingMap(&operand);
      AffineMap producerLoopMap = loopMaps.find(producer)->second;

      FailureOr<AffineMap> composedMap = computeIterationSpaceMapping(
          producerMap, consumerMap, producerLoopMap);
      if (failed(composedMap)) {
        return failure();
      }
      // Reject mappings that are all zeros (e.g., affine_map<(d0) -> (0)>).
      // A zero-dimensional map like affine_map<() -> ()> is a valid
      // scalar-to-scalar mapping and should not be rejected.
      if (composedMap->getNumResults() > 0 &&
          composedMap->getNumResults() == composedMap->getNumOfZeroResults()) {
        return failure();
      }
      // Multi-result producers (e.g. OnlineAttentionOp) can feed a single
      // consumer via operands with different ranks, so two composed maps may
      // each be valid while differing on broadcast-only dimensions. Merge them
      // position-wise, letting concrete expressions override zero broadcast
      // placeholders, and fail only on genuine conflicts.
      FailureOr<AffineMap> merged = mergeComposedMaps(resultMap, *composedMap);
      if (failed(merged)) {
        return failure();
      }
      resultMap = *merged;
    }
  } else {
    // Compute mapping by examining consumer uses.
    for (OpOperand &use : candidateOp->getUses()) {
      auto consumer = dyn_cast<LinalgFusionOpInterface>(use.getOwner());
      if (!consumer || !loopMaps.contains(use.getOwner())) {
        continue;
      }

      AffineMap consumerMap = consumer.getMatchingIndexingMap(&use);
      AffineMap producerMap =
          fusionOp.getIndexingMapMatchingResult(cast<OpResult>(use.get()));
      AffineMap consumerLoopMap = loopMaps.find(use.getOwner())->second;

      // Note: For producers, the argument order is swapped compared to
      // consumers.
      FailureOr<AffineMap> composedMap = computeIterationSpaceMapping(
          consumerMap, producerMap, consumerLoopMap);
      if (failed(composedMap)) {
        return failure();
      }
      FailureOr<AffineMap> merged = mergeComposedMaps(resultMap, *composedMap);
      if (failed(merged)) {
        return failure();
      }
      resultMap = *merged;

      // Producers cannot be more parallel than consumers.
      if (compressUnusedDims(resultMap).getNumDims() !=
          consumerLoopMap.getNumDims()) {
        return failure();
      }
    }
  }

  return resultMap ? resultMap : FailureOr<AffineMap>();
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
