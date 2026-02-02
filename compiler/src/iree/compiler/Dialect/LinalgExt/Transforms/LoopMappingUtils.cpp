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
      if (failed(composedMap) || (resultMap && composedMap != resultMap)) {
        return failure();
      }
      // Reject mappings that are all zeros.
      if (composedMap->getNumResults() == composedMap->getNumOfZeroResults()) {
        return failure();
      }
      resultMap = *composedMap;
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
      if (failed(composedMap) || (resultMap && composedMap != resultMap)) {
        return failure();
      }
      resultMap = *composedMap;

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
