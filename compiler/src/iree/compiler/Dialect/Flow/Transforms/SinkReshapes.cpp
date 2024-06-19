// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--- SinkReshapes.cpp --- Pass to sink reshapes -----------------------===//
//
// This pass sinks reshapes (tensor.expand_shape/tensor.collapse_shape) that
// block producer-consumer fusion. These reshapes are generally produced by
// the `BubbleExpandShapes.cpp` pass that propagates reshapes towards the
// arguments, but get blocked on named op.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/FusionUtils.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-flow-sink-reshapes"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_SINKRESHAPESPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

namespace {

class SinkReshapesPass : public impl::SinkReshapesPassBase<SinkReshapesPass> {
public:
  using Base::Base;

  void runOnOperation() override;
};

/// Returns true if two operations are fusable through tile and fuse. Ideally
/// this should use the same method as dispatch region formation where this
/// fusion analysis actually happens, but that requires a direct producer ->
/// consumer relationship and indexing maps for the right analysis. Here
/// we just approximate it (and try to be optimistic)
static bool isFusableUsingTileAndFuse(Operation *producer,
                                      Operation *consumer) {
  if (llvm::isa_and_nonnull<tensor::UnPackOp, Encoding::UnsetEncodingOp>(
          producer)) {
    return true;
  }

  // If the producer is a linalg op.
  auto producerLinalgOp = dyn_cast_or_null<linalg::LinalgOp>(producer);
  if (!producerLinalgOp) {
    return false;
  }
  // Ignore elementwise linalg op producers.
  if (producerLinalgOp.getNumLoops() ==
      producerLinalgOp.getNumParallelLoops()) {
    return false;
  }
  return true;
}

/// Control function to check if an `tensor.expand_shape` (which is producer of
/// `opOperand`) should be pushed past the `genericOp` (which is the consumer of
/// `opOperand`).
static bool shouldSinkExpandShapeOp(OpOperand *opOperand) {
  auto reshapeOp =
      dyn_cast<tensor::ExpandShapeOp>(opOperand->get().getDefiningOp());
  if (!reshapeOp) {
    return false;
  }
  Operation *consumer = opOperand->getOwner();
  if (!isNonNullAndOutsideDispatch({reshapeOp, consumer})) {
    return false;
  }
  auto consumerGenericOp = dyn_cast<linalg::GenericOp>(consumer);
  if (!consumerGenericOp) {
    return false;
  }
  // Only sink across parallel generic ops for now.
  if (consumerGenericOp.getNumParallelLoops() !=
      consumerGenericOp.getNumLoops()) {
    return false;
  }

  // Do not sink reshapes across dequantize operations since tey are
  // cloned into their producers.
  if (isDequantizationLikeOp(consumer)) {
    return false;
  }

  // First check that the expand_shape producer and consumer can be fused.
  Operation *reshapeProducer = reshapeOp.getSrc().getDefiningOp();
  if (!reshapeProducer) {
    return false;
  }
  if (!isFusableUsingTileAndFuse(reshapeOp.getSrc().getDefiningOp(),
                                 consumer)) {
    return false;
  }

  // If the op is already fusable with producer using tile and fuse,
  // do nothing.
  for (OpOperand &opOperand : consumer->getOpOperands()) {
    Operation *currProducer = opOperand.get().getDefiningOp();
    if (!currProducer) {
      continue;
    }

    // The check for the producer having a single use is not fully
    // worked out. Ideally we can fuse with a producer irrespective
    // of number of uses, but is a good thumb rule in practice.
    if (!llvm::hasSingleElement(currProducer->getUses())) {
      continue;
    }

    // Check if a producer can already be tiled and fused with the consumer.
    if (!isFusableUsingTileAndFuse(currProducer, consumer)) {
      continue;
    }

    // There is already a tile-and-fusable producer to fuse with. Still prefer
    // fusing with the producer whose parallel iteration space rank matches
    // the consumer parallel iteration space rank to avoid loss of parallelism.
    if (isa<linalg::LinalgOp>(reshapeProducer) &&
        isa<linalg::LinalgOp>(currProducer)) {
      unsigned currConsumerNumParallelLoops =
          consumerGenericOp.getNumParallelLoops();
      unsigned currProducerNumParallelLoops =
          cast<linalg::LinalgOp>(currProducer).getNumParallelLoops();
      if (currProducerNumParallelLoops == currConsumerNumParallelLoops) {
        // If the producer has same number of parallel loops as consumer,
        // then this is the operand to fuse along. So do nothing.
        return false;
      }
      // If the producer has less number of parallel loops as the consumer,
      // ignore this operand.
      if (currProducerNumParallelLoops < currConsumerNumParallelLoops) {
        continue;
      }
      unsigned reshapeProducerNumParallelLoops =
          cast<linalg::LinalgOp>(reshapeProducer).getNumParallelLoops();
      if (currProducerNumParallelLoops < reshapeProducerNumParallelLoops) {
        return false;
      }
    }
  }
  return true;
}

void SinkReshapesPass::runOnOperation() {
  MLIRContext *context = &getContext();

  RewritePatternSet sinkReshapePatterns(context);
  linalg::populateFoldReshapeOpsByCollapsingPatterns(sinkReshapePatterns,
                                                     shouldSinkExpandShapeOp);
  // Add patterns to fold `tensor.empty` and reshape ops.
  tensor::populateFoldTensorEmptyPatterns(sinkReshapePatterns);
  if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                          std::move(sinkReshapePatterns)))) {
    getOperation()->emitOpError("failed to sink reshape ops");
    return signalPassFailure();
  }
}

} // namespace

} // namespace mlir::iree_compiler::IREE::Flow
