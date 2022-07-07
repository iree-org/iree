// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--- FusionUtils.cpp - Utilities that are useful for fusion ----------===//
//
// Defines utility functions and analyses that are useful across passes
// to help with fusion before dispatch region formation.
//
//===---------------------------------------------------------------------===//
#include "iree/compiler/Dialect/Flow/Transforms/FusionUtils.h"

#include "llvm/Support/CommandLine.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

static llvm::cl::opt<bool> clEnsureInplaceableConsumer(
    "iree-flow-ensure-inplaceable-consumer",
    llvm::cl::desc("Ensure the consumer is inplaceable for fusion."),
    llvm::cl::init(true));

static llvm::cl::opt<bool> clFuseReductionBroadcastElementwise(
    "iree-flow-fuse-reduction-broadcast-elementwise",
    llvm::cl::desc("Fuse reduction, broadcast, and elementwise op."),
    llvm::cl::init(false));

/// For the fusion of root op -> elementwise operation to be bufferized
/// in-place without use of extra memory, the result of the root operation
/// must be able to reuse the buffer for the result of the elementwise
/// operation. This is possible if input and output are accessed using the same
/// indexing map.
// TODO: This restriction can go away if we can vectorize always, but that has
// a long tail of tasks.
static bool canInsOperandTieWithOutsOperand(OpOperand *insOperand) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(insOperand->getOwner());
  if (!linalgOp) return false;

  AffineMap insOperandIndexingMap = linalgOp.getTiedIndexingMap(insOperand);

  auto canTieWithOutsOperand = [&](OpOperand *outsOperand) {
    if (linalgOp.getTiedIndexingMap(outsOperand) != insOperandIndexingMap) {
      return false;
    }
    // TODO(#8411): Until ops are vectorized (always), we need
    // to check that the elementtype matches for the operands to be tied.
    // For now just doing this check for convolution ops since we expect
    // contraction ops to be vectorized.
    auto producer = insOperand->get().getDefiningOp();
    if (isa<linalg::GenericOp, linalg::ConvolutionOpInterface>(producer) &&
        insOperand->get().getType().cast<ShapedType>().getElementType() !=
            outsOperand->get().getType().cast<ShapedType>().getElementType()) {
      return false;
    }
    return true;
  };
  return llvm::any_of(linalgOp.getOutputOperands(), canTieWithOutsOperand);
}

/// Checks if a linalg op is a simple reduction of the innermost dimensions
/// with identity map for the input.
static bool isReductionOnInnermostDims(linalg::LinalgOp linalgOp) {
  SmallVector<Operation *, 4> combinerOps;

  if (linalgOp.getNumOutputs() != 1) return false;

  // Check if the result dims are d0, d1, ..., which means the reduction is done
  // in the innermost dimensions.
  auto output = linalgOp.getOutputOperand(0);
  auto outputIndexingMap = linalgOp.getTiedIndexingMap(output);
  for (const auto &en : llvm::enumerate(outputIndexingMap.getResults())) {
    auto expr = en.value();
    if (auto dim = expr.dyn_cast<AffineDimExpr>()) {
      if (dim.getPosition() != en.index()) return false;
    } else {
      return false;
    }
  }

  if (!matchReduction(linalgOp.getRegionOutputArgs(), 0, combinerOps) ||
      combinerOps.size() != 1) {
    return false;
  }

  return true;
}

/// Check if the op is elementwise and the output indexing map is identity.
static bool isSimpleElementwise(linalg::LinalgOp op) {
  if (op.getNumLoops() != op.getNumParallelLoops()) return false;

  if (!allIndexingsAreProjectedPermutation(op)) return false;

  for (OpOperand *opOperand : op.getOutputOperands()) {
    if (!op.getTiedIndexingMap(opOperand).isIdentity()) return false;
  }
  return linalg::hasOnlyScalarElementwiseOp(op->getRegion(0));
}

/// Check if the use of operand is a reduction -> broadcast -> elementwise.
/// An example:
///   s = sum(a: <16x32xf32>) -> <16xf32>
///   d = div(a: <16x32xf32>, broadcast(s to <16x32xf32>)) -> <16x32xf32>
static bool isReductionBroadcastElementwise(OpOperand *operand) {
  auto producer = operand->get().getDefiningOp<linalg::LinalgOp>();
  auto consumer = dyn_cast<linalg::LinalgOp>(operand->getOwner());
  if (!producer || !consumer) return false;

  // TODO: might be able to support dynamic shape if the shape is the same for
  // reduction and broadcast.
  if (producer.hasDynamicShape() || consumer.hasDynamicShape()) return false;

  // Check if the producer is a simple reduction.
  if (!isReductionOnInnermostDims(producer)) return false;

  // Check if the reduction is broadcasted back for the elementwise op.
  auto output = producer.getOutputOperand(0);
  auto outputIndexingMap = producer.getTiedIndexingMap(output);
  auto inputIndexingMap = consumer.getTiedIndexingMap(operand);
  if (outputIndexingMap != inputIndexingMap) return false;

  // Check if the consumer is an elementwise with identity output indexing map.
  if (!isSimpleElementwise(consumer)) return false;

  // Check if the reduction has at least one input with identity indexing map.
  OpOperand *reductionInput = nullptr;
  for (OpOperand *operand : producer.getInputOperands()) {
    auto map = producer.getTiedIndexingMap(operand);
    if (map.isIdentity()) {
      reductionInput = operand;
      break;
    }
  }
  if (!reductionInput) return false;

  // Check the reduction input type is the same as the type of the elementwise
  // output.
  if (reductionInput->get().getType() !=
      consumer.getOutputOperand(0)->get().getType())
    return false;

  return true;
}

bool areLinalgOpsFusableUsingTileAndFuse(OpOperand &use) {
  auto producer = use.get().getDefiningOp<linalg::LinalgOp>();
  auto consumer = dyn_cast<linalg::LinalgOp>(use.getOwner());
  if (!producer || !consumer) return false;

  // 1. Producer has a single result.
  if (producer->getNumResults() != 1) return false;

  // 2. Consumer is elementwise parallel.
  if (consumer.getNumLoops() != consumer.getNumParallelLoops()) return false;

  // 3. Check if a reduction result is used in the following elementwise
  // operation with broadcast. If so, we can fuse the reduction into the
  // elementwise op. The elementwise op on the reduced dimension will be
  // serialized to match the workgroup counts of the fused operations.
  // Otherwise, check if the result of producer is accessed using identity
  // indexing.
  AffineMap consumerIndexingMap = consumer.getTiedIndexingMap(&use);
  if (clFuseReductionBroadcastElementwise &&
      isReductionBroadcastElementwise(&use)) {
    return true;
  } else if (!consumerIndexingMap.isIdentity()) {
    return false;
  }

  // 4. In-place bufferization requirements (for now) require that the use in
  // the consumer can re-use the buffer for a result.
  return !clEnsureInplaceableConsumer || canInsOperandTieWithOutsOperand(&use);
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
