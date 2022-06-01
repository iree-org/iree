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

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

static llvm::cl::opt<bool> clEnsureInplaceableConsumer(
    "iree-flow-ensure-inplaceable-consumer",
    llvm::cl::desc("Ensure the consumer is inplaceable for fusion."),
    llvm::cl::init(true));

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

bool areLinalgOpsFusableUsingTileAndFuse(OpOperand &use) {
  auto producer = use.get().getDefiningOp<linalg::LinalgOp>();
  auto consumer = dyn_cast<linalg::LinalgOp>(use.getOwner());
  if (!producer || !consumer) return false;

  // 1. Producer has a single result.
  if (producer->getNumResults() != 1) return false;

  // 2. Consumer is elementwise parallel.
  if (consumer.getNumLoops() != consumer.getNumParallelLoops()) return false;

  // 3. In consumer the result of producer is accessed using identity indexing.
  AffineMap consumerIndexingMap = consumer.getTiedIndexingMap(&use);
  if (!consumerIndexingMap.isIdentity()) return false;

  // 4. In-place bufferization requirements (for now) require that the use in
  // the consumer can re-use the buffer for a result.
  return !clEnsureInplaceableConsumer || canInsOperandTieWithOutsOperand(&use);
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
