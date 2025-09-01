// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--- FusionUtils.h --- Utility functions used in fusion ---------------===//
//
// Utility functions to decide of ops are fusable or not, etc.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::DispatchCreation {

/// Return true of the producer and consumer of `operand` are fusable
/// using elementwise op fusion transformation.
struct ElementwiseOpsFusabilityOptions {
  // Control fusion with consumer that has multiple reduction dimensions.
  bool fuseMultiReduction = false;
  // Control fusion with producer that is a truncate-like operation.
  bool fuseTruncateOps = false;
};
bool areFusableAsElementwiseOps(MLIRContext *context, OpOperand *operand,
                                ElementwiseOpsFusabilityOptions options);

/// Returns the closest producer dispatch region op result and the chain of
/// operations being looked past during the traversal to find the producer
/// dispatch. Returns std::nullopt if the dispatch or any ops in the chain have
/// multiple uses.
std::optional<std::pair<OpResult, SmallVector<Operation *>>>
getProducerDispatchValueAndOpChain(Value operand);

} // namespace mlir::iree_compiler::DispatchCreation
