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

#include <cstdint>

#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::DispatchCreation {

// Maximum number of operands/bindings allowed for a dispatch region.
// This is constrained by the most restrictive GPU backends (CUDA, HIP/ROCm,
// Metal) which all have a limit of 16 bindings. See:
// - IREE_HAL_CUDA_MAX_DISPATCH_BINDING_COUNT = 16
// - IREE_HAL_HIP_MAX_DISPATCH_BINDING_COUNT = 16
constexpr int64_t kIreeMaxOperandCount = 16;

/// Return true of the producer and consumer of `operand` are fusable
/// using elementwise op fusion transformation.
struct ElementwiseOpsFusabilityOptions {
  // Control fusion with consumer that has multiple reduction dimensions.
  bool fuseMultiReduction = false;
  // Control fusion with producer that is a truncate-like operation.
  bool fuseTruncateOps = false;
  // Control fusion with a consumer that is broadcast-like.
  bool fuseBroadcastConsumers = false;
};
bool areFusableAsElementwiseOps(MLIRContext *context, OpOperand *operand,
                                ElementwiseOpsFusabilityOptions options);

/// Returns the closest producer dispatch region op result and the chain of
/// operations being looked past during the traversal to find the producer
/// dispatch. Returns std::nullopt if the dispatch can not be found in the
/// chain or any op in the chain is not a reshape-like op.
std::optional<std::pair<OpResult, SmallVector<Operation *>>>
getProducerDispatchValueAndOpChain(Value operand,
                                   bool enableAggressiveFusion = false);

} // namespace mlir::iree_compiler::DispatchCreation
