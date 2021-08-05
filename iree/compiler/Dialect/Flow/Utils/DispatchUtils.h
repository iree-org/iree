// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/IR/Operation.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

/// Returns the number of maximum parallel dimensions for distribution
int getNumMaxParallelDims();

/// Returns the loops that are partitioned during dispatch region formations, in
/// order, i.e. starting from the outer-most to innermost.
/// Note that this is the same method that is used at the Flow dispatch region
/// formation to tile and distribute the ops.
SmallVector<unsigned> getPartitionedLoops(Operation *op);

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
