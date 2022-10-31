// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_FLOW_TRANSFORMS_DISPATCHREGIONHEURISTIC_H_
#define IREE_COMPILER_DIALECT_FLOW_TRANSFORMS_DISPATCHREGIONHEURISTIC_H_

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/FunctionInterfaces.h"

namespace mlir {
class DominanceInfo;
class Operation;

namespace iree_compiler {
namespace IREE {
namespace Flow {

/// Mapping of root op to other ops that are in the same fusion group.
using FusionGroupMapping =
    llvm::DenseMap<Operation *, llvm::SmallVector<Operation *>>;

/// Return `true` if the given op is the root of a fusion group.
bool isFusionGroupRoot(const FusionGroupMapping &mapping, Operation *op);

/// Return the root op of the fusion group that `op` is contained in. Return
/// `nullptr` if it is not contained in any fusion group.
Operation *getRootOfContainingFusionGroup(const FusionGroupMapping &mapping,
                                          Operation *op);

/// Determine fusion groups.
FusionGroupMapping decideFusableLinalgOps(FunctionOpInterface funcOp,
                                          DominanceInfo const &dominanceInfo,
                                          bool aggressiveFusion);

/// A heuristic that decides which ops should be cloned and fused into a
/// dispatch region.
///
/// Note: This function returns `false` for ops that should be tiled and fused
/// into a dispatch region.
bool isClonableIntoDispatchOp(Operation *op);

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_FLOW_TRANSFORMS_DISPATCHREGIONHEURISTIC_H_
