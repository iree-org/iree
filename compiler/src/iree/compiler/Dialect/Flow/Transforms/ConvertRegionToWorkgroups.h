// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_FLOW_TRANSFORMS_CONVERTREGIONTOWORKGROUPS_H_
#define IREE_COMPILER_DIALECT_FLOW_TRANSFORMS_CONVERTREGIONTOWORKGROUPS_H_

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class BlockArgument;
class Location;
class OpBuilder;
class RewriterBase;

namespace iree_compiler {
namespace IREE {
namespace Flow {
class DispatchRegionOp;
class DispatchWorkgroupsOp;

/// A data structure that holds workload operands and a function that build
/// the workload region of a WorkgroupsOp.
struct WorkloadBuilder {
  /// A function that builds the workload region of a WorkgroupsOp.
  using RegionBuilderFn =
      std::function<void(OpBuilder &, Location, ArrayRef<BlockArgument>)>;

  /// A function that builds the workload region of a WorkgroupsOp.
  RegionBuilderFn regionBuilder;
};

/// Rewrite the DispatchRegionOp into a DispatchWorkgroupsOp. The
/// DispatchRegionOp is not isolated from above and may capture any SSA value
/// that is in scope. The generated DispatchWorkgroupsOp captures all SSA values
/// explicitly and makes them available inside the region via block arguments.
/// If no WorkloadBuilder is provided, the WorkgroupsOp is constructed without
/// workload operands and without a workload body.
FailureOr<DispatchWorkgroupsOp>
rewriteFlowDispatchRegionToFlowDispatchWorkgroups(DispatchRegionOp regionOp,
                                                  RewriterBase &rewriter);

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_FLOW_TRANSFORMS_CONVERTREGIONTOWORKGROUPS_H_
