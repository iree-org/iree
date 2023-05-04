// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef IREE_COMPILER_DIALECT_FLOW_TRANSFORMS_REGIONOPUTILS_H_
#define IREE_COMPILER_DIALECT_FLOW_TRANSFORMS_REGIONOPUTILS_H_

#include "iree/compiler/Dialect/Flow/Transforms/ConvertRegionToWorkgroups.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class Location;
class OpBuilder;
class Operation;
class RewriterBase;
class Value;

namespace iree_compiler {
namespace IREE {
namespace Flow {
class DispatchRegionOp;

/// For a given operation returns the loop ranges needed to compute the op.
SmallVector<Range> getLoopRanges(Operation *op, Location loc,
                                 OpBuilder &builder);

/// Reify the dynamic dimensions of the given value.
LogicalResult reifyDynamicResultDims(OpBuilder &b, Value value,
                                     SmallVector<Value> &dynamicDims);

/// Append a result to the given DispatchRegionOp. The newly created
/// DispatchRegionOp is returned.
FailureOr<Flow::DispatchRegionOp> appendDispatchRegionResult(
    RewriterBase &rewriter, Flow::DispatchRegionOp regionOp, Value result,
    const SmallVector<Value> &dynamicDims);

/// Create an empty DispatchRegionOp.
Flow::DispatchRegionOp makeEmptyDispatchRegion(OpBuilder &builder, Location loc,
                                               ValueRange workload);

/// Clone a `target` op that is preceding the given dispatch region op into the
/// dispatch region.
///
/// All uses of the target inside of the dispatch region are replaced with the
/// results of the cloned op.
///
/// Example:
///
/// %0 = "some_op"() : () -> (tensor<?xf32>)
/// %r = flow.dispatch.region -> (tensor<?xf32>{%d0}) {
///   %1 = "another_op"(%0) : (tensor<?xf32>) -> (tensor<?xf32>)
///   flow.return %1 : tensor<?xf32>
/// }
/// %2 = "yet_another_use"(%0) : (tensor<?xf32>) -> (tensor<?xf32>)
///
/// Returns the cloned target op.
FailureOr<Operation *> clonePrecedingOpIntoDispatchRegion(
    RewriterBase &rewriter, Operation *target, Flow::DispatchRegionOp regionOp);

/// Move a `target` op that is preceding the given dispatch region op into the
/// dispatch region.
///
/// All uses of the target outside of the dispatch region are replaced with the
/// results of the cloned op.
///
/// Example:
///
/// %0 = "some_op"() : () -> (tensor<?xf32>)
/// %r = flow.dispatch.region -> (tensor<?xf32>{%d0}) {
///   %0_clone = "some_op"() : () -> (tensor<?xf32>)
///   %1 = "another_op"(%0_clone) : (tensor<?xf32>) -> (tensor<?xf32>)
///   flow.return %1 : tensor<?xf32>
/// }
/// %2 = "yet_another_use"(%0) : (tensor<?xf32>) -> (tensor<?xf32>)
FailureOr<Flow::DispatchRegionOp> movePrecedingOpIntoDispatchRegion(
    RewriterBase &rewriter, Operation *target, Flow::DispatchRegionOp regionOp);

/// Wrap the given op in a new dispatch region op.
FailureOr<Flow::DispatchRegionOp> wrapOpInDispatchRegion(RewriterBase &rewriter,
                                                         Operation *op);

/// Decide whether the given op should be cloned and fused into a dispatch
/// region using heuristics.
///
/// Note: This function returns `false` for ops that should be tiled and fused
/// into a dispatch region.
bool isClonableIntoDispatchOp(Operation *op);

/// Clone into the region producers of those value used in the region but
/// defined above, to prepare the dispatch region isolated from above.
LogicalResult cloneProducersToRegion(RewriterBase &rewriter,
                                     Flow::DispatchRegionOp regionOp);

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_FLOW_TRANSFORMS_REGIONOPUTILS_H_
