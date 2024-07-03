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
} // namespace mlir

namespace mlir::iree_compiler::IREE::Flow {

class DispatchRegionOp;

/// Check if an operation is not null and is not nested within
/// `flow.dispatch.region` or `flow.dispatch.workgroups` op.
bool isNonNullAndOutsideDispatch(Operation *op);
bool isNonNullAndOutsideDispatch(ArrayRef<Operation *> operations);

/// For a given operation returns the loop ranges needed to compute the op.
SmallVector<Range> getLoopRanges(Operation *op, Location loc,
                                 OpBuilder &builder);

/// Reify the dynamic dimensions of the given value, inserting at the back of
/// 'dynamicDims'.
LogicalResult reifyDynamicResultDims(OpBuilder &b, Value value,
                                     SmallVectorImpl<Value> &dynamicDims);

/// Attemps to create optimized expressions for computing every dynamic
/// dimension of 'value'. If successful, 'dynamicDims' contains a value for each
/// dynamic dimension of 'value'. Returns failure otherwise.
LogicalResult
getOptimizedDynamicResultDims(OpBuilder &b, Value value,
                              SmallVectorImpl<Value> &dynamicDims);

/// Append a result to the given DispatchRegionOp. The newly created
/// DispatchRegionOp is returned.
FailureOr<Flow::DispatchRegionOp> appendDispatchRegionResults(
    RewriterBase &rewriter, Flow::DispatchRegionOp regionOp,
    ArrayRef<Value> results, ArrayRef<SmallVector<Value>> dynamicDims);

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
FailureOr<Operation *>
clonePrecedingOpIntoDispatchRegion(RewriterBase &rewriter, Operation *target,
                                   Flow::DispatchRegionOp regionOp);

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
FailureOr<Flow::DispatchRegionOp>
movePrecedingOpsIntoDispatchRegion(RewriterBase &rewriter,
                                   ArrayRef<Operation *> targets,
                                   Flow::DispatchRegionOp regionOp);

/// Move a `target` op that is following the given dispatch region op into the
/// dispatch region.
///
/// Results of the `target` are appended to the yielded results of the dispatch
/// region, and uses of each result are replaced with the corresponding newly
/// yielded result. Operands of the `target` that are produced by the dispatch
/// region are replaced in the cloned op with the corresponding result yielded
/// inside of the dispatch region.
///
/// Example:
///
/// %r = flow.dispatch.region -> (tensor<?xf32>{%d0}) {
///   %0 = "another_op"(%input) : (tensor<?xf32>) -> (tensor<?xf32>)
///   flow.return %0 : tensor<?xf32>
/// }
/// %1 = "some_op"(%r) : () -> (tensor<?xf32>)
/// %2 = "some_op_use"(%1) : (tensor<?xf32>) -> (tensor<?xf32>)
///
/// Becomes:
///
/// %r:2 = flow.dispatch.region -> (tensor<?xf32>{%d0}, tensor<?xf32>{%d1}) {
///   %0 = "another_op"(%input) : (tensor<?xf32>) -> (tensor<?xf32>)
///   %1 = "some_op"(%0) : (tensor<?xf32>) -> (tensor<?xf32>)
///   flow.return %0, %1 : tensor<?xf32>, tensor<?xf32>
/// }
/// %2 = "some_op_use"(%r#1) : (tensor<?xf32>) -> (tensor<?xf32>)
///
FailureOr<IREE::Flow::DispatchRegionOp>
moveFollowingOpIntoDispatchRegion(RewriterBase &rewriter, Operation *target,
                                  IREE::Flow::DispatchRegionOp regionOp);

/// Wrap the given op in a new dispatch region op.
FailureOr<Flow::DispatchRegionOp> wrapOpInDispatchRegion(RewriterBase &rewriter,
                                                         Operation *op);

/// Decide whether the given op should be cloned and fused into a dispatch
/// region using heuristics.
///
/// Note: This function returns `false` for ops that should be tiled and fused
/// into a dispatch region.
bool isClonableIntoDispatchOp(Operation *op);

/// Hoists an operation out of a dispatch region, as long as it does not have
/// producers inside of the dispatch region, or all of its uses are part of
/// the dispatch region op return. If these criteria are not met, then return
/// failure.
///
/// If all producers are defined outside of the dispatch region, then the op
/// will be hoisted above the dispatch region op. Otherwise, the op will be
/// hoisted below the dispatch region op, and the operands of the hoisted op
/// will be added to the yielded values of the dispatch region op.
FailureOr<Operation *> hoistOutOfDispatch(RewriterBase &rewriter,
                                          Operation *op);

/// Collect all ops that should be cloned into the given dispatch region op.
SmallVector<Operation *> getCloneableOps(Flow::DispatchRegionOp regionOp);

/// Clone into the region producers of those value used in the region but
/// defined above, to prepare the dispatch region isolated from above.
LogicalResult cloneProducersToRegion(RewriterBase &rewriter,
                                     Flow::DispatchRegionOp regionOp);

} // namespace mlir::iree_compiler::IREE::Flow

#endif // IREE_COMPILER_DIALECT_FLOW_TRANSFORMS_REGIONOPUTILS_H_
