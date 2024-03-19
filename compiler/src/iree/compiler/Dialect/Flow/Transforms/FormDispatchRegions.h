// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_FLOW_TRANSFORMS_FORMDISPATCHREGIONS_H_
#define IREE_COMPILER_DIALECT_FLOW_TRANSFORMS_FORMDISPATCHREGIONS_H_

#include "iree/compiler/Dialect/Flow/Transforms/ConvertRegionToWorkgroups.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"

namespace mlir {

class Operation;

/// A rewriter that keeps track of all tensor::DimOps.
class TensorDimTrackingRewriter : public IRRewriter, IRRewriter::Listener {
public:
  /// Create a new rewriter: Scan the given op for tensor::DimOps.
  TensorDimTrackingRewriter(Operation *op);
  /// Return all tracked tensor::DimOps.
  SmallVector<tensor::DimOp> getTensorDimOps();

protected:
  void notifyOperationErased(Operation *op) override;
  void notifyOperationInserted(Operation *op, InsertPoint previous) override;

private:
  SmallPtrSet<Operation *, 16> dimOps;
};

} // namespace mlir

namespace mlir::iree_compiler::IREE::Flow {

/// Computes the workload and provides a workload region builder for the given
/// root op.
FailureOr<Flow::WorkloadBuilder> getWorkloadBuilder(OpBuilder &builder,
                                                    Operation *rootOp);

/// Simplfy the given tensor::DimOps as much as possible.
/// * Static dimensions are replaced by constant.
/// * Dynamic dim ops are pushed as much as possible to the top of the function,
///   i.e., if the dim of a value is known to be equal to the dim of a value on
///   the reverse SSA use-def chain, rewrite the value with a dim op of that
///   value.
LogicalResult simplifyDimOps(RewriterBase &rewriter,
                             const SmallVector<tensor::DimOp> &dimOps);

} // namespace mlir::iree_compiler::IREE::Flow

#endif // IREE_COMPILER_DIALECT_FLOW_TRANSFORMS_FORMDISPATCHREGIONS_H_
