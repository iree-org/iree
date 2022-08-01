// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_FLOW_IR_FLOWOPS_H_
#define IREE_COMPILER_DIALECT_FLOW_IR_FLOWOPS_H_

#include <cstdint>

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTraits.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/SmallBitVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {
class DispatchRegionOp;

// Returns the `combinedOffsets`, `combinedSizes` and `combinedStrides` to use
// when folding a "producer" **into** a "consumer" op that implement
// `OffsetSizeAndStrideOpInterface`.
// The following computations are performed:
//   - offsets = producer_offsets * consumer_strides + consumer_offsets,
//   - strides = producer_strides * consumer_strides.
LogicalResult foldOffsetsSizesAndStrides(
    OpBuilder &builder, Location loc, ArrayRef<OpFoldResult> producerOffsets,
    ArrayRef<OpFoldResult> producerSizes,
    ArrayRef<OpFoldResult> producerStrides,
    const llvm::SmallBitVector &droppedProducerDims,
    ArrayRef<OpFoldResult> consumerOffsets,
    ArrayRef<OpFoldResult> consumerSizes,
    ArrayRef<OpFoldResult> consumerStrides,
    SmallVector<OpFoldResult> &combinedOffsets,
    SmallVector<OpFoldResult> &combinedSizes,
    SmallVector<OpFoldResult> &combinedStrides);
LogicalResult foldOffsetsSizesAndStrides(
    OpBuilder &builder, Location loc, OffsetSizeAndStrideOpInterface producer,
    OffsetSizeAndStrideOpInterface consumer,
    const llvm::SmallBitVector &droppedProducerDims,
    SmallVector<OpFoldResult> &combinedOffsets,
    SmallVector<OpFoldResult> &combinedSizes,
    SmallVector<OpFoldResult> &combinedStrides);

// Populates flow.dispatch.* canonicalization patterns.
void populateFlowDispatchCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);

// Patterns to fold tensor.extract_slice/insert_slice with
// flow.dispatch.tensor.load/store. These patterns may not be canonicalizers,
// since they might change the parallelism semantics in non-obvious ways.
void populateTensorSliceOpWithDispatchTensorOpFoldingPatterns(
    mlir::RewritePatternSet &results, MLIRContext *context);

// Verifies the flow.dispatch.workgroup.size/id/count operations.
LogicalResult verifyDispatchWorkgroupInfoOp(Operation *op, uint64_t dimension);

// Drop unused results of the given dispatch region. Return `true` if the IR
// was modified.
bool dropUnusedDispatchRegionResults(RewriterBase &rewriter,
                                     Flow::DispatchRegionOp regionOp);

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h.inc"  // IWYU pragma: export

#endif  // IREE_COMPILER_DIALECT_FLOW_IR_FLOWOPS_H_
