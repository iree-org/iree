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

namespace mlir::iree_compiler::IREE::Flow {

class DispatchRegionOp;

// Populates flow.dispatch.* canonicalization patterns.
void populateFlowDispatchCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context);

// Patterns to fold tensor.extract_slice/insert_slice with
// flow.dispatch.tensor.load/store. These patterns may not be canonicalizers,
// since they might change the parallelism semantics in non-obvious ways.
void populateTensorSliceOpWithDispatchTensorOpFoldingPatterns(
    RewritePatternSet &results, MLIRContext *context);

// Verifies the flow.dispatch.workgroup.size/id/count operations.
LogicalResult verifyDispatchWorkgroupInfoOp(Operation *op, uint64_t dimension);

// Drop unused results of the given dispatch region. Return `true` if the IR
// was modified.
bool dropUnusedDispatchRegionResults(RewriterBase &rewriter,
                                     Flow::DispatchRegionOp regionOp);

} // namespace mlir::iree_compiler::IREE::Flow

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h.inc" // IWYU pragma: export

#endif // IREE_COMPILER_DIALECT_FLOW_IR_FLOWOPS_H_
