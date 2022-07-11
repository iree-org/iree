// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_FLOW_TRANSFORMEXTENSIONS_FLOWEXTENSIONS
#define IREE_COMPILER_DIALECT_FLOW_TRANSFORMEXTENSIONS_FLOWEXTENSIONS

include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def ForeachThreadToFlowDispatchWorkgroupsOp : Op<Transform_Dialect, "iree.foreach_thread_to_flow",
    [FunctionalStyleTransformOpTrait, 
     MemoryEffectsOpInterface,
     TransformOpInterface,
     TransformEachOpTrait]> {
  let description = [{
    Rewrite an scf.foreach_thread to Flow::DispatchWorkgroups.
    
    
    Return modes:
    =============
    This operation ignores non-scf::ForeachThread ops and drops them in the 
    return.

    If any rewrite fails, the transform definitely fails.

    If all the operations referred to by the `target` PDLOperation generalize
    properly, the transform succeeds. Otherwise the transform silently fails.

    The return handle points to only the subset of successfully produced 
    equivalent flow::DispatchWorkgroups operations, which can be empty.
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::scf::ForeachThreadOp target, 
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results, 
        ::mlir::transform::TransformState &state);
  }];
}
 
#endif // IREE_COMPILER_DIALECT_FLOW_TRANSFORMEXTENSIONS_FLOWEXTENSIONS
